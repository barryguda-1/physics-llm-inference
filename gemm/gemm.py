# GEMM (General Matrix-Matrix Multiply) is the workhorse of deep learning:
# C = A @ B with A (M, K), B (K, N) -> C (M, N). Every output element is a dot
# product of a row of A with a column of B: K multiply-adds, so the whole op is
# 2*M*N*K FLOPs against only 2*(MK + KN + MN) bytes of ideal traffic. That n^3
# compute vs n^2 memory split is why big GEMMs are compute-bound: arithmetic
# intensity (FLOPs per byte) grows as n/3 at fp16, and any intensity above the
# machine's ridge point (peak TFLOPS / memory TB/s) means tensor cores, not HBM,
# are the limit.
#
# H100 SXM numbers used below (verified online, Sep 2026):
#   FP16 tensor dense peak   989.5 TFLOPS   <- the real dense ceiling
#   FP16 tensor w/ sparsity  1979 TFLOPS    <- the text's "2000" is this figure
#   HBM3 bandwidth           3.35 TB/s
#   ridge points             989.5/3.35 ~= 295 (dense) · 2000/3.35 ~= 597 (text)
# This script computes every number from the config below and renders the chart.
# Change the config and re-run to see how the picture moves. Label positions
# are tuned to pass check_layout.py (no text-on-text overlaps, no clipping).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

# ----------------------------- config -----------------------------
HIDDEN = 4096          # Llama-7B hidden dim - the "n = 4096" of the text
FFN_DIM = 11008
VOCAB = 32000
BYTES_FP16 = 2         # bytes per element

PEAK_DENSE = 989.5     # TFLOPS, H100 SXM fp16 tensor, dense
PEAK_TEXT = 2000.0     # TFLOPS, the text's figure (= sparse 1979, rounded)
PEAK_SPARSE = 1979.0
BW = 3.35              # TB/s, H100 SXM HBM3
RIDGE_DENSE = PEAK_DENSE / BW
RIDGE_TEXT = PEAK_TEXT / BW

# measured torch.matmul fp16 on H100 SXM, from the text
MEASURED = [(2048, 617.0), (4096, 709.0), (8192, 686.0)]

# ----------------------------- cost model -----------------------------
def gemm_flops(m, n, k):
    return 2 * m * n * k

def gemm_bytes(m, n, k):
    # ideal single-pass traffic: read A, read B, write C, fp16 elements
    return BYTES_FP16 * (m * k + k * n + m * n)

def intensity(m, n, k):
    return gemm_flops(m, n, k) / gemm_bytes(m, n, k)

def roofline(ai):
    return min(PEAK_DENSE, BW * ai)   # attainable TFLOPS at intensity ai

# ----------------------------- sanity checks against the text -----------------------------
assert round(gemm_flops(4096, 4096, 4096) / 1e9, 1) == 137.4      # 2n^3 = 137.4 GFLOP
assert round((2 / 3) * 4096, -2) == 2700                            # ~2700 FLOPs per element
assert round(intensity(4096, 4096, 4096)) == 1365                   # ~1350 FLOPs per byte
assert round(PEAK_TEXT / BW) == 597                                 # text's ridge point
assert round(PEAK_DENSE / BW) == 295
print(f"GEMM 4096^3: {gemm_flops(4096,4096,4096)/1e9:.1f} GFLOP, "
      f"{gemm_bytes(4096,4096,4096)/1e6:.0f} MB, AI {intensity(4096,4096,4096):.0f} FLOP/byte")
print(f"ridge points: dense {RIDGE_DENSE:.0f} · text(2000) {RIDGE_TEXT:.0f} FLOP/byte")
print(f"GEMM becomes compute-bound at n > {3*RIDGE_DENSE:.0f} (dense ridge) / {3*RIDGE_TEXT:.0f} (text ridge)")
for n_, tf in MEASURED:
    print(f"  {n_}^3: {tf:.0f} TFLOPS = {tf/PEAK_DENSE:.0%} of dense peak = {tf/PEAK_TEXT:.0%} of '2000'")

# ----------------------------- style -----------------------------
C_BLUE, C_DBLUE, C_GREEN, C_AMBER, C_RED = "#3b82f6", "#1e40af", "#10b981", "#f59e0b", "#ef4444"
C_TEXT, C_GRID, C_MID, C_CELL = "#0f172a", "#cbd5e1", "#64748b", "#e2e8f0"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
(a_anat, a_ai, a_roof), (a_phase, a_bench, a_txt) = axes
fig.suptitle("GEMM — the workhorse of deep learning · why matrix multiply is compute-bound\n"
             r"$2{\cdot}M{\cdot}N{\cdot}K$ FLOPs vs $2(MK{+}KN{+}MN)$ bytes: at $n{=}4096$ the arithmetic intensity 1365 dwarfs the H100 ridge point"
             " (dense 295 · the text's 597)",
             fontsize=14.5, fontweight="bold", linespacing=1.5)

# ---------------- panel A: anatomy of one output element ----------------
M, K, N, cell = 4, 6, 4, 0.46
ax = a_anat
ax.axis("off")

def draw_grid(ax, x0, y0, rows, cols):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell, cell,
                                   fc="white", ec=C_GRID, lw=1))

# A: (M, K) at x=0.35, B: (K, N) at x=4.55, C: (M, N) at x=7.75; all centered near y=3.4
ax0, bx0, cx0 = 0.35, 4.55, 7.75
ay0 = 3.4 - M * cell / 2          # A and C share M rows -> same y
by0 = 3.4 - K * cell / 2
draw_grid(ax, ax0, ay0, M, K)
draw_grid(ax, bx0, by0, K, N)
draw_grid(ax, cx0, ay0, M, N)

# highlight: row i=1 of A, column j=2 of B, the single output cell C[1, 2]
for c in range(K):
    ax.add_patch(Rectangle((ax0 + c * cell, ay0 + 1 * cell), cell, cell, fc=C_AMBER, ec=C_GRID))
for r in range(K):
    ax.add_patch(Rectangle((bx0 + 2 * cell, by0 + r * cell), cell, cell, fc=C_BLUE, ec=C_GRID))
ax.add_patch(Rectangle((cx0 + 2 * cell, ay0 + 1 * cell), cell, cell, fc=C_GREEN, ec=C_GRID))

ax.text(ax0 + K * cell / 2, 4.6, "A · (M, K)", ha="center", fontsize=11, fontweight="bold")
ax.text(bx0 + N * cell / 2, 5.05, "B · (K, N)", ha="center", fontsize=11, fontweight="bold")
ax.text(cx0 + N * cell / 2, 2.1, "C · (M, N)", ha="center", fontsize=11, fontweight="bold")
ax.text(4.0, 3.4, "@", ha="center", va="center", fontsize=19)
ax.text(7.05, 3.4, "=", ha="center", va="center", fontsize=19)

ax.annotate("row i of A", xy=(ax0 + 0.5 * cell, ay0 + 1.5 * cell), xytext=(0.1, 6.35),
            fontsize=10, fontweight="bold", color="#b45309",
            arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.6))
ax.annotate("column j of B", xy=(bx0 + 2.5 * cell, by0 + 0.4 * cell), xytext=(2.6, 1.55),
            fontsize=10, fontweight="bold", color=C_BLUE,
            arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.6))
ax.annotate("C[i, j] = row i · col j\n= K multiply-adds", xy=(cx0 + 2.5 * cell, ay0 + 2 * cell),
            xytext=(6.3, 6.0), fontsize=9.5, fontweight="bold", color="#047857",
            arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.6))
ax.text(5.1, 0.75, r"$C_{i,j} = \sum_k A_{i,k}\, B_{k,j}$", ha="center", fontsize=11)
ax.text(5.1, 0.18, r"FLOPs $= 2MNK$  ·  bytes $= 2(MK{+}KN{+}MN)$", ha="center", fontsize=10)
ax.text(5.1, 7.25, "every A element is reused N times, every B element M times —\nthat reuse is why GEMM feeds tensor cores, not HBM",
        ha="center", va="top", fontsize=9, color=C_MID, style="italic")
ax.set_xlim(0, 10.1)
ax.set_ylim(0, 7.5)
ax.set_title("A · One output element = one dot product", fontsize=10.5)

# ---------------- panel B: arithmetic intensity vs matrix size ----------------
n = np.geomspace(256, 8192, 200)
ai = n / 3                                  # square fp16 GEMM: (2/3)n per element / 2 bytes
a_ai.loglog(n, ai, color=C_DBLUE, lw=2.5)
a_ai.fill_between(n, ai, 4000, where=ai > RIDGE_TEXT, color=C_GREEN, alpha=0.10, interpolate=True)
a_ai.fill_between(n, ai, 4000, where=(ai > RIDGE_DENSE) & (ai <= RIDGE_TEXT), color=C_AMBER, alpha=0.12, interpolate=True)
a_ai.fill_between(n, 55, ai, where=ai <= RIDGE_DENSE, color=C_RED, alpha=0.08, interpolate=True)
a_ai.axhline(RIDGE_TEXT, color=C_RED, ls="--", lw=1.6)
a_ai.axhline(RIDGE_DENSE, color=C_BLUE, ls="--", lw=1.6)
a_ai.text(7900, RIDGE_TEXT * 1.18, "597 = 2000 / 3.35", fontsize=8.5, color=C_RED,
          fontweight="bold", ha="right", va="bottom")
a_ai.text(7900, RIDGE_DENSE * 0.82, "295 = 990 / 3.35", fontsize=8.5, color=C_BLUE,
          fontweight="bold", ha="right", va="top")
for n_size in (512, 1024, 2048, 4096, 8192):
    a_ai.plot([n_size], [n_size / 3], "o", color=C_DBLUE, ms=5, zorder=5)
    if n_size == 8192:      # last marker: label to the left so it stays in-panel
        a_ai.annotate(f"{round(n_size/3)}", xy=(n_size, n_size / 3),
                      xytext=(n_size * 0.93, n_size / 3 * 1.02), ha="right",
                      fontsize=9, fontweight="bold", color=C_DBLUE)
    elif n_size == 2048:    # keep clear of the 597 ridge label above the line
        a_ai.annotate(f"{round(n_size/3)}", xy=(n_size, n_size / 3),
                      xytext=(n_size * 0.94, n_size / 3 * 1.02), ha="right",
                      fontsize=9, fontweight="bold", color=C_DBLUE)
    else:
        a_ai.annotate(f"{round(n_size/3)}", xy=(n_size, n_size / 3),
                      xytext=(n_size * 1.07, n_size / 3 * 1.09),
                      fontsize=9, fontweight="bold", color=C_DBLUE)
for n_cross, lab in ((3 * RIDGE_DENSE, "n ≈ 886"), (3 * RIDGE_TEXT, "n ≈ 1791")):
    a_ai.axvline(n_cross, color="#94a3b8", ls=":", lw=1.2)
    a_ai.text(n_cross * 1.03, 60, lab, fontsize=8.5, color=C_MID, rotation=90,
              ha="left", va="bottom")
a_ai.text(2400, 2200, "compute-bound\n(tensor cores are the limit)", fontsize=9.5, color="#047857",
          fontweight="bold", ha="center")
a_ai.set_xlim(256, 8192)
a_ai.set_ylim(55, 4000)
a_ai.set_xticks([256, 512, 1024, 2048, 4096, 8192],
                ["256", "512", "1024", "2048", "4096", "8192"])
a_ai.minorticks_off()
a_ai.set_xlabel("square matrix size n (fp16)")
a_ai.set_ylabel("arithmetic intensity (FLOP / byte)")
a_ai.set_title("B · Cost model: 2n³ FLOPs vs 6n² bytes → intensity = n/3\nbig GEMMs clear the ridge, small ones do not", fontsize=10.5)

# ---------------- panel C: the roofline ----------------
xs = np.geomspace(0.08, 60000, 400)
roof = np.minimum(PEAK_DENSE, BW * xs)
a_roof.fill_between(xs, 0.2, roof, color="#f1f5f9", zorder=0)
a_roof.loglog(xs, BW * xs, color=C_RED, lw=2.2)                   # memory roof
a_roof.axhline(PEAK_DENSE, color=C_BLUE, lw=2.2)                  # dense compute roof
a_roof.axhline(PEAK_SPARSE, color="#94a3b8", lw=1.4, ls="--")     # sparse roof
a_roof.text(56000, 1080, f"dense FP16 peak · {PEAK_DENSE:.0f} TFLOPS", fontsize=9.5,
            color=C_BLUE, fontweight="bold", ha="right", va="top")
a_roof.text(0.09, 2350, "with sparsity · 1979 (the text's '2000')", fontsize=8.5, color=C_MID)
a_roof.text(0.6, 4.0, "memory roof\n3.35 TB/s", fontsize=9, color=C_RED, fontweight="bold")
for x_ridge, lab, y_lab, col in ((RIDGE_DENSE, "ridge 295", 3400, C_BLUE),
                                 (RIDGE_TEXT, "ridge 597", 2450, C_RED)):
    a_roof.axvline(x_ridge, color=col, ls=":", lw=1.3)
    a_roof.text(x_ridge * 1.08, y_lab, lab, fontsize=8.5, color=col)

pts = [  # (AI, TFLOPS attained, color, label, (dx, dy) in points, ha)
    (0.25, roofline(0.25), C_RED, "elementwise op · AI 0.25", (5, -12), "left"),
    (1.0, roofline(1.0), C_RED, "decode GEMV (M=1)\n3.4 TFLOPS = 0.3% of peak", (7, -26), "left"),
    (intensity(4096, 4096, 128), roofline(intensity(4096, 4096, 128)),
     C_AMBER, "attention scores 4K ctx", (-8, 10), "right"),
    (341, roofline(341), C_DBLUE, "GEMM 1024³ · AI 341", (-8, -22), "right"),
    (1365, roofline(1365), C_GREEN, "GEMM 4096³ — prefill", (-4, 8), "right"),
    (2731, roofline(2731), C_GREEN, "8192³", (8, -18), "left"),
]
for x, y, col, lab, (dx, dy), ha in pts:
    a_roof.plot([x], [y], "o", color=col, ms=8, zorder=5)
    a_roof.annotate(lab, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color=col, ha=ha)
a_roof.plot([1365], [709], "o", mfc="none", mec="#334155", ms=10, mew=1.6, zorder=5)
a_roof.annotate("measured: 709 (72% of the roof)", xy=(1365, 709), xytext=(-14, -26),
                textcoords="offset points", fontsize=8.5, color="#334155", ha="right")
a_roof.text(0.12, 150, "attainable region:\nmin(peak, BW · AI)", fontsize=8.5, color="#94a3b8", style="italic")
a_roof.set_xlim(0.08, 60000)
a_roof.set_ylim(0.2, 4000)
a_roof.minorticks_off()
a_roof.set_xticks([0.1, 1, 10, 100, 1000, 10000], ["0.1", "1", "10", "100", "1000", "10000"])
a_roof.set_xlabel("arithmetic intensity (FLOP / byte)")
a_roof.set_ylabel("attainable performance (TFLOPS)")
a_roof.set_title("C · H100 roofline: intensity above the ridge → compute-bound\nleft of the ridge you pay bytes, right of it you pay FLOPs", fontsize=10.5)

# ---------------- panel D: prefill vs decode — intensity vs tokens in flight ----------------
M_range = np.geomspace(1, 16384, 400)
curves = [
    (lambda mm: [intensity(m, HIDDEN, HIDDEN) for m in mm], C_BLUE, "QKV/O proj · (M, 4096) @ (4096, 4096)"),
    (lambda mm: [intensity(m, FFN_DIM, HIDDEN) for m in mm], C_AMBER, "FFN gate · (M, 4096) @ (4096, 11008)"),
    (lambda mm: [intensity(m, VOCAB, HIDDEN) for m in mm], C_GREEN, "LM head · (M, 4096) @ (4096, 32000)"),
]
for fn, col, lab in curves:
    a_phase.loglog(M_range, fn(M_range), color=col, lw=2.2, label=lab)
a_phase.axhline(RIDGE_TEXT, color=C_RED, ls="--", lw=1.5)
a_phase.axhline(RIDGE_DENSE, color=C_BLUE, ls="--", lw=1.5)
a_phase.text(1.1, RIDGE_TEXT * 1.10, "597 (text)", fontsize=8.5, color=C_RED, ha="left")
a_phase.text(1.1, RIDGE_DENSE * 0.84, "295 (dense)", fontsize=8.5, color=C_BLUE, ha="left", va="top")
for m_tok, lab, side in ((1, "b=1", "right"), (64, "b=64", "right"),
                         (512, "b=512", "right"), (4096, "prefill 4K", "left")):
    a_phase.axvline(m_tok, color="#94a3b8", ls=":", lw=1)
    if side == "right":
        a_phase.text(m_tok * 1.1, 0.14, lab, fontsize=8, color=C_MID)
    else:   # keep the last label inside the right edge
        a_phase.text(m_tok * 0.92, 0.14, lab, fontsize=8, color=C_MID, ha="right")
    a_phase.plot([m_tok], [intensity(m_tok, HIDDEN, HIDDEN)], "o", color=C_BLUE, ms=5, zorder=5)
a_phase.annotate("AI ≈ 1 — HBM streams\nthe weights per token",
                 xy=(1, intensity(1, HIDDEN, HIDDEN)), xytext=(3.5, 0.42), fontsize=8.5,
                 fontweight="bold", color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
a_phase.annotate("AI 1365–1925:\ncompute-bound", xy=(4096, intensity(4096, FFN_DIM, HIDDEN)),
                 xytext=(150, 5900), fontsize=8.5, fontweight="bold", color="#047857",
                 va="top",
                 arrowprops=dict(arrowstyle="->", color="#047857", lw=1.2))
a_phase.text(1300, 600, "batching lifts decode\nGEMVs into GEMMs",
             fontsize=8.5, color=C_MID, style="italic", ha="left", va="top")
a_phase.set_xlim(1, 16384)
a_phase.set_ylim(0.05, 6000)
a_phase.set_xticks([1, 16, 256, 4096, 16384], ["1", "16", "256", "4096", "16K"])
a_phase.minorticks_off()
a_phase.set_xlabel("tokens in flight M (batch × seq)")
a_phase.set_ylabel("arithmetic intensity (FLOP / byte)")
a_phase.legend(fontsize=8, loc="upper left", framealpha=0.95)
a_phase.set_title("D · Same weights, different M: prefill is compute-bound,\ndecode is memory-bound — M decides", fontsize=10.5)

# ---------------- panel E: measured vs peak ----------------
labels = ["sparse\npeak", "dense\npeak", "2048³", "4096³", "8192³"]
vals = [PEAK_SPARSE, PEAK_DENSE, 617.0, 709.0, 686.0]
cols = ["#cbd5e1", C_DBLUE, "#6ee7b7", C_GREEN, "#34d399"]
bars = a_bench.bar(range(5), vals, width=0.62, color=cols)
for i, v in enumerate(vals):
    note = f"{v:.0f}" if i < 2 else f"{v:.0f}\n{v/PEAK_DENSE:.0%}"
    a_bench.text(i, v + 35, note, ha="center", fontsize=9.5, fontweight="bold")
a_bench.axhline(PEAK_DENSE, color=C_DBLUE, ls=":", lw=1.4)
a_bench.text(4.45, 1035, "dense ceiling", fontsize=8.5, color=C_DBLUE, ha="right")
a_bench.text(2.5, 2200,
             "the text's torch.matmul benchmark (H100 SXM):\n"
             "plateau near 4096 like cuBLAS. Online cuBLAS\n"
             "at 4096³ spans 498–764 TFLOPS → 709 in range.",
             fontsize=8, color=C_TEXT, ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GRID, alpha=0.95))
a_bench.set_xticks(range(5), labels, fontsize=9)
a_bench.set_ylim(0, 2300)
a_bench.set_ylabel("TFLOPS (fp16) · % = share of dense peak")
a_bench.set_title("E · Measured vs peak: 709 at 4096³ is 72% of the dense peak\n— not 35%; the '2000' denominator is the with-sparsity figure", fontsize=10.5)

# ---------------- panel F: the takeaway + fact check ----------------
a_txt.axis("off")
a_txt.grid(False)
rows = [
    ("C = A @ B   A (M,K) · B (K,N) → C (M,N)", 10.5, "bold", C_TEXT, False, 0.08),
    ("FLOPs     = 2·M·N·K      n³ vs n²", 9, "normal", C_DBLUE, True, 0.08),
    ("bytes     = 2·(MK+KN+MN) fp16 one pass", 9, "normal", C_MID, True, 0.08),
    ("intensity = n/3 · at n=4096 → 1365", 9, "normal", C_DBLUE, True, 0.08),
    ("ridge = peak/BW: 295 dense · 597 text", 9, "normal", C_MID, True, 0.095),
    ("GEMM 4096³ : 1365 >> 597 → compute-bound", 9, "bold", "#047857", True, 0.08),
    ("decode M=1 : AI ≈ 1 → 3.4 TFLOPS (0.3%)", 9, "bold", "#b91c1c", True, 0.11),
    ("Prefill is GEMM, GEMM is compute-bound.", 10.5, "bold", C_TEXT, False, 0.06),
    ("Decode streams weights — memory-bound.", 10.5, "bold", C_TEXT, False, 0.11),
]
y = 0.97
for text, size, weight, color, mono, gap in rows:
    a_txt.text(0.02, y, text, fontsize=size, fontweight=weight, color=color, va="top",
               family="monospace" if mono else None)
    y -= gap
a_txt.text(0.02, 0.22,
           "fact check (Sep 2026): '2000 TFLOPS' is the\n"
           "with-sparsity peak — dense FP16 is 989.5, so\n"
           "the measured 709 = 72% utilization, a healthy\n"
           "cuBLAS figure. B200: 2250 TFLOPS, 8 TB/s → 281.",
           fontsize=8.5, color="#92400e", va="top",
           bbox=dict(boxstyle="round,pad=0.5", fc="#fef3c7", ec=C_AMBER, lw=1))
a_txt.set_title("F · The takeaway", fontsize=10.5)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm.png")
fig.savefig(out, dpi=200, facecolor="white")
print(f"saved {out}")
