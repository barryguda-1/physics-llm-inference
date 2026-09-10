# Batching effects: decode is a memory-bound GEMV because every weight byte is
# used for one token. Batching B sequences turns it into a (B,K)x(K,N) GEMM, so
# each weight element fetched feeds B tokens — the arithmetic intensity of the
# layer is exactly 2B FLOPs per weight element, i.e. B FLOPs per byte at fp16.
#
# The chapter's numbers (H100 SXM, fp16 matmul benchmark, model_dim=4096):
#   peak         ~2000 TFLOPS (the table's 1720 TFLOPS = 86% of peak)
#   bandwidth    ~3.35 TB/s   (2000 TFLOPS / ridge 600)
#   ridge point  ~600 FLOPs/byte -> efficiency = B/600 while memory-bound
#   measured     B=1: 3.4 TFLOPS, 10K tok/s, 0.17%   B=32: 27, 80K, 1.35%
#                B=64: 107, 316K, 5.4%               B=128: 430, 1.27M, 21.5%
#                B=512: 1720, 5.1M, 86%
#   latency      B=1: 0.1 ms/token   B=512: 1.3 ms/token (as quoted in the text)
#   KV ceiling   7B on 80 GB: 13 GB weights + 67 GB KV at ~0.5 GB/seq (GQA, 4K)
#                -> max ~134 sequences, still below the ridge's B=600
#
# Online verification (checked Sep 2026, see panel I):
#   nvidia.com/en-us/data-center/h100 — H100 SXM fp16 tensor 1,979 TFLOPS is the
#     *with-sparsity* figure (dense 989.5) over 3.35 TB/s HBM3, 80 GB. The chapter's
#     ~2000 peak / ridge 600 matches the sparse figure; the dense roof gives ridge ~295.
#   docs.vllm.ai/en/latest/configuration/engine_args — vLLM batching knobs:
#     --max-num-seqs (default 128 per vllm/config/scheduler.py), --gpu-memory-utilization
#     (default 0.92), --max-num-batched-tokens, chunked prefill default-on.
#   docs.sglang.io/docs/advanced_features/server_arguments — SGLang batching knobs:
#     --max-running-requests (default None = auto), --mem-fraction-static (auto,
#     0.88 fallback), --chunked-prefill-size (None = auto, -1 disables),
#     --cuda-graph-max-bs-decode (raise alongside the batch), --schedule-conservativeness 1.0.
# Panels G-H add what the chapter stops short of: how production engines (both run
# continuous batching by default) expose increase/decrease-batch controls.
# This script renders the dashboard from the config below; re-run to explore.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ----------------------------- config: H100 SXM as used in the chapter -----------------------------
model_dim = 4096
peak_tflops = 2000.0          # fp16 peak implied by the table (1720 = 86%)
ridge = 600.0                 # ridge point, FLOPs per byte
bw_tbps = peak_tflops / ridge # ~= 3.35 TB/s HBM

batch_meas = np.array([1, 32, 64, 128, 512])
tflops_meas = np.array([3.4, 27.0, 107.0, 430.0, 1720.0])
tok_s_meas = np.array([10e3, 80e3, 316e3, 1.27e6, 5.1e6])

lat_b1_ms, lat_b512_ms = 0.1, 1.3        # per-token latency, quoted in the text
weights_gb, total_gb, kv_per_seq_gb = 13.0, 80.0, 0.5
max_seqs = int((total_gb - weights_gb) / kv_per_seq_gb)   # KV-cache ceiling

eff_meas = tflops_meas / peak_tflops * 100

# ----------------------------- sanity checks against the text -----------------------------
assert np.allclose(eff_meas, [0.17, 1.35, 5.4, 21.5, 86.0], atol=0.06), eff_meas
assert abs(bw_tbps * 512 - 1707) < 15          # measured B=512 hugs the roof
assert max_seqs == 134 and abs(max_seqs / ridge - 0.223) < 0.01
print(f"AI = B FLOPs/byte at fp16 · ridge {ridge:.0f} -> efficiency = B/{ridge:.0f}")
print(f"measured: " + "  ".join(f"B={b}:{t:.0f} TF ({e:.2f}%)"
      for b, t, e in zip(batch_meas, tflops_meas, eff_meas)))
print(f"ceiling: {total_gb:.0f} GB - {weights_gb:.0f} GB weights = "
      f"{total_gb-weights_gb:.0f} GB KV / {kv_per_seq_gb} GB per seq -> {max_seqs} seqs")

# ----------------------------- style -----------------------------
C_BLUE, C_BLUE_D, C_AMBER, C_GREEN, C_RED = "#3b82f6", "#1e40af", "#f59e0b", "#10b981", "#ef4444"
C_TEXT, C_GRID, C_DIM = "#0f172a", "#cbd5e1", "#475569"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": C_DIM, "ytick.color": C_DIM,
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(3, 3, figsize=(17, 14.2), constrained_layout=True)
(a_ai, a_roof, a_scale), (a_trade, a_mem, a_txt), (a_cont, a_knobs, a_src) = axes
fig.suptitle("Batching Effects — amortize the weight stream across B sequences\n"
             "Throughput and efficiency climb with batch · per-token latency climbs with them · KV memory sets the ceiling · SGLang/vLLM turn the knobs",
             fontsize=15, fontweight="bold", linespacing=1.5)

# ---------------- panel A: arithmetic intensity = 2B ----------------
b = np.logspace(0, 3.1, 400)
a_ai.semilogx(b, np.minimum(b / ridge, 1.0) * 100, color=C_BLUE, lw=2.2, zorder=3)
a_ai.axvspan(1, ridge, color=C_BLUE, alpha=0.12, zorder=1)
a_ai.axvspan(ridge, 1300, color=C_AMBER, alpha=0.15, zorder=1)
a_ai.axvline(ridge, color=C_RED, ls="--", lw=1.6)
a_ai.text(540, 48, "ridge = 600", rotation=90, va="center", ha="center",
          fontsize=9.5, color=C_RED, fontweight="bold")
for bm, em, (dx, dy, ha) in [(1, 0.17, (6, 6, "left")), (16, 2.7, (0, 9, "center")),
                             (64, 10.7, (0, 9, "center")), (256, 43, (0, 9, "center")),
                             (600, 100, (0, 9, "center"))]:
    a_ai.plot([bm], [em], "o", ms=6, color=C_BLUE_D, zorder=4)
    a_ai.annotate(f"B={bm}\n{em:g}%", xy=(bm, em), xytext=(dx, dy),
                  textcoords="offset points", ha=ha, va="bottom", fontsize=9,
                  fontweight="bold")
a_ai.text(2.1, 78, "memory-bound\nefficiency = B/600", fontsize=9.5, color=C_BLUE_D,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
a_ai.set_xlim(1, 1300)
a_ai.set_xticks([1, 16, 64, 256, 600, 1024])
a_ai.minorticks_off()
a_ai.set_ylim(0, 118)
a_ai.set_yticks([0, 25, 50, 75, 100])
a_ai.set_xlabel("batch size B (log scale)")
a_ai.set_ylabel("% of peak FLOPs")
a_ai.set_title("A · Intensity = 2B FLOPs per weight element = B FLOPs/byte (fp16)\n"
               "layer (B,K)·(K,N): 2BKN FLOPs vs ~KN weight bytes", fontsize=10.5, linespacing=1.4)

# ---------------- panel B: roofline with the measured points ----------------
ai = np.logspace(-0.15, 3.15, 400)
a_roof.loglog(ai, np.minimum(ai * bw_tbps, peak_tflops), color=C_BLUE, lw=2.4, zorder=3)
a_roof.axvline(ridge, color=C_RED, ls="--", lw=1.6)
a_roof.axhline(peak_tflops, color=C_AMBER, lw=2.2)
a_roof.plot(batch_meas, tflops_meas, "o", ms=7, color=C_BLUE_D,
            mec="white", mew=1.2, zorder=5)
label_off = {1: (7, -3, "left"), 32: (-4, 8, "right"), 64: (7, -2, "left"),
             128: (-6, -4, "right"), 512: (-6, 8, "right")}
for bm, tf, em in zip(batch_meas, tflops_meas, eff_meas):
    dx, dy, ha = label_off[bm]
    a_roof.annotate(f"B={bm} · {em:g}%", xy=(bm, tf), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, fontweight="bold", ha=ha)
a_roof.text(2.6, 13, "memory-bound roof\n3.35 TB/s x AI", fontsize=9.5, color=C_BLUE_D,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
a_roof.text(1.35, 2650, "~2000 TFLOPS fp16", fontsize=9.5, color="#b45309", fontweight="bold")
a_roof.text(1.35, 1450, "1979 w/ 2:4 sparsity; 989.5 dense -> ridge ~295",
            fontsize=8, color=C_DIM)
a_roof.text(700, 1.7, "ridge = 600", fontsize=9.5, color=C_RED, ha="center", fontweight="bold")
a_roof.text(6, 6.5, "B=32-64: below the roof (tile overheads)", fontsize=9, color=C_DIM)
a_roof.set_xlim(0.7, 1600)
a_roof.set_xticks([1, 10, 100, 600, 1000])
a_roof.minorticks_off()
a_roof.set_ylim(1.2, 5000)
a_roof.set_yticks([1, 10, 100, 1000, 2000])
a_roof.set_xlabel("arithmetic intensity (FLOPs/byte) = batch size B")
a_roof.set_ylabel("TFLOPS (log)")
a_roof.set_title("B · H100 roofline: the measured points ride the roof\n"
                 "B=512 nearly saturates; past 600 more batch buys little", fontsize=10.5, linespacing=1.4)

# ---------------- panel C: measured batch scaling ----------------
a_scale.loglog(batch_meas, tok_s_meas, "o-", color=C_GREEN, lw=2.2, ms=7,
               mec="white", mew=1.2, zorder=4)
for bm, ts, em, ha in [(1, 10e3, 0.17, "left"), (32, 80e3, 1.35, "center"),
                       (64, 316e3, 5.4, "center"), (128, 1.27e6, 21.5, "center"),
                       (512, 5.1e6, 86.0, "right")]:
    val = f"{ts/1e3:.0f}K" if ts < 1e6 else f"{ts/1e6:.2f}M"
    a_scale.annotate(f"{val} tok/s · {em:g}%", xy=(bm, ts), xytext=(0, 10),
                     textcoords="offset points", ha=ha, fontsize=9, fontweight="bold")
a_scale.annotate("", xy=(512, 6.5e3), xytext=(512, 5.1e6),
                 arrowprops=dict(arrowstyle="-", color=C_DIM, ls=":", lw=1))
a_scale.text(1980, 6.6e3, "510x throughput: B=1 -> B=512", fontsize=9.5,
             ha="right", color=C_TEXT,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
a_scale.set_xlim(0.7, 2200)
a_scale.set_xticks([1, 16, 64, 256, 512])
a_scale.minorticks_off()
a_scale.set_ylim(5e3, 3e7)
a_scale.set_yticks([1e4, 1e5, 1e6, 1e7], ["10K", "100K", "1M", "10M"])
a_scale.set_xlabel("batch size B (log scale)")
a_scale.set_ylabel("decode throughput (tokens/sec, log)")
a_scale.set_title("C · Measured scaling (H100 SXM, 4096x4096 fp16 matmul)\n"
                  "throughput and efficiency rise together with B", fontsize=10.5, linespacing=1.4)

# ---------------- panel D: the latency-throughput tradeoff ----------------
a_trade.loglog([1e4, 5.1e6], [lat_b1_ms, lat_b512_ms], "--", color=C_DIM, lw=1.6, zorder=2)
a_trade.plot([1e4], [lat_b1_ms], "o", ms=9, color=C_GREEN, mec="white", mew=1.2, zorder=5)
a_trade.plot([5.1e6], [lat_b512_ms], "o", ms=9, color=C_AMBER, mec="white", mew=1.2, zorder=5)
a_trade.annotate("B=1 · 0.1 ms/token\nlow latency, 0.17% of peak", xy=(1e4, lat_b1_ms),
                 xytext=(1.35e4, 0.048), fontsize=9.5, fontweight="bold", color="#047857")
a_trade.annotate("B=512 · 1.3 ms/token\nhigh throughput, 86% of peak", xy=(5.1e6, lat_b512_ms),
                 xytext=(2.5e5, 2.4), fontsize=9.5, fontweight="bold", color="#b45309",
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1))
a_trade.annotate("510x throughput\ncosts 13x latency", xy=(1.5e5, 0.31), xytext=(3.5e5, 0.85),
                 fontsize=9.5, ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color=C_DIM, lw=1))
a_trade.text(1.2e4, 0.5, "interactive chat\nlatency wins", fontsize=9.5, color="#047857")
a_trade.text(2.2e6, 0.16, "offline batch\nthroughput wins", fontsize=9.5, color="#b45309",
             ha="center")
a_trade.set_xlim(5e3, 1.2e7)
a_trade.set_xticks([1e4, 1e5, 1e6, 1e7], ["10K", "100K", "1M", "10M"])
a_trade.minorticks_off()
a_trade.set_ylim(0.04, 3.5)
a_trade.set_yticks([0.05, 0.1, 0.3, 1, 3])
a_trade.set_xlabel("throughput (tokens/sec, log)")
a_trade.set_ylabel("per-token latency (ms, log)")
a_trade.set_title("D · The tradeoff: throughput is bought with latency\n"
                  "every system picks its point on this curve", fontsize=10.5, linespacing=1.4)

# ---------------- panel E: the hard ceiling - KV cache caps the batch ----------------
a_mem.plot([512, 512], [-0.15, 1.78], ls="--", color=C_GREEN, lw=1.6, zorder=2)
a_mem.barh(1, ridge, height=0.5, color=C_AMBER, alpha=0.9, zorder=3)
a_mem.barh(0, max_seqs, height=0.5, color=C_BLUE, zorder=3)
a_mem.grid(visible=False)
a_mem.text(498, 1.5, "largest measured point B=512", fontsize=9, color="#047857",
           ha="right", fontweight="bold")
a_mem.text(max_seqs - 8, 0, "134", ha="right", va="center", fontsize=11,
           fontweight="bold", color="white", zorder=4)
a_mem.text(ridge - 8, 1, "600", ha="right", va="center", fontsize=11,
           fontweight="bold", color="white", zorder=4)
a_mem.annotate("134 seqs -> only ~22% of peak\nbefore memory runs out",
               xy=(134, -0.28), xytext=(210, -0.48), fontsize=9.5, va="top",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
               arrowprops=dict(arrowstyle="->", color=C_DIM, lw=1))
a_mem.text(270, 0.52, f"{total_gb:.0f} GB = {weights_gb:.0f} GB weights + "
           f"{total_gb-weights_gb:.0f} GB KV\n{total_gb-weights_gb:.0f} GB KV / "
           f"{kv_per_seq_gb} GB per seq = {max_seqs}",
           fontsize=9.5, ha="center", zorder=4,
           bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec=C_GRID))
a_mem.set_yticks([0, 1], ["KV-cache ceiling\n(7B, GQA, 4K ctx)", "ridge point\n(100% of peak)"],
                 fontsize=9.5)
a_mem.set_xlim(0, 700)
a_mem.set_ylim(-0.85, 1.85)
a_mem.set_xlabel("batch size (sequences served together)")
a_mem.set_title("E · One 80 GB H100 cannot reach its own ridge\n"
                "memory, not compute, caps the batch", fontsize=10.5, linespacing=1.4)

# ---------------- panel F: the takeaway ----------------
a_txt.axis("off")
a_txt.grid(False)
rows = [
    ("Batching: every weight byte serves B tokens", 10.5, "bold", C_TEXT, False, 0.075),
    ("AI = 2BKN/KN = 2B per element = B per byte", 9.5, "normal", C_BLUE_D, True, 0.075),
    ("H100 ridge ~600 FLOPs/byte; eff = B/600", 9.5, "normal", C_BLUE_D, True, 0.105),
    ("throughput 10K -> 5.1M tok/s (B=1 -> 512)", 9.5, "normal", "#047857", True, 0.075),
    ("per-token latency 0.1 -> 1.3 ms  (13x)", 9.5, "normal", "#b45309", True, 0.075),
    ("small B = responsive · large B = efficient", 9.5, "normal", C_TEXT, True, 0.105),
    ("7B on 80 GB: 13 weights + 67 GB KV cache", 9.5, "normal", C_TEXT, True, 0.075),
    ("GQA, 4K ctx: ~0.5 GB/seq -> ~134 seqs", 9.5, "normal", C_TEXT, True, 0.075),
    ("134 << 600: memory caps batch first", 9.5, "normal", "#b91c1c", True, 0.105),
    ("H100s need enormous batches - production", 10.5, "bold", C_TEXT, False, 0.075),
    ("systems run large batch sizes", 10.5, "bold", C_TEXT, False, 0.0),
]
y = 0.97
for text, size, weight, color, mono, gap in rows:
    a_txt.text(0.02, y, text, fontsize=size, fontweight=weight, color=color, va="top",
               family="monospace" if mono else None)
    y -= gap
a_txt.set_title("F · The takeaway", fontsize=10.5)

# ---------------- panel G: continuous batching (SGLang/vLLM default) ----------------
SEQ_COLORS = {"A": C_BLUE, "B": C_GREEN, "C": C_AMBER, "D": C_RED,
              "E": "#8b5cf6", "F": "#14b8a6", "G": "#ec4899", "H": "#64748b"}
C_IDLE = "#e2e8f0"

def seq_rect(ax, y, start, length, label):
    ax.add_patch(Rectangle((start, y - 0.36), length, 0.72, facecolor=SEQ_COLORS[label],
                           edgecolor="white", lw=1, zorder=3))
    ax.text(start + length / 2, y, label, ha="center", va="center", color="white",
            fontsize=9, fontweight="bold", zorder=4)

def idle_rect(ax, y, start, length):
    ax.add_patch(Rectangle((start, y - 0.36), length, 0.72, facecolor=C_IDLE,
                           edgecolor="white", lw=1, zorder=2))

# static: 4 slots, batch frozen at launch; short seqs strand their slots
for y, seq, ln in [(8, "A", 6), (7, "B", 4), (6, "C", 3), (5, "D", 5)]:
    seq_rect(a_cont, y, 0, ln, seq)
idle_rect(a_cont, 7, 4, 2)   # B done at iter 4
idle_rect(a_cont, 6, 3, 3)   # C done at iter 3
idle_rect(a_cont, 5, 5, 1)   # D done at iter 5
a_cont.text(4.5, 6, "idle", ha="center", va="center", fontsize=8, color="#94a3b8", zorder=4)
# continuous: scheduler refills every iteration from the waiting queue
for y, runs in [(3, [("A", 0, 6), ("G", 6, 2)]), (2, [("B", 0, 4), ("E", 4, 4)]),
                (1, [("C", 0, 3), ("F", 3, 5)]), (0, [("D", 0, 5), ("H", 5, 3)])]:
    for label, st, ln in runs:
        seq_rect(a_cont, y, st, ln, label)
a_cont.axhline(4.6, color=C_GRID, ls="--", lw=1.2)
a_cont.text(0.05, 9.15, "static: fixed batch of 4", fontsize=10, fontweight="bold")
a_cont.text(8.5, 4.9, "75% busy · E-H wait till iter 6", fontsize=9,
            color=C_DIM, ha="right", va="top")
a_cont.text(0.05, 4.15, "continuous: admit per iteration", fontsize=10, fontweight="bold")
a_cont.text(0.05, -0.72, "8 seqs in 8 iters · 100% busy · ~2x the work done", fontsize=9,
            color="#047857", fontweight="bold")
a_cont.set_xlim(0, 8.6)
a_cont.set_ylim(-1.15, 9.55)
a_cont.set_xticks(range(9))
a_cont.set_yticks([])
a_cont.grid(axis="x", color="#eef2f7")
a_cont.grid(axis="y", visible=False)
a_cont.set_xlabel("decode iteration")
a_cont.set_title("G · Continuous batching — the SGLang/vLLM default\n"
                 "B is re-decided every iteration against free KV memory", fontsize=10.5,
                 linespacing=1.4)

# ---------------- panel H: how to raise/lower the batch in practice ----------------
a_knobs.axis("off")
a_knobs.grid(False)
rows = [
    ("turn it up - throughput (offline):", 10, "bold", C_TEXT, False, 0.06),
    ("vLLM   --max-num-seqs 512  (def 128)", 9.5, "normal", C_TEXT, True, 0.055),
    ("SGLang --max-running-requests 512", 9.5, "normal", C_TEXT, True, 0.055),
    ("SGLang --cuda-graph-max-bs-decode N", 9.5, "normal", C_TEXT, True, 0.09),
    ("make room - KV cache holds B x ctx:", 10, "bold", C_TEXT, False, 0.06),
    ("vLLM   --gpu-memory-utilization 0.92", 9.5, "normal", C_TEXT, True, 0.055),
    ("SGLang --mem-fraction-static 0.88", 9.5, "normal", C_TEXT, True, 0.09),
    ("keep decode fast while prefilling:", 10, "bold", C_TEXT, False, 0.06),
    ("vLLM   --max-num-batched-tokens N", 9.5, "normal", C_TEXT, True, 0.055),
    ("SGLang --chunked-prefill-size N", 9.5, "normal", C_TEXT, True, 0.09),
    ("turn it down - latency (interactive):", 10, "bold", C_TEXT, False, 0.06),
    ("low caps · smaller ctx · fewer seqs", 9.5, "normal", C_TEXT, True, 0.055),
    ("SGLang --schedule-conservativeness 1.0", 9.5, "normal", C_TEXT, True, 0.07),
    ("both engines: continuous batching default", 10, "bold", "#047857", False, 0.0),
]
y = 0.97
for text, size, weight, color, mono, gap in rows:
    a_knobs.text(0.02, y, text, fontsize=size, fontweight=weight, color=color, va="top",
                 family="monospace" if mono else None)
    y -= gap
a_knobs.set_title("H · Turning the batch up or down — SGLang & vLLM", fontsize=10.5)

# ---------------- panel I: sources ----------------
a_src.axis("off")
a_src.grid(False)
rows = [
    ("NVIDIA H100 SXM (product page):", 10, "bold", C_TEXT, False, 0.055),
    ("fp16 1,979 TFLOPS w/ 2:4 sparsity", 9, "normal", C_BLUE_D, True, 0.05),
    ("dense 989.5 · 3.35 TB/s · 80 GB", 9, "normal", C_BLUE_D, True, 0.05),
    ("dense roof -> ridge ~295; chapter", 9, "normal", C_TEXT, True, 0.05),
    ("uses the sparse ~2000 roof (600)", 9, "normal", C_TEXT, True, 0.085),
    ("vLLM engine arguments:", 10, "bold", C_TEXT, False, 0.055),
    ("docs.vllm.ai/en/latest/configuration", 9, "normal", "#047857", True, 0.05),
    ("/engine_args · config/scheduler.py", 9, "normal", "#047857", True, 0.085),
    ("SGLang server arguments:", 10, "bold", C_TEXT, False, 0.055),
    ("docs.sglang.io/docs/advanced_features", 9, "normal", "#047857", True, 0.05),
    ("/server_arguments", 9, "normal", "#047857", True, 0.085),
    ("defaults: vLLM max_num_seqs 128 ·", 9, "normal", C_TEXT, True, 0.05),
    ("gpu_mem_util 0.92 · SGLang static 0.88", 9, "normal", C_TEXT, True, 0.0),
]
y = 0.97
for text, size, weight, color, mono, gap in rows:
    a_src.text(0.02, y, text, fontsize=size, fontweight=weight, color=color, va="top",
               family="monospace" if mono else None)
    y -= gap
a_src.set_title("I · Sources — checked Sep 2026", fontsize=10.5)

fig.savefig("batching-effects.png", dpi=200, facecolor="white")
print("saved batching-effects.png")
