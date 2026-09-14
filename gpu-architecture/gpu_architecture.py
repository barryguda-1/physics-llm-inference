# GPU architecture: everything in inference performance bottoms out here. A GPU
# is ~132 copies of one small processor (the SM) sitting on a memory pyramid
# whose levels differ ~6x in bandwidth and ~300x in latency. The execution
# model (grid -> blocks -> warps of 32) plus the pyramid explain why decode is
# memory-bound (previous sheet: GEMV intensity ~1 vs ridge 295) and what a
# kernel can do about it: raise FLOPs per byte -- batching (x B), shared-memory
# tiles (x T/2 per tile of T), L2 grouping (x G).
#
# The chapter's numbers (H100 SXM), verified online Sep 2026:
#   SMs            132 (16,896 FP32 cores = 128/SM, 528 tensor cores = 4/SM)
#                  nvidia.com H100 datasheet
#   registers      64K x 32-bit = 256 KB per SM          CUDA C Programming
#   shared memory  up to 228 KB per SM (227 usable/block) Guide, compute
#   L2             50 MB, shared by all SMs              capability 9.0 table
#   HBM3           80 GB at 3.35 TB/s
#   threads        warp = 32; 2048 threads/SM = 64 warps; 32 blocks/SM;
#                  1024 threads/block
#   cross-check    132 SMs x 128 cores x 2 FLOPs x 1.98 GHz = 67 TFLOPS FP32
#                  = the datasheet figure, so the SM count and clock close
#   B200 (Blackwell) 192 GB HBM3e at 8 TB/s -- the chapter's figure checks out;
#                  dense BF16 ~2.25 PFLOPS ~= 2.3x H100 (nvidia.com/data-center/
#                  b200 HGX numbers / 8 GPUs / 2 sparse); SM count is not
#                  itemized by NVIDIA (~144 per public teardowns), L2 ~126 MB
#
# Two models in this sheet:
#   Little's law (panel E): bytes in flight >= BW x latency. The per-SM fair
#     share of HBM is 3.35 TB/s / 132 = 25.4 GB/s; at the text's ~300-cycle
#     (~152 ns) latency that is 3.8 KB per SM = ~30 warps each holding one
#     128 B load (32 threads x fp32) in flight. A realistic ~600 ns HBM round
#     trip needs 15.2 KB = ~120 warps > the 64-warp maximum, which is why real
#     kernels unroll loops and keep 4+ loads in flight per warp instead of
#     chasing 100% occupancy -- the chapter's "more occupancy isn't always
#     better".
#   Tiled GEMM intensity (panel D): a T x T output tile fed through shared
#     memory does 2 T^2 K FLOPs on 4 T K HBM bytes -> AI = T/2 FLOP/byte (fp16,
#     MAC = 2 FLOPs, same counting as the GEMV/batching sheets). 228 KB caps a
#     double-buffered fp16 A+B tile pair at T ~ 171, so shared-memory tiling
#     alone tops out near AI 85 -- below the dense ridge 295; reaching the
#     measured 709 TFLOPS needs L2 reuse across a group of blocks too.
# Change the config and re-run to see the picture move.
# Label positions are tuned to pass check_layout.py (panel ~488x375 px @100dpi).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ----------------------------- config: H100 SXM -----------------------------
SMS = 132
FP32_PER_SM = 128
BOOST_GHZ = 1.98                 # SXM5 boost clock
L2_MB, HBM_GB, HBM_TBPS = 50, 80, 3.35
REGS_PER_SM = 65_536             # 32-bit registers
REG_KB = REGS_PER_SM * 4 / 1024  # = 256 KB
SMEM_KB = 228                    # per SM, max carve-out (227 usable per block)
THREADS_PER_SM = 2048
THREADS_PER_BLOCK = 1024
BLOCKS_PER_SM = 32
WARP = 32
WARPS_PER_SM = THREADS_PER_SM // WARP

# the chapter's memory ladder (register/smem/L2 bandwidths are its estimates)
BW_REGS, BW_SMEM, BW_L2 = 20.0, 15.0, 8.0     # TB/s
LAT_REGS, LAT_SMEM, LAT_L2, LAT_HBM = 1, 25, 200, 300   # cycles
LAT_HBM_NS = LAT_HBM / BOOST_GHZ                          # cycles / GHz = ns (~152)
LAT_HBM_NS_REAL = 600.0                        # measured HBM round trips, 400-800 ns
MLP = 4                                        # loads kept in flight per warp (unroll)

# B200, as far as NVIDIA itemizes it
B200_HBM_TBPS, B200_HBM_GB = 8.0, 192
B200_BF16_DENSE_PF = 2.25                      # HGX B200 page: 36 PF sparse / 8 / 2

# ----------------------------- derived model -----------------------------
KB = 1024
resident_threads = SMS * THREADS_PER_SM                     # 270,336
fp32_tflops = SMS * FP32_PER_SM * 2 * BOOST_GHZ / 1000      # 67 TFLOPS
fp16_dense_tflops = 989.5                                   # as in the GEMM sheet
ridge = fp16_dense_tflops / HBM_TBPS                        # 295 FLOP/byte

# Little's law: per-SM bytes that must be in flight to saturate HBM
bw_per_sm = HBM_TBPS * 1e3 / SMS                           # 25.4 GB/s
inflight_text = bw_per_sm * LAT_HBM_NS                      # GB/s × ns = bytes
inflight_real = bw_per_sm * LAT_HBM_NS_REAL                 # bytes
warp_load = WARP * 4                                        # 128 B: one fp32 per thread
warps_needed_text = inflight_text / warp_load               # ~30
warps_needed_real = inflight_real / warp_load               # ~119 > 64 max
cover_at_max = WARPS_PER_SM * warp_load / inflight_real     # 54%

# registers vs occupancy: 65536 / (regs * threads) warps fit
def regs_at_warps(w):
    return REGS_PER_SM / (WARP * w)

# tiled GEMM: AI = T/2; smem caps a double-buffered fp16 A+B pair at 8 T^2 bytes
def ai_tile(t):
    return t / 2

smem_tile_max = np.sqrt(SMEM_KB * KB / 8)                   # ~171

# ----------------------------- sanity checks against the text -----------------------------
assert REG_KB == 256.0 and SMS * FP32_PER_SM == 16896
assert WARPS_PER_SM == 64 and THREADS_PER_BLOCK // WARP == 32
assert round(fp32_tflops) == 67                             # closes the datasheet loop
assert resident_threads == 270336
assert round(ridge) == 295
assert round(bw_per_sm, 1) == 25.4
assert round(warps_needed_text) == 30 and warps_needed_real > 64
assert round(cover_at_max * 100) == 54
assert regs_at_warps(64) == 32 and regs_at_warps(32) == 64
assert round(smem_tile_max, 1) == 170.8
assert round(ai_tile(1), 1) == 0.5 and round(ai_tile(128)) == 64
assert round(HBM_TBPS / BW_REGS, 2) == 0.17                 # ~6x bandwidth gap
assert LAT_HBM / LAT_REGS == 300                            # ~300x latency gap
print(f"H100 SXM: {SMS} SMs x {FP32_PER_SM} FP32 x 2 x {BOOST_GHZ} GHz = "
      f"{fp32_tflops:.1f} TFLOPS FP32 (= datasheet)")
print(f"resident threads: {THREADS_PER_SM}/SM x {SMS} = {resident_threads:,} "
      f"= {WARPS_PER_SM} warps/SM")
print(f"ladder: regs {BW_REGS} / smem {BW_SMEM} / L2 {BW_L2} / HBM {HBM_TBPS} TB/s "
      f"(~{BW_REGS/HBM_TBPS:.1f}x) · latency {LAT_REGS}/{LAT_SMEM}/{LAT_L2}/{LAT_HBM}+ cyc "
      f"({LAT_HBM/LAT_REGS}x)")
print(f"Little's law @ {bw_per_sm:.1f} GB/s/SM: {LAT_HBM_NS:.0f} ns -> {inflight_text/KB:.1f} KB "
      f"= {warps_needed_text:.0f} warps of one {warp_load} B load; "
      f"{LAT_HBM_NS_REAL:.0f} ns -> {inflight_real/KB:.1f} KB = {warps_needed_real:.0f} warps "
      f"> {WARPS_PER_SM} max -> cover only {cover_at_max:.0%} unless unrolled x{MLP} "
      f"({inflight_real/(warp_load*MLP):.0f} warps)")
print(f"occupancy vs registers: 64 warps => {regs_at_warps(64):.0f} regs/thread · "
      f"32 => {regs_at_warps(32):.0f} · 16 => {regs_at_warps(16):.0f}")
print(f"tiling: AI = T/2 · T=128 -> {ai_tile(128):.0f} FLOP/byte = {ai_tile(128)*HBM_TBPS:.0f} TF "
      f"HBM-fed (< {fp16_dense_tflops} peak) · smem caps double-buffered fp16 at "
      f"T~{smem_tile_max:.0f} (AI {ai_tile(smem_tile_max):.0f}) < ridge {ridge:.0f}")
print(f"B200: {B200_HBM_TBPS} TB/s HBM3e (text checks out) · ~{B200_BF16_DENSE_PF} PF dense BF16 "
      f"= {B200_BF16_DENSE_PF*1000/fp16_dense_tflops:.1f}x H100's fp16 peak")

# ----------------------------- style -----------------------------
C_BLUE, C_DBLUE, C_GREEN, C_AMBER, C_RED = "#3b82f6", "#1e40af", "#10b981", "#f59e0b", "#ef4444"
C_TEXT, C_GRID, C_MID, C_CELL = "#0f172a", "#cbd5e1", "#64748b", "#e2e8f0"
C_DRED = "#b91c1c"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
(a_anat, a_exec, a_mem), (a_reuse, a_occ, a_txt) = axes
fig.suptitle("GPU Architecture — the machine under every kernel\n"
             "132 SMs · warps of 32 · a memory pyramid ~6× faster at each level up — "
             "why \"memory-bound\" is the default, and what kernels do about it",
             fontsize=14.5, fontweight="bold", linespacing=1.5)

# ---------------- panel A: the chip — SMs over a memory pyramid ----------------
ax = a_anat
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])

# the die: 132 SM cells
gx0, gy0, cell = 0.35, 3.0, 0.278
gcols, grows = 11, 12
for r in range(grows):
    for c in range(gcols):
        ax.add_patch(Rectangle((gx0 + c * cell, gy0 + r * cell), cell, cell,
                               fc="#dbeafe", ec="#93c5fd", lw=0.4))
ax.add_patch(Rectangle((gx0 - 0.06, gy0 - 0.06), gcols * cell + 0.12, grows * cell + 0.12,
                       fill=False, ec=C_DBLUE, lw=1.5))
ax.text(gx0 + gcols * cell / 2, 6.5, f"{SMS} SMs",
        ha="center", fontsize=10.5, fontweight="bold", color=C_DBLUE)
ax.text(gx0 + gcols * cell / 2, 2.72, "16,896 FP32 · 528 tensor cores",
        ha="center", fontsize=8, color=C_MID)

# L2 + HBM below
bw_bar_w = gcols * cell
ax.add_patch(Rectangle((gx0, 1.98), bw_bar_w, 0.5, fc="#dcfce7", ec=C_GREEN, lw=1.2))
ax.text(gx0 + bw_bar_w / 2, 2.23, "L2 · 50 MB — all SMs", ha="center",
        va="center", fontsize=8, color="#047857", fontweight="bold")
ax.add_patch(Rectangle((gx0, 1.18), bw_bar_w, 0.5, fc="#fee2e2", ec=C_RED, lw=1.2))
ax.text(gx0 + bw_bar_w / 2, 1.43, "HBM3 · 80 GB · 3.35 TB/s", ha="center", va="center",
        fontsize=8, color=C_DRED, fontweight="bold")
for y0, y1 in ((2.52, 2.94), (1.72, 1.94)):
    ax.annotate("", xy=(2.0, y0), xytext=(2.0, y1),
                arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.1))
ax.text(gx0 + bw_bar_w / 2, 0.52, "the chip: 132 copies of one\nsmall processor, stacked on\n"
        "a single memory pyramid", ha="center", va="center",
        fontsize=8.5, color=C_MID, style="italic", linespacing=1.5)

# one SM, up close
zx0, zy0, zx1, zy1 = 4.6, 1.3, 9.9, 6.55
ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fc="#f8fafc", ec=C_GRID, lw=1.2))
ax.plot([gx0 + 7 * cell + cell / 2, zx0], [gy0 + 8 * cell + cell / 2, 6.2],
        ls="--", color="#94a3b8", lw=1)
ax.text((zx0 + zx1) / 2, 6.32, "one SM, up close", ha="center", fontsize=10,
        fontweight="bold")
ax.text((zx0 + zx1) / 2, 6.05, "4 sub-partitions (SMSPs)", ha="center", fontsize=8,
        color=C_MID)

sp_w, sp_h, sp_y = 1.13, 2.55, 3.0
warp_colors = ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
for i in range(4):
    x = 4.8 + i * 1.27
    ax.add_patch(Rectangle((x, sp_y), sp_w, sp_h, fc="white", ec=C_DBLUE, lw=1))
    ax.text(x + sp_w / 2, sp_y + sp_h - 0.18, f"SMSP {i}", ha="center", fontsize=8,
            fontweight="bold", color=C_DBLUE)
    # 32 lanes as an 8x4 grid of dots
    for rr in range(8):
        for cc in range(4):
            ax.plot(x + sp_w / 2 - 0.26 + cc * 0.17, sp_y + 0.88 + rr * 0.16, "o",
                    ms=2.4, color=warp_colors[i])
ax.text((zx0 + zx1) / 2, 2.74, "each SMSP: 32 FP32 lanes + 1 tensor core",
        ha="center", va="center", fontsize=8, color=C_TEXT)

ax.add_patch(Rectangle((4.8, 2.12), zx1 - 0.15 - 4.8, 0.42, fc="#dbeafe", ec=C_DBLUE, lw=1))
ax.text((zx0 + zx1) / 2, 2.33, "registers · 256 KB (64K × 32-bit)",
        ha="center", va="center", fontsize=8, color=C_DBLUE, fontweight="bold")
ax.add_patch(Rectangle((4.8, 1.56), zx1 - 0.15 - 4.8, 0.42, fc="#d1fae5", ec=C_GREEN, lw=1))
ax.text((zx0 + zx1) / 2, 1.77, "shared memory / L1 · up to 228 KB",
        ha="center", va="center", fontsize=8, color="#047857", fontweight="bold")
ax.set_xlim(0, 10)
ax.set_ylim(0.15, 6.85)
ax.set_title("A · The chip: SMs above a memory pyramid", fontsize=10.5)

# ---------------- panel B: grid -> blocks -> warps + the work distributor ----------------
ax = a_exec
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])

# the grid
ax.text(1.8, 7.32, "kernel<<<(4,3), 1024>>>", ha="center", fontsize=9,
        family="monospace", fontweight="bold", color=C_TEXT)
bc, bcell = 0.35, 0.52
for r in range(3):
    for c in range(4):
        bid = r * 4 + c
        fc = "#dbeafe" if bid != 8 else C_AMBER
        ax.add_patch(Rectangle((bc + c * bcell, 5.6 + r * bcell), bcell, bcell,
                               fc=fc, ec=C_DBLUE, lw=0.8))
        ax.text(bc + c * bcell + bcell / 2, 5.6 + r * bcell + bcell / 2, str(bid),
                ha="center", va="center", fontsize=8.5,
                fontweight="bold" if bid == 8 else "normal",
                color="white" if bid == 8 else C_DBLUE)
ax.text(1.58, 5.30, "gridDim (4,3) · 12 blocks", ha="center", fontsize=8.5,
        fontweight="bold", color=C_DBLUE)
ax.annotate("", xy=(3.38, 5.95), xytext=(2.62, 5.85),
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2))

# one block = warps of 32
tx0, ty0, tcell = 3.5, 5.45, 0.19
for r in range(4):
    ax.text(tx0 - 0.12, ty0 + r * tcell + tcell / 2, f"w{r}", ha="right", va="center",
            fontsize=7.5, color=C_MID)
    for c in range(32):
        ax.add_patch(Rectangle((tx0 + c * tcell, ty0 + r * tcell), tcell, tcell,
                               fc=warp_colors[r], ec="white", lw=0.3))
ax.text(6.54, 6.45, "blockDim = 1024 threads = 32 warps (4 shown)",
        ha="center", fontsize=8.5, fontweight="bold")
ax.text(6.54, 5.20, "one instruction per warp — 32 lanes together (SIMT)\n"
        "a divergent branch runs both paths — serialized",
        ha="center", va="top", fontsize=8, color=C_MID)

# the work distributor: a mini Gantt of blocks on 4 SMs
ax.text(5.0, 4.50, "the work distributor: every finished block frees its SM for the next one",
        ha="center", fontsize=8.5, fontweight="bold")
sched = {0: [(0, 2.2, "0"), (2.2, 1.6, "4"), (3.8, 1.9, "8")],
         1: [(0, 1.7, "1"), (1.7, 1.9, "5"), (3.6, 2.1, "9")],
         2: [(0, 2.9, "2"), (2.9, 1.4, "6"), (4.3, 1.7, "10")],
         3: [(0, 2.2, "3"), (2.2, 2.4, "7"), (4.6, 1.4, "11")]}
lane_y = {0: 3.82, 1: 3.12, 2: 2.42, 3: 1.72}
lane_h = 0.5
for sm, segs in sched.items():
    ax.text(0.95, lane_y[sm] + lane_h / 2, f"SM {sm}", ha="right", va="center",
            fontsize=8.5, fontweight="bold")
    for start, dur, bid in segs:
        x0 = 1.1 + start * 1.15
        w = dur * 1.15
        fc = C_AMBER if bid == "8" else "#bfdbfe"
        ax.add_patch(Rectangle((x0, lane_y[sm]), w, lane_h, fc=fc, ec="white", lw=1))
        ax.text(x0 + w / 2, lane_y[sm] + lane_h / 2, bid, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white" if bid == "8" else C_DBLUE)
ax.annotate("SM 0 frees → grabs block 8", xy=(1.1 + 3.8 * 1.15, lane_y[0] + lane_h + 0.04),
            xytext=(6.65, 3.42), fontsize=8, fontweight="bold", color="#b45309",
            arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.3))
ax.annotate("", xy=(8.15, 1.32), xytext=(1.1, 1.32),
            arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.2))
ax.text(8.25, 1.32, "time", ha="left", va="center", fontsize=8, color=C_MID)
ax.text(5.0, 0.72, "no block→SM affinity — the scheduler keeps all 132 SMs fed",
        ha="center", fontsize=8.5, color=C_MID, style="italic")
ax.set_xlim(0, 10)
ax.set_ylim(0.4, 7.5)
ax.set_title("B · Grid → blocks → warps of 32", fontsize=10.5)

# ---------------- panel C: the memory pyramid ----------------
ax = a_mem
ax.axis("off")
ax.grid(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.1)

rows = [
    ("REGISTERS\nper thread", BW_REGS, C_DBLUE, "20 TB/s",
     "255/thread · 256 KB/SM", "1 cycle"),
    ("SHARED MEMORY\nper block, in the SM", BW_SMEM, C_BLUE, "15 TB/s",
     "228 KB per SM", "~25 cycles"),
    ("L2 CACHE\nall SMs, automatic", BW_L2, C_GREEN, "8 TB/s",
     "50 MB (all SMs)", "~200 cycles"),
    ("HBM / GLOBAL\nall SMs", HBM_TBPS, C_RED, None,
     "80 GB · tensors live here", "~300 cyc · 400–800 ns"),
]
ax.text(1.35, 6.82, "level · scope", ha="center", fontsize=8, fontweight="bold", color=C_MID)
ax.text(4.9, 6.82, "bandwidth (est.)", ha="center", fontsize=8,
        fontweight="bold", color=C_MID)
ax.text(8.55, 6.82, "size · latency", ha="center", fontsize=8, fontweight="bold", color=C_MID)

lo, hi = np.log10(3), np.log10(25)
bx0, bmax = 2.75, 4.3
ys = [5.85, 4.35, 2.85, 1.35]
for (name, bw, col, inside, size, lat), y in zip(rows, ys):
    ax.text(0.05, y, name, ha="left", va="center", fontsize=9, fontweight="bold",
            color=C_TEXT, linespacing=1.5)
    w = (np.log10(bw) - lo) / (hi - lo) * bmax
    ax.add_patch(Rectangle((bx0, y - 0.26), w, 0.52, fc=col, ec="none", alpha=0.85))
    if inside:      # bar is wide enough to hold its label
        ax.text(bx0 + w / 2, y, inside, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")
    else:           # HBM's short bar gets an outside label
        ax.text(bx0 + w + 0.14, y, f"{HBM_TBPS} TB/s", ha="left", va="center", fontsize=9,
                fontweight="bold", color=col)
    ax.text(7.0, y + 0.15, size, ha="left", va="center", fontsize=8.5, color=C_TEXT)
    ax.text(7.0, y - 0.15, lat, ha="left", va="center", fontsize=8.5, color=C_MID)
ax.text(5.0, 0.30, "6× the bandwidth and ~300× the latency from bottom to top —\n"
        "hoist data up the pyramid and reuse it there", ha="center", va="bottom",
        fontsize=8, color=C_MID, style="italic", linespacing=1.5)
ax.set_title("C · The memory pyramid: bandwidth vs capacity", fontsize=10.5)

# ---------------- panel D: reuse beats streaming (tiling) ----------------
T = np.geomspace(1, 1024, 300)
a_reuse.loglog(T, ai_tile(T), color=C_DBLUE, lw=2.5, zorder=3)
a_reuse.axhspan(ridge, 2000, color=C_GREEN, alpha=0.06, zorder=0)
a_reuse.axhline(ridge, color=C_BLUE, ls="--", lw=1.6)
a_reuse.axhline(597, color=C_RED, ls=":", lw=1.4)
a_reuse.text(1.15, 330, "dense ridge 295 = 989.5 TF / 3.35 TB/s", fontsize=8.5,
             color=C_BLUE, fontweight="bold", va="bottom")
a_reuse.text(1.15, 660, "the text's 597 (sparse peak)", fontsize=8, color=C_RED, va="top")
a_reuse.axvline(smem_tile_max, color=C_RED, ls=":", lw=1.2)
a_reuse.axvspan(smem_tile_max, 1024, color=C_RED, alpha=0.04, zorder=0)
a_reuse.text(140, 0.6, "228 KB caps a 2-buffered\nfp16 tile at T≈171",
             fontsize=8, color=C_RED, va="bottom")
a_reuse.text(9, 4, "AI = T/2 (fp16)", fontsize=9.5, color=C_DBLUE, fontweight="bold",
             rotation=32, rotation_mode="anchor", ha="center")

a_reuse.plot([1], [1], "o", color=C_RED, ms=7, zorder=5)
a_reuse.annotate("T=1 · GEMV decode: AI 1\n2.6 TFLOPS measured", xy=(1, 1),
                 xytext=(1.6, 0.52), fontsize=8.5, fontweight="bold", color=C_DRED)
a_reuse.plot([32], [32], "o", color=C_AMBER, ms=7, zorder=5)
a_reuse.annotate("batch B=32: AI 32\n107 TFLOPS measured", xy=(32, 32),
                 xytext=(3.4, 75), fontsize=8.5, fontweight="bold", color="#b45309")
a_reuse.plot([128], [64], "o", color=C_BLUE, ms=7, zorder=5)
a_reuse.annotate("smem tile T=128: AI 64\n→ 214 TF from HBM", xy=(128, 64),
                 xytext=(105, 12), fontsize=8.5, fontweight="bold", color=C_DBLUE)
a_reuse.plot([128], [512], "*", color=C_GREEN, ms=15, zorder=5)
a_reuse.annotate("…+ L2 grouping (8 blocks): AI 512\n→ compute-bound · 709 TF measured",
                 xy=(128, 512), xytext=(118, 950), ha="right", fontsize=8.5,
                 fontweight="bold", color="#047857")
a_reuse.plot([128, 128], [64, 512], ls=":", color=C_GREEN, lw=1.2, zorder=4)

a_reuse.text(0.02, 0.03, "reuse ladder: batch ×B · smem tile ×T/2 · L2 group ×G",
             transform=a_reuse.transAxes, fontsize=8.5, color=C_MID, style="italic")
a_reuse.set_xlim(1, 1024)
a_reuse.set_ylim(0.3, 2000)
a_reuse.set_xticks([1, 8, 32, 128, 512, 1024], ["1", "8", "32", "128", "512", "1024"])
a_reuse.set_yticks([0.5, 10, 100, 1000], ["0.5", "10", "100", "1000"])
a_reuse.minorticks_off()
a_reuse.set_xlabel("tile size T (fp16) — T=1 is the GEMV case; batch B ≈ tile T=2B")
a_reuse.set_ylabel("arithmetic intensity (FLOP / byte)")
sec = a_reuse.secondary_yaxis("right", functions=(lambda v: v * HBM_TBPS,
                                                  lambda v: v / HBM_TBPS))
sec.set_yticks([1, 10, 100, 1000], ["3.4", "34", "335", "1000*"])
sec.set_ylabel("HBM-fed ceiling (TFLOPS)")
a_reuse.set_title("D · Reuse beats streaming: intensity = T/2 per smem tile of T\n"
                  "tiling alone can't reach the ridge — L2 reuse finishes the job",
                  fontsize=10.5)

# ---------------- panel E: occupancy — Little's law ----------------
w = np.linspace(0.01, WARPS_PER_SM, 400)
def cover(warps, latency_ns, bytes_per_warp):
    return np.minimum(100, warps * bytes_per_warp / (bw_per_sm * latency_ns) * 100)

a_occ.plot(w, cover(w, LAT_HBM_NS, warp_load), color=C_BLUE, lw=2.2,
           label=f"{LAT_HBM_NS:.0f} ns (text ~300 cyc) · 1 load/warp")
a_occ.plot(w, cover(w, LAT_HBM_NS_REAL, warp_load), color=C_RED, lw=2.2,
           label=f"{LAT_HBM_NS_REAL:.0f} ns (real HBM) · 1 load/warp")
a_occ.plot(w, cover(w, LAT_HBM_NS_REAL, warp_load * MLP), color=C_RED, lw=1.6, ls="--",
           label=f"{LAT_HBM_NS_REAL:.0f} ns · {MLP} loads/warp (unrolled)")
a_occ.axvline(32, color="#94a3b8", ls=":", lw=1.1)
a_occ.axvline(WARPS_PER_SM, color="#334155", ls="-", lw=1.2)
a_occ.text(31, 4, "50%", fontsize=8, color="#94a3b8", ha="right", va="bottom")
a_occ.text(63, 4, "64 = max", fontsize=8, color="#334155",
           ha="right", va="bottom")
a_occ.text(44, 60, "one load/warp tops out at 54%", fontsize=8, color=C_DRED,
           ha="center", va="center")

a_occ.text(0.02, 0.98, "in-flight ≥ BW × latency (Little)\n"
           f"152 ns: {inflight_text/KB:.1f} KB → {warps_needed_text:.0f} warps\n"
           f"600 ns: {inflight_real/KB:.1f} KB → {warps_needed_real:.0f} > 64 max",
           transform=a_occ.transAxes, ha="left", va="top", fontsize=8.5,
           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GRID, alpha=0.95))
a_occ.text(0.98, 0.98, "warps hide latency: one computes\nwhile another waits — but registers\n"
           "trade off: 64 warps ⇒ ≤32 regs\n32 ⇒ 64 · FlashAttention runs\n"
           "25–50% occupancy on purpose",
           transform=a_occ.transAxes, ha="right", va="top", fontsize=8, color="#334155",
           bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec=C_GRID, alpha=0.95))
a_occ.set_xlim(0, WARPS_PER_SM)
a_occ.set_ylim(0, 130)
a_occ.set_yticks([0, 25, 50, 75, 100])
a_occ.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 64])
a_occ.set_xlabel("resident warps per SM (occupancy)")
a_occ.set_ylabel("% of per-SM HBM share coverable")
a_occ.legend(fontsize=8, loc="center right", framealpha=0.95)
secx = a_occ.secondary_xaxis("top", functions=(lambda v: REGS_PER_SM / (WARP * np.maximum(v, 1e-9)),
                                               lambda v: REGS_PER_SM / (WARP * np.maximum(v, 1e-9))))
secx.set_xticks([256, 128, 64, 32], ["256", "128", "64", "32"])
secx.set_xlabel("registers per thread allowed at this occupancy (65536 / 32·warps)",
                fontsize=9)
a_occ.set_title("E · Occupancy = latency hiding (Little's law)", fontsize=10.5)

# ---------------- panel F: the takeaway + fact check ----------------
a_txt.axis("off")
a_txt.grid(False)
a_txt.set_xticks([]); a_txt.set_yticks([])
rows = [
    ("132 SMs × (128 FP32 + 4 tensor) + 50 MB L2 + 80 GB HBM", 10, "bold", C_TEXT, False, 0.068),
    ("grid → blocks (any SM) → warps of 32", 9, "normal", C_DBLUE, True, 0.055),
    ("pyramid: 20 · 15 · 8 · 3.35 TB/s — 6× per level", 9, "normal", C_MID, True, 0.055),
    ("latency: 1 · ~25 · ~200 · ~300+ cyc — hide with warps", 9, "normal", C_MID, True, 0.055),
    ("2048 threads/SM = 64 warps · 270,336 on the chip", 9, "normal", C_DBLUE, True, 0.075),
    ("regs ↔ warps: 64 warps ⇒ 32 regs/thread · 32 ⇒ 64", 9, "normal", C_MID, True, 0.055),
    ("reuse ladder: batch ×B · tile ×T/2 · L2 ×G → ridge 295", 9, "bold", "#b45309", True, 0.088),
    ("decode GEMV (AI≈1) can't climb — batching is the lever", 9, "bold", C_DRED, True, 0.050),
    ("Compute is cheap, bytes are dear. Spend warps to hide", 10, "bold", C_TEXT, False, 0.062),
    ("latency; spend the pyramid to multiply reuse.", 10, "bold", C_TEXT, False, 0.0),
]
y = 0.97
for text, size, weight, color, mono, gap in rows:
    a_txt.text(0.02, y, text, fontsize=size, fontweight=weight, color=color, va="top",
               family="monospace" if mono else None)
    y -= gap
a_txt.text(0.02, 0.36,
           "fact check (Sep 2026; nvidia.com + CUDA guide cc 9.0): all H100\n"
           "SXM figures check out — 132 SMs · 228 KB smem · 256 KB regs ·\n"
           "50 MB L2 · 3.35 TB/s HBM3; 132×128×2×1.98 GHz = 67 TFLOPS FP32\n"
           "= datasheet, so SM count and clock close. B200: 8 TB/s HBM3e ✓ ·\n"
           "192 GB · dense BF16 ~2.25 PF ≈ 2.3× H100 (HGX page ÷8÷2); SM\n"
           "count not itemized (~144 reported). 'Lockstep' is the pre-Volta\n"
           "model — independent thread scheduling still serializes divergent\n"
           "warps. '~300 cycles' ≈ L2-class; real HBM round trips 400–800 ns.\n"
           "Reg/smem/L2 bandwidths are order-of-magnitude estimates.",
           fontsize=8.5, color="#92400e", va="top",
           bbox=dict(boxstyle="round,pad=0.5", fc="#fef3c7", ec=C_AMBER, lw=1))
a_txt.set_title("F · The takeaway", fontsize=10.5)

# free-floating labels are hand-placed in data coordinates; keep them out of
# constrained layout (texts default to in_layout=True, and long lines compound
# until the panels collapse) — panel sizes then come from the grid alone
for ax_ in fig.axes:
    for t in ax_.texts:
        t.set_in_layout(False)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu-architecture.png")
fig.savefig(out, dpi=200, facecolor="white")
print(f"saved {out}")
