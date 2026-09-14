# Hello CUDA: vector addition is the "hello world" of GPU programming, and it
# exercises the whole mental model on one page -- a __global__ function
# launched as a grid of blocks of threads, each thread computing its own
# index (blockIdx.x * blockDim.x + threadIdx.x) and guarding the tail
# (if idx < n), the grid-stride variant, explicit host<->device memory, and
# Python bindings via torch's load_inline.
#
# The chapter's numbers, checked:
#   n = 1,000,000 floats, block_size 256 -> num_blocks = ceil(1e6/256) = 3907
#   (plain division floors 3906.25 to 3906; +255 first rounds it up). Threads
#   launched = 3907*256 = 1,000,192, so the last block holds idx 999,936..
#   1,000,191: 64 threads pass the guard, 192 idle. Worked index example:
#   17*256 + 42 = 4,394.
#   Memory: 3 arrays x 4 MB = 12 MB of traffic per run. On HBM3 at 3.35 TB/s
#   that is ~3.6 us of ideal kernel time. The PCIe copies dominate: 8 MB up +
#   4 MB down at ~25 GB/s practical (PCIe 4 x16; PCIe 5 doubles it) = ~480 us
#   ~= 134x the kernel -- the reason inference keeps tensors resident on the
#   GPU between ops.
#   Arithmetic intensity: one add (1 FLOP) per 12 bytes (2 loads + 1 store of
#   fp32) -> AI = 1/12 ~= 0.083 FLOP/byte, vs the H100 ridge ~295 (from the
#   GEMV / gpu-architecture sheets): hopelessly memory-bound. The chapter's
#   "1/12 FLOPs per byte" is exactly this; its "3 FLOPs of compute" is better
#   read as 3 memory ops (2 loads + 1 store) -- one add is 1 FLOP.
#   256 threads/block: a multiple of the 32-thread warp; the max is 1024.
# Change the config and re-run to see the picture move. Label positions are
# tuned to pass check_layout.py (panel ~488x375 px @100dpi).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ----------------------------- config -----------------------------
N, BLOCK = 1_000_000, 256
HBM_TBPS, PCIE_GBPS = 3.35, 25.0        # H100 SXM HBM3; PCIe 4 x16 practical
FP32_TFLOPS = 67.0                      # 132 SMs x 128 FP32 x 2 x 1.98 GHz
RIDGE = 295.0                           # FLOP/byte, from the earlier sheets

# ----------------------------- derived + sanity checks -----------------------------
num_blocks = (N + BLOCK - 1) // BLOCK          # the ceil-div launch idiom
threads = num_blocks * BLOCK
tail_first = (num_blocks - 1) * BLOCK          # first idx of the last block
tail_active = N - tail_first                   # threads that pass idx < n
tail_idle = threads - N
assert num_blocks == 3907 and threads == 1_000_192
assert tail_first == 999_936 and tail_active == 64 and tail_idle == 192
assert 17 * 256 + 42 == 4_394

AI = 1 / 12                                    # FLOP per byte: 1 add / (2+1) floats
hbm_ceiling = HBM_TBPS * AI                    # TFLOPS HBM can feed: 0.279
kernel_us = 12e6 / (HBM_TBPS * 1e12) * 1e6     # 12 MB at full HBM bandwidth
h2d_us = 8e6 / (PCIE_GBPS * 1e9) * 1e6
d2h_us = 4e6 / (PCIE_GBPS * 1e9) * 1e6
assert round(hbm_ceiling, 3) == 0.279
assert round(kernel_us, 1) == 3.6 and h2d_us == 320 and d2h_us == 160
assert round((h2d_us + d2h_us) / kernel_us) == 134
assert round(FP32_TFLOPS / hbm_ceiling) == 240  # compute headroom wasted
assert round(RIDGE / AI) == 3_540               # how far below the ridge we sit

print(f"launch: n={N:,}, block={BLOCK} -> {num_blocks} blocks x {BLOCK} threads "
      f"= {threads:,} threads ({tail_idle} guarded off in the tail block)")
print(f"index examples: 0*256+0={0*BLOCK+0} · 0*256+255={0*BLOCK+255} · "
      f"1*256+0={1*BLOCK+0} · 17*256+42={17*BLOCK+42}")
print(f"memory: 12 MB/run -> kernel {kernel_us:.1f} us on HBM vs "
      f"copies {h2d_us:.0f}+{d2h_us:.0f} us on PCIe = "
      f"{(h2d_us+d2h_us)/kernel_us:.0f}x more time moving than computing")
print(f"roofline: AI = 1/12 = {AI:.3f} FLOP/byte vs ridge {RIDGE:.0f} -> "
      f"{RIDGE/AI:,.0f}x below; HBM-fed ceiling {hbm_ceiling:.2f} TFLOPS "
      f"vs {FP32_TFLOPS:.0f} TFLOPS FP32 peak")

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
(a_code, a_idx, a_bound), (a_stride, a_mem, a_py) = axes
fig.suptitle("Hello CUDA — vector addition, the “hello world” of GPU programming\n"
             "one __global__ function · a grid of blocks × threads · explicit host↔device memory · Python via load_inline",
             fontsize=14.5, fontweight="bold", linespacing=1.5)

# ---------------- panel A: host code launches, device code runs ----------------
ax = a_code
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.6)

ax.add_patch(Rectangle((0.1, 4.05), 4.35, 3.05, fc="#f8fafc", ec=C_GRID, lw=1.2))
ax.text(0.35, 6.85, "HOST — the CPU", fontsize=10, fontweight="bold", color=C_DBLUE)
ax.text(0.35, 6.55, "your program, one thread of control", fontsize=7.5, color=C_MID)
ax.text(0.35, 6.25,
        "int n = 1'000'000;\n"
        "int block_size = 256;\n"
        "int num_blocks=(n+255)/256;  //3907\n"
        "vector_add<<<num_blocks,\n"
        "    block_size>>>(d_a,d_b,d_c,n);",
        va="top", ha="left", fontsize=7, family="monospace", linespacing=1.55)

ax.add_patch(Rectangle((5.55, 4.05), 4.35, 3.05, fc="#f8fafc", ec=C_GRID, lw=1.2))
ax.text(5.8, 6.85, "DEVICE — the GPU", fontsize=10, fontweight="bold", color=C_DBLUE)
ax.text(5.8, 6.55, "thousands of simple cores in lockstep", fontsize=7.5, color=C_MID)
ax.text(5.8, 6.25,
        "__global__ void vector_add(\n"
        "    float* a, float* b,\n"
        "    float* c, int n) {\n"
        "  int idx = blockIdx.x * blockDim.x\n"
        "          + threadIdx.x;\n"
        "  if (idx < n)\n"
        "    c[idx] = a[idx] + b[idx];\n"
        "}",
        va="top", ha="left", fontsize=7, family="monospace", linespacing=1.55)

ax.annotate("", xy=(5.45, 5.3), xytext=(4.55, 5.3),
            arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.6))

ax.text(5.0, 3.62,
        "one launch: 3907 blocks × 256 threads = 1,000,192 threads run this same code",
        ha="center", fontsize=8.5, fontweight="bold")
ax.text(5.0, 3.18,
        "__global__ marks a function that runs on the GPU but is called from the CPU",
        ha="center", fontsize=8.5, color=C_MID)
ax.text(5.0, 2.74,
        "every thread executes the same function on different data — SIMT, in warps of 32",
        ha="center", fontsize=8.5, color=C_MID)
ax.text(5.0, 2.1,
        "the win: thousands of simple cores doing the same op at once",
        ha="center", fontsize=9, fontweight="bold", color=C_BLUE)
ax.text(5.0, 1.5,
        "the first thing every thread must work out: “which element is mine?”  →  panel B",
        ha="center", fontsize=8.5, color=C_MID, style="italic")
ax.set_title("A · A kernel is a function that runs on the GPU — the CPU launches it",
             fontsize=10.5)

# ---------------- panel B: thread indexing ----------------
ax = a_idx
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.15, 7.65)

ax.text(5.0, 7.15, "idx = blockIdx.x * blockDim.x + threadIdx.x",
        ha="center", fontsize=11, fontweight="bold", family="monospace")
ax.text(5.0, 6.68, "which block am I in  ×  how many threads per block  +  which seat in the block",
        ha="center", fontsize=8.5, color=C_MID)

for k in range(4):
    bx = 0.5 + k * 2.3
    ax.add_patch(Rectangle((bx, 4.85), 2.15, 1.35, fc="white", ec=C_DBLUE, lw=1.1))
    ax.text(bx + 1.075, 6.05, f"block {k}", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=C_DBLUE)
    for t in range(8):
        r, c = divmod(t, 4)
        cx, cy = bx + 0.12 + c * 0.49, 5.5 - r * 0.48
        g = 8 * k + t
        hot = (k == 1 and t == 0)
        ax.add_patch(Rectangle((cx, cy), 0.44, 0.42,
                               fc=C_AMBER if hot else "#dbeafe",
                               ec="white" if hot else "#93c5fd", lw=0.6))
        ax.text(cx + 0.22, cy + 0.21, str(g), ha="center", va="center",
                fontsize=7, fontweight="bold" if hot else "normal",
                color="white" if hot else C_DBLUE)
    ax.text(bx + 1.075, 4.62, f"blockIdx.x = {k}", ha="center", va="center",
            fontsize=7, color=C_MID)
ax.text(5.0, 4.22,
        "drawn with 8 threads per block — the real launch: blockDim.x = 256 · gridDim.x = 3907",
        ha="center", fontsize=8.5, color=C_MID)

ax.add_patch(Rectangle((0.5, 0.55), 9.05, 2.6, fc="#f8fafc", ec=C_GRID, lw=1.1))
ax.text(0.75, 2.92, "worked examples — block_size = 256:",
        ha="left", fontsize=8.5, fontweight="bold")
ax.text(0.75, 2.6,
        "thread   0 of block  0 :   0 × 256 +   0  =       0\n"
        "thread 255 of block  0 :   0 × 256 + 255  =     255\n"
        "thread   0 of block  1 :   1 × 256 +   0  =     256    <- the amber cell\n"
        "thread  42 of block 17 :  17 × 256 +  42  =   4,394",
        va="top", ha="left", fontsize=8.5, family="monospace", linespacing=1.6)
ax.set_title("B · Thread indexing: every thread computes its own element", fontsize=10.5)

# ---------------- panel C: the bounds check ----------------
ax = a_bound
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.7)

ax.text(5.0, 7.18, "num_blocks = (n + 255) / 256 = ceil(1,000,000 / 256) = 3907",
        ha="center", fontsize=9, family="monospace", fontweight="bold")
ax.text(5.0, 6.78, "plain division floors 3906.25 to 3906 — the +255 first makes it round up",
        ha="center", fontsize=8.5, color=C_MID)
ax.text(0.2, 6.55, "3,906 full blocks — every thread writes", ha="left",
        fontsize=8.5, fontweight="bold", color=C_DBLUE)
ax.text(9.8, 6.55, "the 3,907th block runs out of elements", ha="right",
        fontsize=8.5, fontweight="bold", color="#b45309")

ax.add_patch(Rectangle((0.2, 5.35), 1.25, 1.05, fc="#dbeafe", ec=C_DBLUE, lw=1))
ax.text(0.825, 5.875, "block 0", ha="center", va="center", fontsize=7.5, color=C_DBLUE)
ax.text(1.78, 5.875, "···", ha="center", va="center", fontsize=11, color=C_MID)
ax.add_patch(Rectangle((2.15, 5.35), 1.55, 1.05, fc="#dbeafe", ec=C_DBLUE, lw=1))
ax.text(2.925, 5.875, "block 3905", ha="center", va="center", fontsize=7.5, color=C_DBLUE)
ax.add_patch(Rectangle((3.95, 5.35), 1.1, 1.05, fc="#d1fae5", ec=C_GREEN, lw=1.1))
ax.text(4.5, 5.875, "64 threads\nwrite", ha="center", va="center", fontsize=7,
        color="#047857", fontweight="bold", linespacing=1.4)
ax.add_patch(Rectangle((5.05, 5.35), 2.35, 1.05, fc=C_CELL, ec="#94a3b8", lw=1.1))
ax.text(6.225, 5.875, "192 idle: idx >= n", ha="center", va="center",
        fontsize=7.5, color="#475569")
ax.text(5.0, 5.02, "last block (3906): idx 999,936 … 1,000,191", ha="center",
        fontsize=7.5, color=C_MID)

ax.add_patch(Rectangle((0.2, 2.75), 9.6, 1.85, fc="#f8fafc", ec=C_GRID, lw=1.1))
ax.text(0.45, 4.4,
        "threads launched = 3907 × 256 = 1,000,192   (192 more than elements)\n"
        "    idx <  1,000,000  ->  64 write their c[idx] = a[idx] + b[idx]\n"
        "    idx >= 1,000,000  ->  192 return having done nothing at all",
        va="top", ha="left", fontsize=8.5, family="monospace", linespacing=1.65)
ax.text(5.0, 2.25,
        "no guard: 192 threads scribble past the end of c — corrupted memory",
        ha="center", fontsize=9, fontweight="bold", color=C_DRED)
ax.text(5.0, 1.65,
        "one integer compare buys correctness for every n that isn’t a multiple of 256",
        ha="center", fontsize=8.5, color=C_MID, style="italic")
ax.text(5.0, 1.1,
        "Python habit: it’s why you write range(n), never range(num_threads)",
        ha="center", fontsize=8.5, color=C_MID)
ax.set_title("C · The bounds check: 1,000,000 is not a multiple of 256", fontsize=10.5)

# ---------------- panel D: grid-stride loop ----------------
ax = a_stride
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.65)

ax.add_patch(Rectangle((0.15, 5.05), 9.7, 2.15, fc="#f8fafc", ec=C_GRID, lw=1.1))
ax.text(0.45, 6.95,
        "int idx    = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "int stride = blockDim.x * gridDim.x;   // total threads in the grid\n"
        "for (int i = idx; i < n; i += stride)\n"
        "    c[i] = a[i] + b[i];",
        va="top", ha="left", fontsize=8, family="monospace", linespacing=1.6)

ax.text(0.4, 4.75, "one thread per element: 20 threads, each does one add",
        ha="left", fontsize=8.5, fontweight="bold")
for i in range(20):
    x = 0.4 + i * 0.455
    ax.add_patch(Rectangle((x, 4.1), 0.42, 0.5, fc="#dbeafe", ec="#93c5fd", lw=0.5))
    ax.text(x + 0.21, 4.35, str(i), ha="center", va="center", fontsize=6.5, color=C_DBLUE)
    ax.text(x + 0.21, 3.95, f"t{i}", ha="center", va="center", fontsize=6, color=C_MID)

ax.text(0.4, 3.42, "grid-stride with 8 threads: stride = 8, thread t takes t, t+8, t+16, …",
        ha="left", fontsize=8.5, fontweight="bold")
thread_colors = ["#1e40af", "#2563eb", "#0891b2", "#059669",
                 "#d97706", "#ea580c", "#dc2626", "#7c3aed"]
for i in range(20):
    x = 0.4 + i * 0.455
    tcol = thread_colors[i % 8]
    ax.add_patch(Rectangle((x, 2.6), 0.42, 0.5, fc=tcol, ec="white", lw=0.5))
    ax.text(x + 0.21, 2.85, str(i), ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold")
    ax.text(x + 0.21, 2.44, f"t{i % 8}", ha="center", va="center",
            fontsize=6.5, color=tcol, fontweight="bold")

ax.text(0.4, 2.0,
        "thread 0 takes 0, 8, 16 · thread 1 takes 1, 9, 17 · threads 4–7: one element each",
        ha="left", fontsize=8.5, color=C_TEXT)
ax.text(5.0, 1.45,
        "no need for one thread per element — any grid size is now correct",
        ha="center", fontsize=9, fontweight="bold")
ax.text(5.0, 0.98,
        "fewer blocks, less scheduling overhead — often faster for big arrays",
        ha="center", fontsize=8.5, color=C_MID)
ax.text(5.0, 0.52,
        "like splitting a for-loop across workers — an H100 offers 270,336 resident threads",
        ha="center", fontsize=8.5, color=C_MID)
ax.set_title("D · Grid-stride loop: let each thread do several elements", fontsize=10.5)

# ---------------- panel E: host <-> device memory ----------------
ax = a_mem
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.65)

ax.add_patch(Rectangle((0.15, 4.7), 3.0, 2.15, fc="#f8fafc", ec=C_GRID, lw=1.2))
ax.text(0.35, 6.6, "HOST RAM", ha="left", fontsize=9.5, fontweight="bold", color=C_DBLUE)
ax.text(0.35, 6.15, "h_a, h_b, h_c\nfloat × 1M = 4 MB each\nplain RAM, GB/s",
        va="top", ha="left", fontsize=7.5, family="monospace", linespacing=1.6)
ax.add_patch(Rectangle((6.85, 4.7), 3.0, 2.15, fc="#f8fafc", ec=C_GRID, lw=1.2))
ax.text(7.05, 6.6, "GPU HBM (device)", ha="left", fontsize=9.5, fontweight="bold", color=C_DBLUE)
ax.text(7.05, 6.15, "d_a, d_b, d_c\n80 GB · 3.35 TB/s\nseparate memory chips",
        va="top", ha="left", fontsize=7.5, family="monospace", linespacing=1.6)

ax.annotate("", xy=(6.75, 6.05), xytext=(3.25, 6.05),
            arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=2))
ax.text(5.0, 6.42, "cudaMemcpy — PCIe ≈ 25 GB/s",
        ha="center", fontsize=8.5, fontweight="bold", color=C_MID)
ax.text(5.0, 5.62, "up: a, b (8 MB) · down: c (4 MB)",
        ha="center", fontsize=8, color=C_TEXT)
ax.text(5.0, 5.12, "~130× slower than HBM", ha="center", fontsize=8,
        fontweight="bold", color=C_RED)

ax.text(0.35, 4.35,
        "1  cudaMalloc(&d_a, ...)          allocate once on the GPU\n"
        "2  cudaMemcpy H2D  h_a,h_b->d_a   8 MB   ~320 µs\n"
        "3  vector_add<<<3907,256>>>(...)  12 MB  ~3.6 µs on HBM\n"
        "4  cudaMemcpy D2H  d_c->h_c       4 MB   ~160 µs",
        va="top", ha="left", fontsize=8.5, family="monospace", linespacing=1.65)
ax.text(5.0, 2.6, "~480 µs of copies vs ~3.6 µs of kernel — keep data on the GPU",
        ha="center", fontsize=9.5, fontweight="bold", color=C_DRED)

ax.add_patch(Rectangle((0.3, 0.55), 9.4, 1.55, fc="#fef3c7", ec=C_AMBER, lw=1.1))
ax.text(0.6, 1.92, "why the kernel itself is slow — the roofline view:",
        ha="left", fontsize=8.5, fontweight="bold", color="#92400e")
ax.text(0.6, 1.55,
        "12 bytes moved per element for 1 add  ->  AI = 1/12 ≈ 0.08 FLOP/byte\n"
        "the H100 ridge is ~295 FLOP/byte — this kernel sits 3,500× below it\n"
        "memory-bound: save bytes, not FLOPs — that’s every optimization in this book",
        va="top", ha="left", fontsize=8.5, linespacing=1.6)
ax.set_title("E · Two memories: the copies dwarf the kernel", fontsize=10.5)

# ---------------- panel F: python bindings + when to go custom ----------------
ax = a_py
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.05, 7.65)

pipe = [("cuda_sources\n(a Python string)", "#dbeafe"),
        ("nvcc JIT-compiles\n(first call ~30–60 s)", "#f8fafc"),
        ("cached .so in\n~/.cache/\ntorch_extensions", "#f8fafc"),
        ("module.add_cuda(a, b)\n= your kernel", "#d1fae5")]
for i, (label, fc) in enumerate(pipe):
    bx = 0.15 + i * 2.45
    ax.add_patch(Rectangle((bx, 6.15), 2.2, 1.0, fc=fc, ec=C_GRID, lw=1))
    ax.text(bx + 1.1, 6.65, label, ha="center", va="center", fontsize=6.5,
            linespacing=1.4)
    if i < 3:
        ax.annotate("", xy=(bx + 2.4, 6.65), xytext=(bx + 2.25, 6.65),
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.3))
ax.text(5.0, 5.88, "torch.utils.cpp_extension.load_inline — write CUDA inside Python, JIT on import",
        ha="center", fontsize=8.5, color=C_MID)

ax.add_patch(Rectangle((0.15, 3.15), 9.7, 2.4, fc="#f8fafc", ec=C_GRID, lw=1.1))
ax.text(0.45, 5.35,
        "from torch.utils.cpp_extension import load_inline\n"
        "module = load_inline(name='custom_add',\n"
        "    cpp_sources=cpp_source, cuda_sources=cuda_source,\n"
        "    functions=['add_cuda'])          # exported to Python\n"
        "a = torch.randn(1_000_000, device='cuda'); b = torch.randn_like(a)\n"
        "c = module.add_cuda(a, b)            # your kernel, from Python",
        va="top", ha="left", fontsize=7.6, family="monospace", linespacing=1.6)

ax.text(0.25, 2.82, "who should implement your op?", ha="left",
        fontsize=9, fontweight="bold")
for j, row in enumerate([
        "linear layers  -> cuBLAS (torch.nn.Linear)",
        "attention      -> FlashAttention / FlashInfer",
        "normalization  -> PyTorch defaults are fine",
        "uncovered fusions -> your custom kernel"]):
    last = j == 3
    ax.text(0.25, 2.48 - j * 0.34, row, ha="left", fontsize=8.5,
            fontweight="bold" if last else "normal",
            color="#b45309" if last else C_TEXT)
ax.plot([5.45, 5.45], [1.35, 2.9], color=C_GRID, lw=0.8)
ax.text(5.6, 2.82, "write custom only when:", ha="left",
        fontsize=9, fontweight="bold")
for j, row in enumerate([
        "1) fuse ops nothing else fuses",
        "2) a genuinely novel algorithm",
        "3) a profiler blames this kernel",
        "4) the speedup outlives the upkeep"]):
    ax.text(5.6, 2.48 - j * 0.34, row, ha="left", fontsize=8.5, color=C_TEXT)

ax.text(5.0, 0.75, "Default to libraries. Drop to custom CUDA only with evidence.",
        ha="center", fontsize=10, fontweight="bold")
ax.text(5.0, 0.28, "bugs in CUDA are harder to find than in Python — that cost is real",
        ha="center", fontsize=8, color=C_MID, style="italic")
ax.set_title("F · Call it from Python — and when NOT to write kernels", fontsize=10.5)

# free-floating labels are hand-placed in data coordinates; keep them out of
# constrained layout (texts default to in_layout=True, and long lines compound
# until the panels collapse) — panel sizes then come from the grid alone
for ax_ in fig.axes:
    for t in ax_.texts:
        t.set_in_layout(False)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hello-cuda.png")
fig.savefig(out, dpi=200, facecolor="white")
print(f"saved {out}")
