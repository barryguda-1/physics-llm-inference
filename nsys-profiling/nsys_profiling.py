# Profiling with nsys: before optimizing anything, profile it -- intuition
# about bottlenecks is usually wrong. One sheet: the nsys workflow, reading
# the `nsys stats` kernel table, reading the GUI timeline, the three classic
# problem patterns (gaps between kernels, blocking memcpys, serialized
# streams), NVTX ranges, and the four metrics worth extracting.
# README.md in this folder adds the field guide: command reference plus the
# recipes for profiling a KServe-deployed predictor pod (vLLM /start_profile
# endpoints, nsys-wrapped entrypoint, capture windows, k8s gotchas).
#
# The chapter's stats table, checked (Instances x Avg must reconstruct Total):
#   ampere_fp16_s16816gemm   45.2%  892.1 ms   1024 x 871.2 us = 892.2 ms
#   void flash_fwd_kernel    23.1%  456.3 ms   1024 x 445.6 us = 456.3 ms
#   vectorized_elementwise   12.4%  244.8 ms   2048 x 119.5 us = 244.7 ms
#   void rms_norm_kernel      8.2%  161.9 ms   1024 x 158.1 us = 161.9 ms
#   listed share 88.9% -> implied total GPU kernel time 892.1/0.452 ~ 1974 ms,
#   everything else ~ 219 ms (11.1%). GEMM + attention = 68.3% of kernel time,
#   so rms_norm tuning is capped at 8.2% -- the table sets the ceiling.
#   Kernel names decode as arch _ precision _ mma-tile _ op: s16816 is the
#   16x8x16 tensor-core mma instruction (A100-era fp16 GEMM).
#   Panel D arithmetic: 20 us kernels with ~30 us Python-dispatch gaps = 40%
#   busy; ~10 us gaps = 67%; CUDA-graph replay ~1 us gaps = 95% -- the same
#   window fits 7 vs 16 steps = 2.3x the tokens.
#   Panel E: 100 MB over PCIe at ~25 GB/s (the hello-cuda sheet's practical
#   figure) = 4.0 ms; the same bytes inside HBM at 3.35 TB/s = ~30 us -> 134x.
#   A synchronous cudaMemcpy blocks the CPU and empties the compute lanes.
#
# Online verification (checked Sep 2026):
#   developer.nvidia.com/nsight-systems — 2026.5.1 is the current release;
#     the product now also sells multi-node profiling with automatic
#     performance-limiter diagnosis, GPU metrics sampling (SM utilization,
#     tensor-core activity, DRAM/NVLink/PCIe throughput, occupancy), a
#     JupyterLab extension, and Python backtrace sampling.
#   Nsight Systems User Guide (CLI reference) —
#     --cuda-graph-trace: graph (default) traces whole CUDA graphs, node
#       shows each kernel inside the graph ("may cause significant runtime
#       overhead") — panel D's fix hides kernels unless you switch this on.
#     --capture-range=cudaProfilerApi: nsys collects ONLY between
#       cudaProfilerStart/Stop — the missing half of panel F's snippet;
#       --capture-range-end (default stop-shutdown) also takes repeat[:N].
#     --delay/-y and --duration/-d confirmed for skipping warmup.
#     On Blackwell+ --trace=cuda uses hardware tracing (HES) by default;
#       --trace=cuda-sw forces software tracing for MPS/MIG/vGPU/Cc.
#     nsys export --type also takes hdf, arrow, parquet, jsonl — sqlite
#       remains the SQL-friendly one shown in panel A.
#   docs.pytorch.org (2.14) cuda.html — torch.cuda.nvtx.range / range_push /
#     range_pop are current, nothing NVTX deprecated; torch.cuda.profiler
#     .start()/stop() are the thin wrappers over cudaProfilerStart/Stop.
#   The two flags are drawn, not just cited: panel D's mini-diagram shows the
#   graph-vs-node trace visibility; panel F's bar shows the capture window.
# Change the config and re-run to see the picture move. Label positions are
# tuned to pass check_layout.py (panel ~488x375 px @100dpi).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ----------------------------- config -----------------------------
# the chapter's example `nsys stats` output: (pct, total_ms, instances, avg_us, name)
ROWS = [
    (45.2, 892.1, 1024, 871.2, "ampere_fp16_s16816gemm"),
    (23.1, 456.3, 1024, 445.6, "void flash_fwd_kernel"),
    (12.4, 244.8, 2048, 119.5, "vectorized_elementwise"),
    (8.2,  161.9, 1024, 158.1, "void rms_norm_kernel"),
]
KERNEL_US, PY_GAP_US, CPP_GAP_US, GRAPH_GAP_US = 20.0, 30.0, 10.0, 1.0
MB, PCIE_GBPS, HBM_TBPS = 100.0, 25.0, 3.35    # same PCIe/HBM figures as hello-cuda

# ----------------------------- derived + sanity checks -----------------------------
for pct, tot, n, avg, name in ROWS:
    assert abs(n * avg / 1000 - tot) <= 0.15, name      # Instances x Avg ~= Total
listed_pct = round(sum(r[0] for r in ROWS), 1)          # 88.9
total_ms = ROWS[0][1] / (ROWS[0][0] / 100)              # 1973.7 implied
other_ms = total_ms - sum(r[1] for r in ROWS)           # ~218.6
other_pct = round(100 - listed_pct, 1)                  # 11.1
assert listed_pct == 88.9 and other_pct == 11.1
assert round(ROWS[0][0] + ROWS[1][0], 1) == 68.3        # GEMM + attention share

busy_py   = KERNEL_US / (KERNEL_US + PY_GAP_US)         # 40.0%
busy_cpp  = KERNEL_US / (KERNEL_US + CPP_GAP_US)        # 66.7%
busy_graph= KERNEL_US / (KERNEL_US + GRAPH_GAP_US)      # 95.2%
steps_py, steps_cpp, steps_graph = 7, 11, 16            # pairs that fit the strip
assert round(100 * busy_py) == 40 and round(100 * busy_cpp) == 67
assert round(100 * busy_graph) == 95 and round(steps_graph / steps_py, 1) == 2.3

pcie_ms = MB * 1e6 / (PCIE_GBPS * 1e9) * 1e3            # 4.0 ms
hbm_us  = MB * 1e6 / (HBM_TBPS * 1e12) * 1e6            # 29.9 us
assert round(pcie_ms, 1) == 4.0 and round(hbm_us) == 30 and round(pcie_ms * 1e3 / hbm_us) == 134

print(f"table: {listed_pct:.1f}% listed -> implied total GPU kernel time "
      f"{total_ms:,.0f} ms; everything else ~{other_ms:.0f} ms ({other_pct}%)")
print(f"priorities: GEMM + attention = 68.3% of kernel time; rms_norm capped at 8.2%")
print(f"gaps: 20 us kernels + {PY_GAP_US:.0f} us gaps = {100*busy_py:.0f}% busy; "
      f"+{CPP_GAP_US:.0f} us = {100*busy_cpp:.0f}%; CUDA graphs = {100*busy_graph:.0f}% "
      f"({steps_graph} vs {steps_py} steps in the window = 2.3x)")
print(f"memcpy: {MB:.0f} MB over PCIe = {pcie_ms:.1f} ms vs {hbm_us:.0f} us in HBM = 134x")
print("verified Sep 2026 (Nsight Systems 2026.5.1): --cuda-graph-trace=node sees inside "
      "graphs; --capture-range=cudaProfilerApi records only the marked window")

# ----------------------------- style -----------------------------
C_BLUE, C_DBLUE, C_GREEN, C_AMBER, C_RED = "#3b82f6", "#1e40af", "#10b981", "#f59e0b", "#ef4444"
C_TEXT, C_GRID, C_MID, C_CELL = "#0f172a", "#cbd5e1", "#64748b", "#e2e8f0"
C_LBLUE, C_DRED = "#93c5fd", "#b91c1c"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
(a_flow, a_stats, a_time), (a_gaps, a_copy, a_nvtx) = axes
fig.suptitle("Profiling with nsys — see where the time actually goes\n"
             "wrap the run → the stats table names the culprit · the timeline shows why · "
             "the three classic patterns · NVTX to mark your regions",
             fontsize=14.5, fontweight="bold", linespacing=1.5)

# ---------------- panel A: the workflow ----------------
ax = a_flow
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.65)

ax.text(0.3, 7.15, "$ nsys profile -o profile_output python inference.py",
        ha="left", fontsize=9, family="monospace", fontweight="bold", color=C_DBLUE)

ax.add_patch(Rectangle((0.15, 4.95), 9.7, 1.9, fc="#f8fafc", ec=C_GRID, lw=1.2))
ax.text(0.4, 6.6, "what just happened", ha="left", fontsize=9, fontweight="bold")
ax.text(0.4, 6.35,
        "nsys shadows your run: every CUDA API call, kernel launch,\n"
        "memcpy and NVTX range — each one timestamped\n"
        "overhead is small (single-digit %) — profile the real thing:\n"
        "production batch sizes and lengths, steady state not startup\n"
        "output: profile_output.nsys-rep — a complete, replayable trace",
        va="top", ha="left", fontsize=7.5, family="monospace", linespacing=1.42)

ax.annotate("", xy=(2.5, 4.78), xytext=(2.5, 4.93),
            arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.6))
ax.annotate("", xy=(7.5, 4.78), xytext=(7.5, 4.93),
            arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.6))

ax.add_patch(Rectangle((0.15, 3.4), 4.75, 1.35, fc="white", ec=C_DBLUE, lw=1.1))
ax.text(0.4, 4.5, "nsys stats profile_output.nsys-rep", ha="left",
        fontsize=7.5, family="monospace", fontweight="bold", color=C_DBLUE)
ax.text(0.4, 4.24,
        "text summaries on the terminal:\n"
        "the kernel table (panel B), API call\ncounts, memcpy totals",
        va="top", ha="left", fontsize=7.5, color=C_TEXT, linespacing=1.42)

ax.add_patch(Rectangle((5.1, 3.4), 4.75, 1.35, fc="white", ec=C_GREEN, lw=1.1))
ax.text(5.35, 4.5, "the Nsight Systems GUI", ha="left", fontsize=8.5,
        fontweight="bold", color="#047857")
ax.text(5.35, 4.24,
        "the full timeline: CPU, CUDA API,\n"
        "kernels on their streams, transfers\n— panels C onward",
        va="top", ha="left", fontsize=7.5, color=C_TEXT, linespacing=1.42)

ax.add_patch(Rectangle((0.15, 2.12), 9.7, 1.18, fc="#f8fafc", ec=C_GRID, lw=1.1))
ax.text(0.4, 3.12, "ground rules:", ha="left", fontsize=8.5, fontweight="bold")
ax.text(0.4, 2.98,
        "warm up first — skip CUDA init and JIT (--delay/--duration do it for you)\n"
        "change one thing, re-profile — keep a baseline report to diff against\n"
        "CUDA graphs hide their kernels — see inside with --cuda-graph-trace=node",
        va="top", ha="left", fontsize=7.5, linespacing=1.42)
ax.text(0.3, 1.98, "power move: nsys export --type sqlite  ->  query every table with SQL",
        ha="left", fontsize=7, family="monospace", color=C_MID)

ax.text(5.0, 1.4, "before optimizing anything — profile it",
        ha="center", fontsize=9.5, fontweight="bold")
ax.text(5.0, 0.9, "where the time actually goes is rarely where you think it goes",
        ha="center", fontsize=8.5, color=C_MID, style="italic")
ax.set_title("A · The workflow — wrap your script, get a report", fontsize=10.5)

# ---------------- panel B: the stats table ----------------
ax = a_stats
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.65)

ax.text(5.0, 7.15, "$ nsys stats profile_output.nsys-rep", ha="center",
        fontsize=9, family="monospace", fontweight="bold", color=C_DBLUE)
ax.text(9.8, 4.75, "the chapter's example output", ha="right", fontsize=6.5, color=C_MID)

table = (" Time(%)   Total Time   Instances      Avg  Name\n"
         " -------   ----------   ---------   ------  -----------------------\n")
for pct, tot, n, avg, name in ROWS:
    table += (f" {pct:7.1f}   {tot:8.1f} ms   {n:8}   {avg:5.1f} us  {name}\n")
table += (f" {other_pct:7.1f}   ~{other_ms:4.0f} ms        ---     ---  "
          f"everything else (many tiny)")
ax.text(0.3, 7.0, table[:-1], va="top", ha="left", fontsize=7,
        family="monospace", linespacing=1.5)

BAR_X0, BAR_SCALE, BAR_H, BAR_PITCH = 3.3, 0.10, 0.34, 0.52
bar_rows = [(ROWS[0][0], ROWS[0][4], C_BLUE, "1024 x 871.2 us"),
            (ROWS[1][0], ROWS[1][4], C_GREEN, "1024 x 445.6 us"),
            (ROWS[2][0], ROWS[2][4], C_LBLUE, "2048 x 119.5 us"),
            (ROWS[3][0], ROWS[3][4], C_AMBER, "1024 x 158.1 us"),
            (other_pct, "everything else", "#cbd5e1", "many tiny kernels")]
for k, (pct, name, color, note) in enumerate(bar_rows):
    y = 4.05 - k * BAR_PITCH
    ax.text(0.3, y + BAR_H / 2, name, ha="left", va="center",
            fontsize=7, family="monospace", color=C_TEXT)
    ax.add_patch(Rectangle((BAR_X0, y), pct * BAR_SCALE, BAR_H,
                           fc=color, ec="white" if color != "#cbd5e1" else "#94a3b8", lw=0.6))
    ax.text(BAR_X0 + pct * BAR_SCALE - 0.08, y + BAR_H / 2, f"{pct:.1f}%",
            ha="right", va="center", fontsize=7.5, fontweight="bold", color="white")
    ax.text(9.8, y + BAR_H / 2, note, ha="right", va="center",
            fontsize=6.8, family="monospace", color=C_MID)

ax.text(5.0, 1.2, "GEMM + attention = 68.3% — that's where optimization actually pays",
        ha="center", fontsize=8, fontweight="bold")
ax.text(5.0, 0.8, "rms_norm tuning is capped at 8.2% — the table sets the ceiling",
        ha="center", fontsize=7.5, color=C_MID)
ax.text(5.0, 0.45, "1024 x 871.2 us ~ 892 ms — Instances x Avg always reconstructs Total",
        ha="center", fontsize=7, family="monospace", color=C_MID)
ax.set_title("B · The stats table — it names the culprit", fontsize=10.5)

# ---------------- panel C: the timeline ----------------
ax = a_time
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.65)

# nvtx brackets
for x0, x1, label in [(1.75, 4.92, "nvtx range: prefill"), (5.0, 9.8, "nvtx range: decode")]:
    ax.plot([x0, x1], [6.78, 6.78], color="#b45309", lw=1.4)
    ax.plot([x0, x0], [6.68, 6.78], color="#b45309", lw=1.4)
    ax.plot([x1, x1], [6.68, 6.78], color="#b45309", lw=1.4)
    ax.text((x0 + x1) / 2, 6.9, label, ha="center", va="bottom",
            fontsize=7, fontweight="bold", color="#b45309")

def lane(y0, h, label):
    ax.add_patch(Rectangle((0.15, y0), 9.7, h, fc="#f8fafc", ec=C_GRID, lw=1))
    ax.plot([1.62, 1.62], [y0, y0 + h], color=C_CELL, lw=0.8)
    ax.text(0.88, y0 + h / 2, label, ha="center", va="center",
            fontsize=6.5, color=C_DBLUE, fontweight="bold", linespacing=1.4)

lane(5.95, 0.6, "CPU\npython+torch")
for i in range(4):                                    # prefill dispatch
    ax.add_patch(Rectangle((1.85 + i * 0.68, 6.12), 0.55, 0.26, fc=C_DBLUE, ec="white", lw=0.4))
for i in range(11):                                   # decode dispatch
    ax.add_patch(Rectangle((4.7 + i * 0.42, 6.12), 0.3, 0.26, fc=C_DBLUE, ec="white", lw=0.4))

lane(5.15, 0.6, "CUDA API\ndriver calls")
for i in range(4):
    ax.add_patch(Rectangle((1.91 + i * 0.68, 5.42), 0.5, 0.13, fc="#60a5fa", ec="none"))
for i in range(11):
    ax.add_patch(Rectangle((4.62 + i * 0.42, 5.42), 0.26, 0.13, fc="#60a5fa", ec="none"))

lane(3.85, 1.2, "GPU\nstream 7")
for x0, x1, color, label in [(1.75, 2.70, C_BLUE, "gemm"), (2.80, 3.50, C_GREEN, "flash"),
                             (3.60, 4.50, C_BLUE, "gemm")]:
    ax.add_patch(Rectangle((x0, 4.0), x1 - x0, 0.9, fc=color, ec="white", lw=0.6))
    ax.text((x0 + x1) / 2, 4.45, label, ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold")
ax.add_patch(Rectangle((4.58, 4.25), 0.34, 0.4, fc=C_LBLUE, ec="white", lw=0.5))
dec_colors = [C_BLUE, C_GREEN, C_AMBER, C_LBLUE]
for i in range(11):
    ax.add_patch(Rectangle((5.05 + i * 0.405, 4.25), 0.27, 0.5,
                           fc=dec_colors[i % 4], ec="white", lw=0.4))

lane(3.1, 0.6, "MEMORY\nPCIe copies")
ax.add_patch(Rectangle((1.75, 3.28), 0.7, 0.24, fc="#94a3b8", ec="none"))
ax.text(2.1, 3.4, "H2D", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
ax.add_patch(Rectangle((9.3, 3.28), 0.5, 0.24, fc="#64748b", ec="none"))
ax.text(9.55, 3.4, "D2H", ha="center", va="center", fontsize=6, color="white", fontweight="bold")

ax.annotate("", xy=(9.8, 2.82), xytext=(1.75, 2.82),
            arrowprops=dict(arrowstyle="-|>", color=C_MID, lw=1.2))
ax.text(5.75, 2.5, "time — one forward pass, schematic", ha="center",
        fontsize=7, color=C_MID, style="italic")

ax.annotate("", xy=(5.1, 4.15), xytext=(4.75, 5.35),
            arrowprops=dict(arrowstyle="-|>", color=C_MID, lw=0.9, ls="--"))
ax.text(0.3, 2.15,
        "top lanes = what was asked for: CPU ops and driver API calls",
        fontsize=7.5, color=C_TEXT, ha="left")
ax.text(0.3, 1.83, "GPU lane = what actually ran — in launch order, one stream at a time",
        fontsize=7.5, color=C_TEXT, ha="left")
ax.text(0.3, 1.51, "memory lane = every byte over PCIe — big blocks are a red flag (panel E)",
        fontsize=7.5, color=C_TEXT, ha="left")
ax.text(5.0, 0.95, "the GPU row is the truth — everything above it explains it",
        ha="center", fontsize=8.5, fontweight="bold")
ax.text(5.0, 0.5, "prefill shows as a few big kernels, decode as swarms of small ones",
        ha="center", fontsize=7.5, color=C_MID, style="italic")
ax.set_title("C · The timeline — four lanes, one truth", fontsize=10.5)

# ---------------- panel D: pattern 1, gaps ----------------
ax = a_gaps
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.65)

ax.text(5.0, 7.15, "symptom: a GPU lane that is mostly white — short kernels, long gaps",
        ha="center", fontsize=8, family="monospace", fontweight="bold", color=C_DBLUE)

def strip(y0, label, timing, k_w, g_w, pairs, busy, busy_color, steps):
    ax.add_patch(Rectangle((0.3, y0), 9.55, 0.92, fc="#f8fafc", ec=C_GRID, lw=1.1))
    ax.text(0.45, y0 + 1.04, label, ha="left", fontsize=8, fontweight="bold")
    ax.text(9.65, y0 + 0.76, timing, ha="right", va="center", fontsize=6.5,
            family="monospace", color=C_MID)
    for i in range(pairs):
        ax.add_patch(Rectangle((0.55 + i * (k_w + g_w), y0 + 0.18), k_w, 0.42,
                               fc=C_BLUE, ec="white", lw=0.5))
    ax.text(8.95, y0 + 0.42, f"{100 * busy:.0f}% busy", ha="center", va="center",
            fontsize=10, fontweight="bold", color=busy_color)
    ax.text(8.95, y0 + 0.13, f"{steps} steps", ha="center", va="center",
            fontsize=7, color=C_MID)

strip(5.58, "eager decode from Python — dispatch through the framework",
      "kernels 20 us - gaps ~30 us", 0.46, 0.69, steps_py, busy_py, C_RED, steps_py)
strip(4.28, "less CPU in the loop — C++/compiled dispatch",
      "kernels 20 us - gaps ~10 us", 0.46, 0.23, steps_cpp, busy_cpp, C_AMBER, steps_cpp)
strip(2.98, "CUDA graphs — capture once, replay with a single launch",
      "kernels 20 us - gaps ~1 us", 0.46, 0.02, steps_graph, busy_graph, "#047857", steps_graph)

# what the profiler itself sees once the step is graphed (2026 flag check)
ax.text(0.3, 2.72, "once you CUDA-graph the step, tell nsys to still see the kernels",
        ha="left", fontsize=8, fontweight="bold")
ax.text(0.3, 2.33, "graph (default):", ha="left", va="center",
        fontsize=6.5, family="monospace", color=C_MID)
ax.add_patch(Rectangle((2.3, 2.16), 3.9, 0.34, fc="#94a3b8", ec="#475569", lw=0.8))
ax.text(4.25, 2.33, "cudaGraphLaunch — kernels hidden", ha="center", va="center",
        fontsize=6, color="white", fontweight="bold")
ax.text(6.4, 2.33, "kernels invisible in the timeline", ha="left", va="center",
        fontsize=7, color=C_RED)
ax.text(0.3, 1.86, "node:", ha="left", va="center",
        fontsize=6.5, family="monospace", color=C_MID)
for i in range(9):
    ax.add_patch(Rectangle((2.3 + i * 0.30, 1.69), 0.24, 0.34,
                           fc=dec_colors[i % 4], ec="white", lw=0.4))
ax.text(5.15, 1.86, "each kernel visible again — with some overhead", ha="left", va="center",
        fontsize=7, fontweight="bold", color="#047857")

ax.text(5.0, 1.05, "a kernel launch costs ~5–10 us of CPU; a Python torch op, tens of us",
        ha="center", fontsize=8, fontweight="bold")
ax.text(5.0, 0.7, "if kernels are as short as their gaps, the GPU starves — a CPU problem",
        ha="center", fontsize=7.5, color=C_MID)
ax.text(5.0, 0.35, f"the same window, {steps_graph / steps_py:.1f}x the tokens: the GPU was never the bottleneck",
        ha="center", fontsize=8, fontweight="bold", color=C_DBLUE)
ax.set_title("D · Pattern 1 — gaps between kernels (the GPU waits)", fontsize=10.5)

# ---------------- panel E: patterns 2 & 3, copies + streams ----------------
ax = a_copy
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.1, 7.65)

ax.text(5.0, 7.15, "symptom: one long cudaMemcpy stretches across every lane",
        ha="center", fontsize=8, family="monospace", fontweight="bold", color=C_DBLUE)

ax.add_patch(Rectangle((0.15, 5.85), 9.7, 0.7, fc="#f8fafc", ec=C_GRID, lw=1))
ax.text(0.25, 6.2, "CPU", ha="left", va="center", fontsize=7.5, fontweight="bold", color=C_DBLUE)
for x0, x1 in [(0.9, 1.5), (1.6, 2.0), (6.7, 7.3), (7.4, 8.0), (8.1, 8.7), (8.8, 9.5)]:
    ax.add_patch(Rectangle((x0, 6.05), x1 - x0, 0.4, fc=C_DBLUE, ec="white", lw=0.5))
ax.add_patch(Rectangle((2.1, 6.05), 4.5, 0.4, fc="#fee2e2", ec=C_RED, lw=1, hatch="//"))
ax.text(4.35, 6.25, "CPU blocked — a sync copy returns only when it is done",
        ha="center", va="center", fontsize=7, color=C_DRED,
        bbox=dict(fc="#fee2e2", ec="none", pad=2))

ax.add_patch(Rectangle((0.15, 4.95), 9.7, 0.7, fc="#f8fafc", ec=C_GRID, lw=1))
ax.text(0.25, 5.3, "GPU", ha="left", va="center", fontsize=7.5, fontweight="bold", color=C_DBLUE)
for x0, x1 in [(0.9, 1.5), (1.6, 2.0), (6.7, 7.3), (7.4, 8.0), (8.1, 8.7), (8.8, 9.5)]:
    ax.add_patch(Rectangle((x0, 5.15), x1 - x0, 0.4, fc=C_BLUE, ec="white", lw=0.5))
ax.add_patch(Rectangle((2.1, 5.15), 4.5, 0.4, fc="#94a3b8", ec="#475569", lw=1))
ax.text(4.35, 5.35, "cudaMemcpy D2H · 100 MB · ~4 ms — compute lanes empty",
        ha="center", va="center", fontsize=7, color="white", fontweight="bold")

ax.text(5.0, 4.55, "100 MB at ~25 GB/s over PCIe ~ 4.0 ms - in HBM: ~30 us — 134x",
        ha="center", fontsize=7.5, family="monospace", color=C_TEXT)
ax.text(5.0, 4.12, "fixes: cudaMemcpyAsync on its own stream - pinned memory - or don't copy at all",
        ha="center", fontsize=7, color=C_MID)

ax.text(0.25, 3.42, "cousin: independent kernels serialized on one stream",
        ha="left", fontsize=8.5, fontweight="bold")
ax.text(0.25, 3.02, "one stream", ha="left", va="center", fontsize=7.5, fontweight="bold")
ax.add_patch(Rectangle((1.7, 2.72), 1.7, 0.42, fc=C_BLUE, ec="white", lw=0.6))
ax.text(2.55, 2.93, "A", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.add_patch(Rectangle((3.55, 2.72), 1.7, 0.42, fc=C_GREEN, ec="white", lw=0.6))
ax.text(4.4, 2.93, "B", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.text(5.45, 2.93, "wall time = A + B", ha="left", va="center", fontsize=7.5, color=C_MID)

ax.text(0.25, 2.42, "two streams", ha="left", va="center", fontsize=7.5, fontweight="bold")
ax.text(1.55, 2.21, "s1", ha="right", va="center", fontsize=6.5, color=C_MID)
ax.add_patch(Rectangle((1.7, 2.02), 1.7, 0.38, fc=C_BLUE, ec="white", lw=0.6))
ax.text(2.55, 2.21, "A", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.text(1.55, 1.69, "s2", ha="right", va="center", fontsize=6.5, color=C_MID)
ax.add_patch(Rectangle((1.7, 1.5), 1.7, 0.38, fc=C_GREEN, ec="white", lw=0.6))
ax.text(2.55, 1.69, "B", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.text(3.6, 2.05, "wall time = max(A, B) if independent", ha="left", va="center",
        fontsize=7.5, fontweight="bold", color="#047857")

ax.text(5.0, 1.05, "every pattern here has the same silhouette: the GPU lanes go empty",
        ha="center", fontsize=7.5, fontweight="bold")
ax.text(5.0, 0.65, "find the empty spans first — then ask what caused each one",
        ha="center", fontsize=7.5, color=C_MID)
ax.set_title("E · Patterns 2 & 3 — blocking copies, serialized streams", fontsize=10.5)

# ---------------- panel F: nvtx + key metrics ----------------
ax = a_nvtx
ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(0.05, 7.65)

ax.add_patch(Rectangle((0.15, 3.5), 4.75, 3.8, fc="#f8fafc", ec=C_GRID, lw=1.1))
ax.text(0.35, 7.12,
        'from torch.cuda import nvtx\n'
        '\n'
        'with nvtx.range("prefill"):\n'
        '    output = model(input_ids)\n'
        'with nvtx.range("decode"):\n'
        '    for i in range(100):\n'
        '        output = model(token)\n'
        '\n'
        '# nsys: --capture-range=cudaProfilerApi\n'
        'torch.cuda.cudart().cudaProfilerStart()\n'
        'result = model(input)\n'
        'torch.cuda.cudart().cudaProfilerStop()',
        va="top", ha="left", fontsize=6.5, family="monospace", linespacing=1.42)
ax.text(0.25, 3.2, "brackets on the timeline (panel C)",
        ha="left", fontsize=7.5, color=C_MID)

ax.text(5.15, 7.1, "the four metrics worth extracting", ha="left", fontsize=8, fontweight="bold")
items = [
    ("1 - kernel duration",  "a few long kernels, or swarms of tiny", "ones? tiny -> fuse or CUDA-graph them"),
    ("2 - occupancy",        "are the SMs actually full while the",   "kernel runs? (GPU metrics sample it)"),
    ("3 - memory throughput","memory-bound kernels near the HBM",    "ceiling? then save bytes, not FLOPs"),
    ("4 - launch overhead",  "the white gaps: one kernel ends,",      "how long until the next begins? (D)"),
]
for k, (lead, d1, d2) in enumerate(items):
    y = 6.7 - k * 0.86
    ax.text(5.15, y, lead, ha="left", fontsize=8, fontweight="bold")
    ax.text(5.15, y - 0.27, d1, ha="left", fontsize=7, color=C_MID)
    ax.text(5.15, y - 0.5, d2, ha="left", fontsize=7, color=C_MID)

# what --capture-range=cudaProfilerApi records: only the marked window
ax.text(5.15, 3.38, "nsys --capture-range=cudaProfilerApi", ha="left",
        fontsize=6.5, family="monospace", color=C_DBLUE)
ax.add_patch(Rectangle((5.15, 3.02), 1.1, 0.2, fc=C_CELL, ec="#94a3b8", lw=0.5))
ax.text(5.7, 3.12, "skipped", ha="center", va="center", fontsize=5, color=C_MID)
ax.add_patch(Rectangle((6.35, 3.02), 2.2, 0.2, fc=C_GREEN, ec="none"))
ax.text(7.45, 3.12, "RECORDED", ha="center", va="center", fontsize=6,
        color="white", fontweight="bold")
ax.add_patch(Rectangle((8.65, 3.02), 1.1, 0.2, fc=C_CELL, ec="#94a3b8", lw=0.5))
ax.text(9.2, 3.12, "skipped", ha="center", va="center", fontsize=5, color=C_MID)
ax.text(6.45, 2.9, "cudaProfilerStart", ha="center", va="center",
        fontsize=5.5, family="monospace", color=C_MID)
ax.text(8.45, 2.9, "cudaProfilerStop", ha="center", va="center",
        fontsize=5.5, family="monospace", color=C_MID)

ax.add_patch(Rectangle((0.15, 0.9), 9.7, 1.9, fc="#fef3c7", ec=C_AMBER, lw=1.1))
ax.text(0.55, 2.5, "profile first — then optimize:", ha="left", fontsize=9,
        fontweight="bold", color="#92400e")
ax.text(0.55, 2.12,
        "the stats table says WHAT is slow — attack the top of it (panel B)\n"
        "the timeline says WHY — gaps, stalls, serialization (panels C-E)\n"
        "and intuition is usually wrong — that is the point of profiling",
        va="top", ha="left", fontsize=7.5, linespacing=1.42)
ax.set_title("F · NVTX ranges + the four metrics worth extracting", fontsize=10.5)

# free-floating labels are hand-placed in data coordinates; keep them out of
# constrained layout (texts default to in_layout=True, and long lines compound
# until the panels collapse) — panel sizes then come from the grid alone
for ax_ in fig.axes:
    for t in ax_.texts:
        t.set_in_layout(False)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nsys-profiling.png")
# write via buffer + os.replace: a straight savefig can hit Errno 22 on Windows
# when a viewer/upload still holds the previous png open
import io
import PIL.Image
buf = io.BytesIO()
fig.savefig(buf, dpi=200, facecolor="white", format="png")
buf.seek(0)
tmp = out + ".new.png"
PIL.Image.open(buf).save(tmp)
os.replace(tmp, out)
print(f"saved {out}")
