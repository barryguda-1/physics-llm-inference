"""Roofline model visualization for LLM inference: H100 SXM vs B200.

H100 SXM (HBM3, 3.35 TB/s):
  - dense fp16:  ~990 TFLOPS  -> ridge ~295 FLOPs/byte
  - sparse fp16: ~1979 TFLOPS -> ridge ~590 FLOPs/byte
B200 (Blackwell, HBM3e, ~8 TB/s):
  - dense FP8: ~2250 TFLOPS -> ridge ~280 FLOPs/byte
  (FP4 omitted: these GEMMs don't run dense FP4.)

Run:  .venv/bin/python roofline.py     (writes roofline.png next to this file)

Note: B200 figures are from secondary sources; verify against an NVIDIA
datasheet before publishing.
"""
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent / "roofline.png"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Roofline:
    """One GPU ceiling: bandwidth plus one or more compute peaks.

    Bandwidth in TB/s, peaks in TFLOPS, so bandwidth * intensity (FLOPs/byte)
    comes out in TFLOPS directly.
    """
    gpu: str
    bandwidth: float                      # TB/s
    peaks: dict                           # {precision label: peak TFLOPS}
    line_style: str = "-"
    line_color: str = "#1f77b4"

    def achievable(self, intensity: float, precision: str) -> float:
        """TFLOPs attainable at a given arithmetic intensity."""
        return min(self.bandwidth * intensity, self.peaks[precision])

    def ridge(self, precision: str) -> float:
        """Intensity (FLOPs/byte) where memory and compute limits meet."""
        return self.peaks[precision] / self.bandwidth

    @property
    def top_peak(self) -> float:
        return max(self.peaks.values())


GPUS = [
    Roofline("H100 SXM", 3.35,
             {"dense fp16": 990.0, "sparse fp16": 1979.0},
             line_style="-", line_color="#1f77b4"),
    Roofline("B200", 8.0,
             {"dense FP8": 2250.0},
             line_style="--", line_color="#d62728"),
]
BASELINE = ("H100 SXM", "dense fp16")    # roofline the operation points use


@dataclass(frozen=True)
class Op:
    """An inference operation and its arithmetic intensity."""
    name: str
    intensity: float                      # FLOPs/byte


OPS = [
    Op("Decode GEMV (b=1)",    0.5),
    Op("Decode b=8",           4.0),
    Op("Decode b=32",         16.0),
    Op("Decode b=128",        64.0),
    Op("Decode b=512",       256.0),
    Op("Prefill seq=2048",  1024.0),
]


def roofline_analysis(flops, bytes_accessed, peak_flops, bandwidth):
    """The chapter's own analysis function: where an operation lands.

    flops/peak_flops in FLOPs (any unit); bandwidth in bytes/s scaled to
    match peak_flops' unit, e.g. TF/s for TFLOPS peaks.
    Returns (intensity, achievable, bound, utilization).
    """
    intensity = flops / bytes_accessed
    memory_bound = bandwidth * intensity
    achievable = min(memory_bound, peak_flops)
    bound = "memory" if memory_bound < peak_flops else "compute"
    return intensity, achievable, bound, achievable / peak_flops


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_roofline(ax, gpus, baseline):
    """Left panel: log-log rooflines with per-precision ridge markers."""
    base_gpu, base_prec = baseline
    base = next(g for g in gpus if g.gpu == base_gpu)

    x = np.logspace(-1, 4.2, 500)
    for gpu in gpus:
        ax.plot(x, np.minimum(gpu.bandwidth * x, gpu.top_peak),
                gpu.line_style, color=gpu.line_color, lw=2.2,
                label=f"{gpu.gpu} ({gpu.bandwidth:g} TB/s, "
                      f"peak {gpu.top_peak:g} TF)")
        for prec, peak in gpu.peaks.items():
            ridge = gpu.ridge(prec)
            ax.axvline(ridge, ls=":", color=gpu.line_color, lw=1.2,
                       alpha=0.8)
            ax.text(ridge * 1.1, 0.05 if gpu.gpu == "B200" else 0.02,
                    f"{gpu.gpu} ridge ({prec}): {ridge:.0f}",
                    fontsize=7.5, color=gpu.line_color, rotation=90,
                    va="bottom")

    ax.text(1.2, 4, "Memory-bound slope (BW × intensity)", fontsize=9,
            color="dimgray", rotation=33)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(OPS)))
    for op, c in zip(OPS, colors):
        p = base.achievable(op.intensity, base_prec)
        ax.plot(op.intensity, p, "o", ms=9, color=c, zorder=5)
        ax.annotate(
            f"{op.name}\n{op.intensity:g} F/B → {p:g} TF "
            f"({p / base.peaks[base_prec] * 100:.1f}%)",
            (op.intensity, p), textcoords="offset points",
            xytext=(10, -4 if op.intensity < base.ridge(base_prec) else -22),
            fontsize=8, color=c)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Achievable performance (TFLOPS)")
    ax.set_title("Roofline: H100 (dense fp16) vs B200 — decode stays "
                 "memory-bound")
    ax.set_xlim(0.3, 30000)
    ax.set_ylim(0.01, 6000)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)


def plot_utilization(ax, gpus, baseline):
    """Right panel: utilization of each op on each GPU/precision."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(OPS)))
    labels, utils, bar_colors = [], [], []
    for op, c in zip(OPS, colors):
        for gpu in gpus:
            for prec, peak in gpu.peaks.items():
                labels.append(f"{op.name} — {gpu.gpu} {prec}")
                utils.append(gpu.achievable(op.intensity, prec) / peak * 100)
                bar_colors.append(c)

    bars = ax.barh(range(len(utils)), utils, color=bar_colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Compute utilization (% of that GPU's peak, log scale)")
    ax.set_title("Compute utilization by GPU & precision")
    for bar, u in zip(bars, utils):
        ax.text(bar.get_width() * 1.25, bar.get_y() + bar.get_height() / 2,
                f"{u:.2f}%" if u < 1 else f"{u:.1f}%", va="center",
                fontsize=7)
    ax.set_xlim(0.05, 400)
    ax.grid(True, axis="x", which="both", alpha=0.25)
    ax.invert_yaxis()


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_roofline(ax1, GPUS, BASELINE)
    plot_utilization(ax2, GPUS, BASELINE)
    fig.suptitle("Batching raises arithmetic intensity; new GPUs move the "
                 "ridge point right, so decode remains bandwidth-bound",
                 fontsize=12, y=0.99)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
