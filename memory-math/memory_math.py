# Memory math: understanding exactly where memory goes is essential for inference
# optimization. GPU memory is the scarcest resource, and the split between model weights,
# KV cache, and activations determines what's possible.
#
# The three terms of the budget:
#   weights      2 bytes x num_params                          fixed, always resident
#   KV cache     2 * layers * kv_heads * head_dim * 2 bytes    per token, scales with
#                (then x batch x seq_len)                      batch * seq_len - the
#                                                              compounding cost
#   activations  attention scores b * heads * s * s             O(s^2), needed only during
#                                                              the forward pass - at
#                                                              inference we free it at once
# This script computes every number from the config below and renders the chart.
# Change the config and re-run to see how the picture moves.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ----------------------------- config: a Llama-7B shaped model -----------------------------
vocab_size = 32000
hidden_dim = 4096
num_layers = 32
intermediate_dim = 11008
num_heads = 32            # query heads
kv_heads_mha = 32         # full MHA: every query head gets its own KV head
kv_heads_gqa = 8          # GQA: 8 KV heads shared across 32 query heads
head_dim = 128
bytes_per_param = 2       # fp16

# ----------------------------- derived: weights -----------------------------
emb_params = vocab_size * hidden_dim
qkvo_params_per_layer = 4 * hidden_dim ** 2
ffn_params_per_layer = 3 * hidden_dim * intermediate_dim
total_params = emb_params + num_layers * (qkvo_params_per_layer + ffn_params_per_layer)
# GB is decimal (1e9 bytes) to match the text's weight math: 6.6B params * 2 B = 13.2 GB.
# KB is binary (1024 B) because the per-token cache figures are exact powers of two.
GB = 10 ** 9
KB = 1024
emb_gb = emb_params * bytes_per_param / GB
qkvo_gb = num_layers * qkvo_params_per_layer * bytes_per_param / GB
ffn_gb = num_layers * ffn_params_per_layer * bytes_per_param / GB
weights_gb = total_params * bytes_per_param / GB

# ----------------------------- derived: KV cache -----------------------------
def kv_bytes_per_token(kv_heads):
    # K and V, every layer: 2 * layers * kv_heads * head_dim * 2 bytes
    return 2 * num_layers * kv_heads * head_dim * bytes_per_param

mha_kb_per_token = kv_bytes_per_token(kv_heads_mha) / 1024
gqa_kb_per_token = kv_bytes_per_token(kv_heads_gqa) / 1024

def kv_cache_gb(seq_len, batch, kb_per_token):
    return kb_per_token * KB * seq_len * batch / GB

# ----------------------------- derived: activations -----------------------------
def scores_gb(seq_len, batch=1):
    # attention scores b * heads * s * s, fp16
    return batch * num_heads * seq_len ** 2 * bytes_per_param / GB

# ----------------------------- sanity checks against the text -----------------------------
assert round(weights_gb, 1) == 13.2, weights_gb
assert mha_kb_per_token == 512 and gqa_kb_per_token == 128
assert round(scores_gb(4096)) == 1
print(f"weights: {total_params/1e9:.1f}B params -> {weights_gb:.1f} GB at fp16")
print(f"KV per token: MHA {mha_kb_per_token:.0f} KB, GQA {gqa_kb_per_token:.0f} KB")
print(f"attention scores at s=4096, b=1: {scores_gb(4096):.2f} GB")

# ----------------------------- style -----------------------------
C_EMB, C_QKVO, C_FFN, C_TOTAL = "#93c5fd", "#3b82f6", "#1e40af", "#334155"
C_MHA, C_GQA, C_SCORES = "#f59e0b", "#10b981", "#ef4444"
C_FREE, C_TEXT, C_GRID = "#e2e8f0", "#0f172a", "#cbd5e1"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
(a_wt, a_kv, a_grow), (a_act, a_gpu, a_txt) = axes
fig.suptitle("LLM Inference Memory Math — a 7B model at fp16\n"
             "Weights are the fixed cost · the KV cache is the compounding one · attention scores are the prefill spike",
             fontsize = 15, fontweight = "bold", linespacing = 1.5)

# ---------------- panel A: weight breakdown (waterfall) ----------------
labels = ["Embedding", "QKVO proj\n× 32 layers", "FFN\n× 32 layers", "Total\nweights"]
vals = [emb_gb, qkvo_gb, ffn_gb]
cum = [0, emb_gb, emb_gb + qkvo_gb]
colors = [C_EMB, C_QKVO, C_FFN, C_TOTAL]
for i, (lab, v) in enumerate(zip(labels, vals + [weights_gb])):
    bottom = cum[i] if i < 3 else 0
    a_wt.bar(i, v, bottom=bottom, width=0.62, color=colors[i])
    a_wt.text(i, bottom + v + 0.25, (f"+{v:.2f}" if i < 3 else f"{v:.1f}") + " GB",
              ha="center", fontsize=9.5, fontweight="bold")
    if i < 3 and i < 2:
        a_wt.plot([i + 0.31, i + 1 - 0.31], [cum[i + 1]] * 2, ls="--", lw=1, color="#94a3b8")
a_wt.set_xticks(range(4), labels, fontsize=9)
a_wt.set_ylim(0, 15.5)
a_wt.set_ylabel("GB at fp16 (2 bytes/param)")
a_wt.set_title(f"A · Model weights: {weights_gb:.1f} GB — the fixed cost\n"
               f"{total_params/1e9:.1f}B params, resident for the model's lifetime", fontsize=10.5)

# ---------------- panel B: KV cache per token, MHA vs GQA ----------------
bars = a_kv.bar(["MHA\n32 KV heads", "GQA\n8 KV heads"], [mha_kb_per_token, gqa_kb_per_token],
                width=0.5, color=[C_MHA, C_GQA])
for b, v in zip(bars, [mha_kb_per_token, gqa_kb_per_token]):
    a_kv.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:.0f} KB", ha="center",
              fontweight="bold", fontsize=11)
a_kv.annotate("4× smaller", xy=(1, 250), xytext=(0.42, 380), fontsize=11, fontweight="bold",
              color=C_GQA, arrowprops=dict(arrowstyle="->", color=C_GQA, lw=1.6))
a_kv.set_ylim(0, 580)
a_kv.set_ylabel("KB per token (all 32 layers)")
a_kv.set_title("B · KV cache per token\n2 × layers × kv_heads × head_dim × 2 bytes", fontsize=10.5)

# ---------------- panel C: cache growth vs sequence length ----------------
s = np.linspace(256, 32768, 400)
for batch, shade in [(1, "#a7f3d0"), (8, "#34d399"), (16, C_GQA)]:
    a_grow.plot(s, kv_cache_gb(s, batch, gqa_kb_per_token), color=shade, lw=2,
                label=f"GQA · batch {batch}")
a_grow.plot(s, kv_cache_gb(s, 16, mha_kb_per_token), color=C_MHA, lw=2, ls="--",
            label="MHA · batch 16")
h100_free = 80 - weights_gb
a_grow.axhline(h100_free, color="#64748b", ls=":", lw=1.5)
a_grow.text(700, h100_free + 2.2, f"H100 usable after weights ({h100_free:.0f} GB)",
            fontsize = 9, color = "#475569",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
a_grow.plot([32768], [kv_cache_gb(32768, 16, gqa_kb_per_token)], "o", color=C_GQA, ms=5)
a_grow.annotate(f"batch 16 × 32K ≈ {kv_cache_gb(32768, 16, gqa_kb_per_token):.0f} GB —\nfills an H100 all by itself",
                xy=(32768, kv_cache_gb(32768, 16, gqa_kb_per_token)),
                xytext=(15200, 84), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1))
a_grow.set_xlim(0, 33500)
a_grow.set_ylim(0, 100)
a_grow.set_yticks([0, 20, 40, 60, 80, 100])
a_grow.set_xticks([0, 8192, 16384, 24576, 32768], ["0", "8K", "16K", "24K", "32K"])
a_grow.set_xlabel("sequence length (tokens)")
a_grow.set_ylabel("KV cache (GB)")
a_grow.legend(fontsize=9, loc="upper left", framealpha=0.95)
a_grow.set_title("C · Cache grows linearly: batch × seq_len\n512 KB/token (MHA) vs 128 KB/token (GQA)", fontsize=10.5)

# ---------------- panel D: activation memory, O(s^2) ----------------
s2 = np.linspace(1024, 65536, 400)
a_act.loglog(s2, [scores_gb(x) for x in s2], color=C_SCORES, lw=2,
             label="attention scores  b·heads·s²  (O(s²))")
a_act.loglog(s2, [kv_cache_gb(x, 1, gqa_kb_per_token) for x in s2], color=C_GQA, lw=2,
             label="KV cache · GQA · batch 1  (O(s))")
for seq in (4096, 32768):
    a_act.plot([seq, seq], [scores_gb(seq), kv_cache_gb(seq, 1, gqa_kb_per_token)],
               color="#94a3b8", ls=":", lw=1)
    a_act.annotate(f"{scores_gb(seq):.0f} GB" if seq >= 32768 else f"{scores_gb(seq):.1f} GB",
                   xy=(seq, scores_gb(seq)), xytext=(seq * 1.15, scores_gb(seq) * 1.25),
                   fontsize=9, color=C_SCORES, fontweight="bold")
    a_act.annotate(f"{kv_cache_gb(seq, 1, gqa_kb_per_token):.1f} GB" if seq < 8192
                   else f"{kv_cache_gb(seq, 1, gqa_kb_per_token):.0f} GB",
                   xy=(seq, kv_cache_gb(seq, 1, gqa_kb_per_token)),
                   xytext=(seq * 1.15, kv_cache_gb(seq, 1, gqa_kb_per_token) * 0.5),
                   fontsize=9, color="#047857", fontweight="bold")
a_act.set_xlim(1024, 65536 * 1.45)
a_act.set_xticks([1024, 4096, 16384, 65536], ["1K", "4K", "16K", "64K"])
a_act.minorticks_off()
a_act.set_xlabel("sequence length (tokens), batch 1")
a_act.set_ylabel("GB")
a_act.legend(fontsize=10, loc="lower right", framealpha=0.95)
a_act.set_title("D · Activations: scores are the O(s²) spike\nfreed right after attention at inference", fontsize=10.5)

# ---------------- panel E: what fits on the GPU ----------------
gpus = [("H100", 80), ("B200", 192)]
for i, (name, total_gb) in enumerate(gpus):
    free = total_gb - weights_gb
    a_gpu.barh(i, weights_gb, height=0.5, color=C_QKVO)
    a_gpu.barh(i, free, left=weights_gb, height=0.5, color=C_FREE)
    n_4k_mha = int(free * GB / (mha_kb_per_token * KB * 4096))
    n_4k_gqa = int(free * GB / (gqa_kb_per_token * KB * 4096))
    n_16k_mha = int(free * GB / (mha_kb_per_token * KB * 16384))
    tok_mha = free * GB / (mha_kb_per_token * KB) / 1000
    if name == "H100":
        note = (f"{free:.0f} GB free ≈ {tok_mha:.0f}K tokens (MHA)\n"
                f"{n_4k_mha} × 4K-token seqs · ~{n_4k_gqa} with GQA")
    else:
        note = (f"{free:.0f} GB free ≈ {tok_mha:.0f}K tokens (MHA)\n"
                f"{n_4k_mha} × 4K or {n_16k_mha} × 16K seqs\n~{n_4k_gqa} × 4K with GQA")
    a_gpu.text(weights_gb + free / 2, i, note, ha="center", va="center", fontsize=9)
a_gpu.set_yticks(range(2), [f"{n} · {t} GB" for n, t in gpus])
a_gpu.set_xlim(0, 200)
a_gpu.set_ylim(-0.55, 1.6)
a_gpu.grid(axis="y", visible=False)
a_gpu.set_xlabel("GB")
a_gpu.set_title("E · What fits: capacity after the fixed cost", fontsize=10.5)
a_gpu.legend(handles=[Patch(color=C_QKVO, label=f"weights ({weights_gb:.1f} GB)"),
                      Patch(color=C_FREE, label="free memory")],
             fontsize=9, loc="lower right", framealpha=0.95)

# ---------------- panel F: the budget, in words ----------------
a_txt.axis("off")
a_txt.grid(False)
rows = [
    ("total GPU memory = weights + KV cache + activations", 11.5, "bold", C_TEXT, False, 0.07),
    ("weights      2 B × 6.6 B params   →  13.2 GB fixed", 10, "normal", C_QKVO, True, 0.12),
    ("KV cache     batch × seq × 512 KB/token (MHA) · 128 (GQA)", 10, "normal", C_GQA, True, 0.12),
    ("activations  scores b·heads·s²  O(s²) — prefill only", 10, "normal", C_SCORES, True, 0.12),
    ("             freed right after attention at inference", 10, "normal", C_TEXT, True, 0.07),
    ("H100 80 GB   13 + 67 free → ~31 × 4K seqs · ~124 GQA", 10, "normal", C_TEXT, True, 0.12),
    ("B200 192 GB  13 + 179 free → ~83 × 4K or ~20 × 16K", 10, "normal", C_TEXT, True, 0.07),
    ("Memory is the constraint. Scheduling, batching, caching —", 11, "bold", C_TEXT, False, 0.12),
    ("everything optimizes around these numbers.", 11, "bold", C_TEXT, False, 0.0),
]
y = 0.97
for text, size, weight, color, mono, gap in rows:
    a_txt.text(0.02, y, text, fontsize=size, fontweight=weight, color=color, va="top",
               family="monospace" if mono else None)
    y -= gap
a_txt.set_title("F · The budget", fontsize=10.5)

fig.savefig("memory-math.png", dpi=200, facecolor="white")
print("saved memory-math.png")
