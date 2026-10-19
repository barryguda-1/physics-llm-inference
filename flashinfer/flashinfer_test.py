"""
flashinfer_test.py — run and verify the FlashInfer concepts from flashinfer.html
================================================================================

Runs top to bottom and prints as it goes:

  PART 0  environment check (GPU arch, versions)
  PART 1  build a ragged batch + a *scattered* paged KV cache
          (page table = kv_indices — same simulation as section 03 of the page)
  PART 2  BatchDecodeWithPagedKVCacheWrapper: plan() + run(), then verify the
          output against a plain-PyTorch float32 reference
  PART 3  BatchPrefillWithPagedKVCacheWrapper: causal prefill, verified the same way
  PART 4  the padded SDPA path: verified too, so the benchmark below compares two
          *correct* implementations that differ only in memory layout
  PART 5  the chapter's benchmark: batch decode, SDPA vs FlashInfer at B = 1/8/32,
          plus the "plan once, reuse across layers" measurement from section 06

Run on Google Colab
-------------------
1. Runtime -> Change runtime type -> GPU.  You need sm80+ (Ampere or newer):
   A100, L4, A10G all work.  The free-tier T4 is sm75 — NOT supported.
2. In a notebook cell:
       !pip install -q flashinfer-python
       !flashinfer download-kernels        # optional, prebuilt cubin cache (0.7+)
3. Then:
       !python flashinfer_test.py

Notes
-----
* The book's `begin_forward(...)` / `forward(...)` are `plan(...)` / `run(...)` since
  v0.2 — same CPU-plan / GPU-run split, new names.
* The first call per (head_dim, dtype, page_size) triggers a JIT compile or a
  cubin-wheel download. Warmup absorbs that one-time cost before benchmarking.
* The reference below runs in float32; the kernels run float16 with float32
  accumulation. Expect max |err| ~ 1e-3; the script asserts < 4e-2.
"""

import math
import random
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

DEV = "cuda"
FP16 = torch.float16

try:
    import flashinfer
except ImportError:
    sys.exit("flashinfer is not installed. On Colab run:  !pip install -q flashinfer-python")

# chapter config: GQA 32 query heads / 8 kv heads, head_dim 128, page size 16
HQ, HKV, D, PAGE = 32, 8, 128, 16
GROUP = HQ // HKV


def plan_compat(wrapper, **kw):
    """plan() kwargs grew over releases; drop dtype args if this build doesn't take them."""
    try:
        wrapper.plan(**kw)
    except TypeError:
        kw.pop("q_data_type", None)
        kw.pop("kv_data_type", None)
        wrapper.plan(**kw)


# ================================ PART 0 ================================
print("=" * 72)
print("PART 0 · environment")
print("=" * 72)
if not torch.cuda.is_available():
    sys.exit("No CUDA device. Colab: Runtime -> Change runtime type -> GPU.")
major, _ = torch.cuda.get_device_capability()
print(f"GPU          : {torch.cuda.get_device_name(0)} (sm{major}{torch.cuda.get_device_capability()[1]})")
print(f"torch        : {torch.__version__}")
print(f"flashinfer   : {flashinfer.__version__}")
if major < 8:
    sys.exit("FlashInfer needs sm80+ (Ampere). The free-tier Colab T4 is sm75 — switch to A100/L4.")

# one pre-allocated scratch buffer, shared by BOTH wrappers (section 06, step 1).
# sequential reuse is safe; two wrappers may not use it concurrently.
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)


# ================================ PART 1 ================================
def build_batch(lens, num_pages_total, scatter=True):
    """Allocate lens[b] tokens of KV in scattered pages; return the three arrays
    the wrapper plan() consumes, plus the cache itself.

    kv_data            [num_pages, 2, PAGE, HKV, D]  — dim 1 is (K, V)
    kv_indptr          [B+1]  cumulative page counts  — where each seq's pages live in kv_indices
    kv_indices         [total_pages]  physical page id of each *logical* page (the page table)
    last_page_len      [B]    tokens used in each seq's last page
    """
    B = len(lens)
    free = list(range(num_pages_total))
    if scatter:
        random.shuffle(free)  # allocations land wherever a free slot was — like section 03

    pages_per_seq = []
    for L in lens:
        n_pages = (L + PAGE - 1) // PAGE
        pages_per_seq.append([free.pop() for _ in range(n_pages)])

    kv_data = torch.randn(num_pages_total, 2, PAGE, HKV, D, dtype=FP16, device=DEV)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    kv_indices = torch.tensor(sum(pages_per_seq, []), dtype=torch.int32, device=DEV)
    last_page_len = torch.tensor([(L - 1) % PAGE + 1 for L in lens], dtype=torch.int32, device=DEV)
    for b, pages in enumerate(pages_per_seq):
        kv_indptr[b + 1] = kv_indptr[b] + len(pages)
    return pages_per_seq, kv_data, kv_indptr, kv_indices, last_page_len


def gather_kv(kv_data, pages, L):
    """Follow the page table to materialize one sequence's contiguous K/V — the
    gather the fused kernel does internally. Returns float32 [L, HKV, D]."""
    k = torch.cat([kv_data[p, 0] for p in pages], dim=0)[:L].float()
    v = torch.cat([kv_data[p, 1] for p in pages], dim=0)[:L].float()
    return k, v


def expand_gqa(x):
    """[L, HKV, D] -> [L, HQ, D]: each kv head feeds GROUP query heads."""
    return x.repeat_interleave(GROUP, dim=1)


print()
print("=" * 72)
print("PART 1 · a ragged batch in a scattered paged cache")
print("=" * 72)
lens = [521, 1307, 64, 2048]
B = len(lens)
NUM_PAGES = 512
pages_per_seq, kv_data, kv_indptr, kv_indices, last_page_len = build_batch(lens, NUM_PAGES)
for b, pages in enumerate(pages_per_seq):
    print(f"  seq {b}: {lens[b]:5d} tokens · {len(pages):3d} pages · kv_indices {pages}")
print(f"  free pages left: {NUM_PAGES - sum(len(p) for p in pages_per_seq)} / {NUM_PAGES}")
print("  kv_indptr       :", kv_indptr.tolist(), " (index into kv_indices per sequence)")
print("  last_page_len   :", last_page_len.tolist(), " (tokens used in each final page)")
waste = sum(len(p) for p in pages_per_seq) * PAGE - sum(lens)
print(f"  cache slots: {sum(len(p) for p in pages_per_seq) * PAGE} · real tokens: {sum(lens)} · waste: {waste} (≤ 1 page per seq)")


# ================================ PART 2 ================================
print()
print("=" * 72)
print("PART 2 · batch decode: plan + run, then verify vs a float32 reference")
print("=" * 72)
q = torch.randn(B, HQ, D, dtype=FP16, device=DEV)  # one new token per sequence

decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace)
plan_compat(
    decode,
    indptr=kv_indptr,                # was begin_forward(indptr, ...) in the book
    indices=kv_indices,              # the page table
    last_page_len=last_page_len,
    num_qo_heads=HQ,
    num_kv_heads=HKV,
    head_dim=D,
    page_size=PAGE,
    q_data_type=FP16,
)
out_fi = decode.run(q, kv_data)      # was decode.forward(...) — one fused launch
print(f"  output shape: {tuple(out_fi.shape)}  (B × HQ × D)")


def ref_decode(q, kv_data, pages_per_seq, lens):
    """Straightforward per-sequence GQA attention over gathered pages."""
    outs = []
    for b, L in enumerate(lens):
        k, v = gather_kv(kv_data, pages_per_seq[b], L)   # [L, HKV, D]
        k, v = expand_gqa(k), expand_gqa(v)              # [L, HQ, D]
        scores = (q[b].float() @ k.T) / math.sqrt(D)     # [HQ, L]
        probs = scores.softmax(dim=-1)
        outs.append(probs @ v)                           # [HQ, D]
    return torch.stack(outs)


out_ref = ref_decode(q, kv_data, pages_per_seq, lens)
err = (out_fi.float() - out_ref).abs().max().item()
print(f"  max |FlashInfer − reference| = {err:.2e}   (fp16 kernel vs fp32 math)")
assert err < 4e-2, "decode output does not match reference"
print("  ✓ PASS — the fused kernel computes the same attention over scattered pages")


# ================================ PART 3 ================================
print()
print("=" * 72)
print("PART 3 · causal prefill, verified the same way")
print("=" * 72)
qo_lens = [128, 512, 64, 2048]                 # prompt tokens per request
pages_p, kv_data_p, kv_indptr_p, kv_indices_p, lpl_p = build_batch(qo_lens, NUM_PAGES)
qo_indptr = torch.zeros(len(qo_lens) + 1, dtype=torch.int32, device=DEV)
for b, L in enumerate(qo_lens):
    qo_indptr[b + 1] = qo_indptr[b] + L        # ragged query layout — section 02
T_total = int(qo_indptr[-1])
q_prefill = torch.randn(T_total, HQ, D, dtype=FP16, device=DEV)

prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace)   # shares the scratch buffer
plan_compat(
    prefill,
    qo_indptr=qo_indptr,
    paged_kv_indptr=kv_indptr_p,
    paged_kv_indices=kv_indices_p,
    paged_kv_last_page_len=lpl_p,
    num_qo_heads=HQ,
    num_kv_heads=HKV,
    head_dim_qk=D,
    page_size=PAGE,
    causal=True,                               # row i attends to keys ≤ i
    q_data_type=FP16,
)
out_pf = prefill.run(q_prefill, kv_data_p)
print(f"  output shape: {tuple(out_pf.shape)}  ({T_total} query tokens, ragged)")


def ref_prefill(q_flat, kv_data, pages_per_seq, lens, qo_indptr):
    outs = []
    for b, L in enumerate(lens):
        qb = q_flat[qo_indptr[b]:qo_indptr[b + 1]].float()       # [T, HQ, D]
        k, v = gather_kv(kv_data, pages_per_seq[b], L)
        k, v = expand_gqa(k), expand_gqa(v)
        scores = (qb @ k.transpose(0, 1)) / math.sqrt(D)         # [T, T]
        T = qb.shape[0]
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        outs.append(scores.softmax(dim=-1) @ v)
    return torch.cat(outs, dim=0)


out_pf_ref = ref_prefill(q_prefill, kv_data_p, pages_p, qo_lens, qo_indptr)
err_pf = (out_pf.float() - out_pf_ref).abs().max().item()
print(f"  max |FlashInfer − reference| = {err_pf:.2e}")
assert err_pf < 4e-2, "prefill output does not match reference"
print("  ✓ PASS — prefill is the GEMM-shaped phase (section 04); same ragged+paged inputs")


# ================================ PART 4 ================================
print()
print("=" * 72)
print("PART 4 · the padded SDPA path — same math, different layout")
print("=" * 72)
S_max = max(lens)
k_pad = torch.zeros(B, HKV, S_max, D, dtype=FP16, device=DEV)   # dense [B, S, ...] — what
v_pad = torch.zeros(B, HKV, S_max, D, dtype=FP16, device=DEV)   # SDPA wants (section 01)
attn_mask = torch.zeros(B, 1, 1, S_max, dtype=torch.bool, device=DEV)
for b, L in enumerate(lens):
    k, v = gather_kv(kv_data, pages_per_seq[b], L)              # copy the same cache in
    k_pad[b, :, :L] = k.transpose(0, 1).to(FP16)
    v_pad[b, :, :L] = v.transpose(0, 1).to(FP16)
    attn_mask[b, :, :, :L] = True

qh = q.unsqueeze(2)                                             # [B, HQ, 1, D]
try:
    out_sdpa = F.scaled_dot_product_attention(qh, k_pad, v_pad, attn_mask=attn_mask, enable_gqa=True)
except TypeError:  # torch < 2.5: no enable_gqa — materialize the expanded cache (that's the tax)
    kx = k_pad.repeat_interleave(GROUP, dim=1)
    vx = v_pad.repeat_interleave(GROUP, dim=1)
    out_sdpa = F.scaled_dot_product_attention(qh, kx, vx, attn_mask=attn_mask)
out_sdpa = out_sdpa.squeeze(2)
err_sdpa = (out_sdpa.float() - out_ref).abs().max().item()
print(f"  max |SDPA(padded) − reference| = {err_sdpa:.2e}  — padded slots hidden by the mask")
assert err_sdpa < 4e-2
padded_slots = B * S_max
print(f"  SDPA consumed {padded_slots} KV slots for {sum(lens)} real tokens "
      f"({100 * (padded_slots - sum(lens)) // padded_slots}% of them padding)")
print("  ✓ PASS — both paths are correct; the benchmark below differs only in bytes moved")


# ================================ PART 5 ================================
print()
print("=" * 72)
print("PART 5 · the chapter's benchmark: batch decode, SDPA vs FlashInfer")
print("=" * 72)
print("  H100, seq_len 2048, GQA 32/8 (chapter numbers in brackets below)")


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


S = 2048
results = []
for B_bench in [1, 8, 32]:
    lens_b = [S] * B_bench
    pages_b, kv_data_b, kv_indptr_b, kv_indices_b, lpl_b = build_batch(
        lens_b, B_bench * (S // PAGE))
    wb = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace)
    plan_compat(
        wb,
        indptr=kv_indptr_b,
        indices=kv_indices_b,
        last_page_len=lpl_b,
        num_qo_heads=HQ,
        num_kv_heads=HKV,
        head_dim=D,
        page_size=PAGE,
        q_data_type=FP16,
    )
    q_b = torch.randn(B_bench, HQ, D, dtype=FP16, device=DEV)
    t_fi = bench(lambda: wb.run(q_b, kv_data_b))

    # padded dense path for SDPA (this is the memory the rectangle costs)
    k_pad_b = torch.randn(B_bench, HKV, S, D, dtype=FP16, device=DEV)
    v_pad_b = torch.randn(B_bench, HKV, S, D, dtype=FP16, device=DEV)
    qh_b = q_b.unsqueeze(2)                                     # [B, HQ, 1, D]
    try:
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(qh_b, k_pad_b, v_pad_b, enable_gqa=True))
    except TypeError:
        kx = k_pad_b.repeat_interleave(GROUP, dim=1)
        vx = v_pad_b.repeat_interleave(GROUP, dim=1)
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(qh_b, kx, vx))

    results.append((B_bench, t_sdpa, t_fi, t_sdpa / t_fi))
    del k_pad_b, v_pad_b

chapter_sdpa = {1: 0.02, 8: 0.11, 32: 0.41}
chapter_fi = {1: 0.02, 8: 0.03, 32: 0.10}
print(f"\n  {'B':>4} {'SDPA (ms)':>11} {'FlashInfer (ms)':>17} {'speedup':>9}   chapter: sdpa / fi")
for B_bench, t_sdpa, t_fi, sp in results:
    print(f"  {B_bench:>4} {t_sdpa:>11.3f} {t_fi:>17.3f} {sp:>8.1f}x   "
          f"{chapter_sdpa[B_bench]:.2f} / {chapter_fi[B_bench]:.2f}")
print("\n  Your absolute numbers differ by GPU — the *shape* of the gap is the point:")
print("  FlashInfer stays near-flat as B grows; SDPA's cost scales with the padded batch.")

# plan once per step, reuse across layers (section 06)
q32 = torch.randn(32, HQ, D, dtype=FP16, device=DEV)
_, kv_data_32, kv_indptr_32, kv_indices_32, lpl_32 = build_batch([S] * 32, 32 * (S // PAGE))
wb32 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace)
plan_kwargs = dict(indptr=kv_indptr_32, indices=kv_indices_32, last_page_len=lpl_32,
                   num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, page_size=PAGE,
                   q_data_type=FP16)
plan_compat(wb32, **plan_kwargs)
t_run = bench(lambda: wb32.run(q32, kv_data_32))
t_plan = bench(lambda: plan_compat(wb32, **plan_kwargs), warmup=3, iters=20)

LAYERS = 32
good = t_plan + LAYERS * t_run
naive = LAYERS * (t_plan + t_run)
print(f"\n  plan() = {t_plan * 1e3:,.0f} µs (CPU) · run() = {t_run * 1e3:,.0f} µs (GPU) per layer")
print(f"  one {LAYERS}-layer step, plan once : {good * 1e3:,.0f} µs")
print(f"  one {LAYERS}-layer step, plan per layer: {naive * 1e3:,.0f} µs"
      f"  ({naive / good:.1f}× more overhead)")

print()
print("=" * 72)
print("ALL CHECKS PASSED — ragged batches, paged KV, plan/run, and the")
print("batch-decode gap, verified end to end on your GPU.")
print("=" * 72)
