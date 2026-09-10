# Cached attention: during decode we only run the model on the newest token, so we only
# compute its Q, K, V. The K and V of every previous token are already in the cache.
# Two things change compared to full attention:
# 1. The query is a single row (one new token), so scores have shape (batch, heads, 1, S).
# 2. No causal mask is needed. Position i can never see positions > i because those
#    positions are not in the cache yet - the mask is implicit in what we've stored.
# During prefill (the first forward pass over the whole prompt) the cache is empty and we
# still need the mask, same as standard training attention.
#
# The tradeoff between computation and memory:
# Without the cache, generating token i means re-projection and re-attention over all i
# previous tokens - O(i^2) work overall for the sequence. With the cache, each step does
# O(i) work: one projection plus attention against i cached entries.
# What we pay is memory. Per token, per layer, the cache holds K and V:
#   2 * kv_heads * head_dim * dtype_bytes
# For a Llama-2-7B style model (32 layers, 32 heads, head_dim 128, fp16) that is
# 2 * 32 * 32 * 128 * 2 bytes = 512 KB per token, so a 4096-token sequence holds ~2 GB.
# Grouped-query attention (see the grouped-query-attention section) shrinks this by the
# KV sharing factor: Qwen3-72B has 8 KV heads instead of 64, cutting the cache 8x.

import torch
import math
from KVCache import KVCache


def split_heads(t, num_heads):
    """(batch, seq, hidden) -> (batch, seq, num_heads, head_dim)"""
    B, S, _ = t.shape
    return t.view(B, S, num_heads, -1)


def cached_attention(q, cache, use_causal_mask=False):
    """
    Attend queries against the keys and values stored in the cache.

    Args:
        q: Queries for the new token(s) of shape (batch, q_len, num_heads, head_dim)
        cache: KVCache holding all previous K and V
        use_causal_mask: True for prefill (q covers many positions), False for decode
                         (a single query row, nothing ahead of it exists in the cache)

    Returns:
        Attention output of shape (batch, q_len, num_heads, head_dim)
    """
    k, v = cache.get()  # (batch, S, num_heads, head_dim), only the filled region
    q_h = q.transpose(1, 2)  # (batch, heads, q_len, head_dim)
    k_h = k.transpose(1, 2)  # (batch, heads, S, head_dim)
    v_h = v.transpose(1, 2)

    d_k = q.shape[-1]
    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, heads, q_len, S)

    if use_causal_mask:
        # Only during prefill: q_len == S, so hide future positions as usual
        mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1]), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_h)  # (batch, heads, q_len, head_dim)
    return out.transpose(1, 2)  # back to (batch, q_len, heads, head_dim)


class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, cache):
        B, S, _ = x.shape
        q = split_heads(self.q_proj(x), self.num_heads)
        k = split_heads(self.k_proj(x), self.num_heads)
        v = split_heads(self.v_proj(x), self.num_heads)
        # Prefill covers the whole prompt (many positions, need the mask);
        # decode covers one new position (no mask - the future isn't in the cache yet).
        use_causal_mask = cache.seq_len == 0
        cache.update(k, v)
        out = cached_attention(q, cache, use_causal_mask=use_causal_mask)
        B, S, H, D = out.shape
        return self.o_proj(out.reshape(B, S, H * D))


if __name__ == "__main__":
    # Sanity check: run the same fixed token embeddings through both paths and compare
    # each position's output. (Feeding outputs back as inputs would amplify ordinary
    # float noise through these unnormalized layers and swamp the comparison.)
    torch.manual_seed(0)
    B, hidden_dim, num_heads = 1, 64, 4
    prompt_len, steps = 8, 6
    total_len = prompt_len + steps
    layer = AttentionLayer(hidden_dim, num_heads)
    tokens = torch.randn(B, total_len, hidden_dim)

    # Cached path: prefill over the prompt, then one token per decode step
    cache = KVCache(B, total_len, num_heads, layer.head_dim)
    cached_out = layer(tokens[:, :prompt_len], cache)
    for i in range(steps):
        nxt = layer(tokens[:, prompt_len + i:prompt_len + i + 1], cache)
        cached_out = torch.cat([cached_out, nxt], dim=1)

    # Uncached path: recompute attention over the full prefix at every position
    def full_forward(seq):
        B, S, _ = seq.shape
        q = split_heads(layer.q_proj(seq), num_heads)
        k = split_heads(layer.k_proj(seq), num_heads)
        v = split_heads(layer.v_proj(seq), num_heads)
        q_h = q.transpose(1, 2)  # (batch, heads, S, head_dim)
        k_h = k.transpose(1, 2)
        v_h = v.transpose(1, 2)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(layer.head_dim)
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
        weights = torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1)
        out = torch.matmul(weights, v_h)  # (batch, heads, S, head_dim)
        return layer.o_proj(out.transpose(1, 2).reshape(B, S, hidden_dim))

    diffs = [(full_forward(tokens[:, :i + 1])[0, -1] - cached_out[0, i]).abs().max().item()
             for i in range(total_len)]
    max_diff = max(diffs)
    print(f"max |cached - uncached| across all {total_len} positions: {max_diff:.2e}")
    assert max_diff < 1e-5, "cached and uncached paths diverged"
    print("cached generation matches full recompute")
