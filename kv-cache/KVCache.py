# The KV cache is the central data structure of LLM inference. It stores the key and value
# projections for all previous tokens so we don't recompute them during decode.
# Each transformer layer produces K and V tensors. For a token at position i, these projections
# only depend on the input at position i - not on any other position. Once computed, they never
# change. The attention mechanism needs K and V for all positions 0 through i-1 to compute output
# at position i. Without a cache, we'd recompute all of them. With a cache, we just look them up.
# The cache pre-allocates memory for the maximum sequence length, then fills it one position at
# a time as generation proceeds.

import torch

class KVCache:
    def __init__(self, batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float32, device="cpu"):
        """
        Pre-allocate the full buffer once. We never reallocate during generation,
        we only advance seq_len, so the memory footprint is fixed at creation time.
        Layout is (batch, seq, num_heads, head_dim) so appending a new token is a
        slice write along the sequence dimension.
        """
        self.k_cache = torch.zeros(batch_size, max_seq_len, num_heads, head_dim, dtype=dtype, device=device)
        self.v_cache = torch.zeros(batch_size, max_seq_len, num_heads, head_dim, dtype=dtype, device=device)
        self.seq_len = 0  # how many positions are actually filled; the rest is zero padding

    def update(self, k_new, v_new):
        """
        Write the projections of the newly processed token(s) into the next free slots.

        Args:
            k_new: Key tensor of shape (batch, new_tokens, num_heads, head_dim)
            v_new: Value tensor of shape (batch, new_tokens, num_heads, head_dim)
        """
        new_len = k_new.shape[1]
        self.k_cache[:, self.seq_len:self.seq_len + new_len] = k_new
        self.v_cache[:, self.seq_len:self.seq_len + new_len] = v_new
        self.seq_len += new_len

    def get(self):
        """
        Return only the filled region. Attention must never see the zero padding
        beyond seq_len - it would add spurious positions to the softmax.
        """
        return self.k_cache[:, :self.seq_len], self.v_cache[:, :self.seq_len]
