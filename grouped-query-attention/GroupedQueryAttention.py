'''
The key change is repeat_interleave. We project K and V to fewer heads, then expand them to
match the query headcount.
This is where grouped-query attention (GQA) helps. Instead of giving each query head its own KV
head, we share KV heads across groups of query heads. Qwen3-72B uses 8 KV heads shared across
64 query heads. Each KV head serves 8 query heads.
'''
import torch.nn as nn
from attention import causal_attention
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_q_heads, num_kv_heads):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # K and V project to fewer heads than Q - this is the whole point of GQA:
        # num_kv_heads * head_dim instead of hidden_dim, which is what shrinks the KV cache
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)
        # causal_attention masks and softmaxes over the last two dims, so heads must come
        # before the sequence: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        out = causal_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, -1))