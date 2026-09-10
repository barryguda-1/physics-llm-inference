import torch.nn as nn
from attention import causal_attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    # Foward pass reshapes Q, K, V from (batch, seq, hidden) to (batch, heads, seq, head_dim)
    # Runs attention, then reshapes back to (batch, seq, hidden) and projects to output
    # Args:
    #     x: Input tensor of shape (batch, seq, hidden)
    # Returns:
    #     Output tensor of shape (batch, seq, hidden)
    # The transpose moves the head dimension before the sequence dimension: (batch, seq, heads,
    # head_dim) becomes (batch,heads, seq,head_dim). This is because we want to apply attention
    # to each head separately.
    
    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = causal_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, -1))