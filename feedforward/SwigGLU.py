'''
SwiGLU is a gating mechanism that is used in the Transformer architecture.
It is used to process the output of the attention layer and produce the output of the Transformer.
'''

import torch.nn as nn
import torch.nn.functional as F
class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))