'''
FeedForward is a simple neural network that is used in the Transformer architecture.
It is used to process the output of the attention layer and produce the output of the Transformer.
'''
import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim)
    
    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))
