# Attention is a weighted sum of values
# The weights are computed by a softmax function applied to the dot product of the query and key vectors
# Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
# Q, K, V are matrices of shape (batch_size, seq_len, d_k)
# d_k is the dimension of the key vectors

import torch
import math

def attention(Q, K, V):
    """
    Compute attention scores
    
    Args:
        Q: Query matrix of shape (batch_size, seq_len, d_k)
        K: Key matrix of shape (batch_size, seq_len, d_k)
        V: Value matrix of shape (batch_size, seq_len, d_v)
    
    Returns:
        Attention scores of shape (batch_size, seq_len, d_v)
    """
    d_k = Q.shape[-1] # get last index of Q.size()
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

