# Causal attention is a type of attention mechanism that is used in causal language models
# For autoregressuve generation, we need to prevent the model from attending to future tokens
# It is a type of self-attention that is used to compute the attention scores for each token in the sequence
# The attention scores are computed by a softmax function applied to the dot product of the query and key vectors
# Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
# Q, K, V are matrices of shape (batch_size, seq_len, d_k)
# d_k is the dimension of the key vectors

import torch
import math

def causal_attention(Q, K, V):
    """
    Compute causal attention scores

    Args:
        Q: Query matrix of shape (batch_size, seq_len, d_k)
        K: Key matrix of shape (batch_size, seq_len, d_k)
        V: Value matrix of shape (batch_size, seq_len, d_v)

    Returns:
        Attention scores of shape (batch_size, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Mask out future tokens
    mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1]), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
