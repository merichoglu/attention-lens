"""
Attention mechanism implementations for transformer models.

This module provides the core attention mechanisms including scaled dot-product
attention and multi-head attention used in transformer architectures.
"""

import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.

    Implements the attention mechanism as described in "Attention Is All You Need".
    Computes attention weights and applies them to values.

    Args:
        dropout (float): Dropout probability for attention weights. Default: 0.1
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Apply scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, heads, seq_len, d_k)
            k (torch.Tensor): Key tensor of shape (batch, heads, seq_len, d_k)
            v (torch.Tensor): Value tensor of shape (batch, heads, seq_len, d_k)
            mask (torch.Tensor, optional): Attention mask for padding or causality

        Returns:
            tuple: (output, attention_weights)
                - output: Attended values of shape (batch, heads, seq_len, d_k)
                - attention_weights: Attention weights of shape (batch, heads, seq_len, seq_len)
        """
        # q, k, v: (batch, heads, seq_len, d_k)
        d_k = q.size(-1)

        # compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask if provided (for padding or causal)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # apply attention to values
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Applies multiple attention heads in parallel and combines their outputs.
    Each head learns different types of relationships between positions.

    Args:
        d_model (int): Model dimension (must be divisible by n_heads)
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability. Default: 0.1

    Raises:
        AssertionError: If d_model is not divisible by n_heads
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Apply multi-head attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, seq_len, d_model)
            k (torch.Tensor): Key tensor of shape (batch, seq_len, d_model)
            v (torch.Tensor): Value tensor of shape (batch, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask

        Returns:
            tuple: (output, attention_weights)
                - output: Multi-head attention output of shape (batch, seq_len, d_model)
                - attention_weights: Attention weights from all heads
        """
        batch_size = q.size(0)

        # linear projections and split into heads
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # apply attention
        output, attn_weights = self.attention(q, k, v, mask)

        # concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # final linear projection
        output = self.w_o(output)

        return output, attn_weights


# quick test
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, n_heads)
    output, attn = mha(x, x, x)

    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")
    print(f"attention weights shape: {attn.shape}")
