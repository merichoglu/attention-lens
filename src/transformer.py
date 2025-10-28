"""
Transformer model implementation for sequence prediction tasks.

This module contains the complete transformer architecture including
encoder layers, feed-forward networks, and the main transformer model
for arithmetic sequence prediction.
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from utils import PositionalEncoding, create_padding_mask, create_causal_mask


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two-layer MLP with GELU activation used in transformer layers.

    Args:
        d_model (int): Model dimension
        d_ff (int): Hidden dimension of feed-forward network
        dropout (float): Dropout probability. Default: 0.1
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Apply feed-forward transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Consists of multi-head self-attention followed by position-wise
    feed-forward network, with residual connections and layer normalization.

    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Hidden dimension of feed-forward network
        dropout (float): Dropout probability. Default: 0.1
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply encoder layer transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask

        Returns:
            tuple: (output, attention_weights)
                - output: Layer output of shape (batch, seq_len, d_model)
                - attention_weights: Attention weights from self-attention
        """
        # self attention with residual
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # feed forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Complete transformer encoder stack.

    Combines token embeddings with positional encoding and passes through
    multiple encoder layers. Stores attention weights for visualization.

    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Model dimension. Default: 128
        n_heads (int): Number of attention heads. Default: 4
        n_layers (int): Number of encoder layers. Default: 3
        d_ff (int): Hidden dimension of feed-forward network. Default: 512
        max_len (int): Maximum sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
        pad_idx (int): Padding token index. Default: 0
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_len=512,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # for storing attention weights during forward pass
        self.attention_weights = []

    def forward(self, x, mask=None):
        """
        Apply transformer encoder to input sequence.

        Args:
            x (torch.Tensor): Input token indices of shape (batch, seq_len)
            mask (torch.Tensor, optional): Attention mask

        Returns:
            torch.Tensor: Encoded representations of shape (batch, seq_len, d_model)
        """
        self.attention_weights = []

        # embedding and positional encoding
        x = self.embedding(x) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )
        x = self.pos_encoding(x)

        # pass through encoder layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            self.attention_weights.append(attn_weights)

        return x

    def get_attention_weights(self):
        """
        Get attention weights from all layers.

        Returns:
            list: List of attention weight tensors, one per layer
                  Each element has shape (batch, heads, seq_len, seq_len)
        """
        return self.attention_weights


class TransformerForSequencePrediction(nn.Module):
    """
    Transformer model for sequence prediction tasks.

    Combines transformer encoder with output projection layer for
    next-token prediction in arithmetic sequences.

    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Model dimension. Default: 128
        n_heads (int): Number of attention heads. Default: 4
        n_layers (int): Number of encoder layers. Default: 3
        d_ff (int): Hidden dimension of feed-forward network. Default: 512
        max_len (int): Maximum sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
        pad_idx (int): Padding token index. Default: 0
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_len=512,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout, pad_idx
        )

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass for sequence prediction.

        Args:
            x (torch.Tensor): Input token indices of shape (batch, seq_len)
            mask (torch.Tensor, optional): Attention mask

        Returns:
            torch.Tensor: Logits for next token prediction of shape (batch, seq_len, vocab_size)
        """
        if mask is None:
            # (batch, 1, 1, seq_len)
            pad_mask = create_padding_mask(x, self.encoder.pad_idx)
            # (1, 1, seq_len, seq_len)
            causal_mask = create_causal_mask(x.size(1), device=x.device)
            # Combine masks: (batch, 1, seq_len, seq_len)
            mask = pad_mask & causal_mask

        encoded = self.encoder(x, mask)
        logits = self.output_projection(encoded)
        return logits

    def get_attention_weights(self):
        """
        Get attention weights from all encoder layers.

        Returns:
            list: List of attention weight tensors from encoder layers
        """
        return self.encoder.get_attention_weights()


# quick test
if __name__ == "__main__":
    vocab_size = 100
    batch_size = 2
    seq_len = 10

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = TransformerForSequencePrediction(
        vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256
    )

    output = model(x)
    attn_weights = model.get_attention_weights()

    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")
    print(f"number of layers: {len(attn_weights)}")
    print(f"attention weights shape per layer: {attn_weights[0].shape}")