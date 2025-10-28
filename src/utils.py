"""
Utility functions and classes for transformer models and data processing.

This module provides essential utilities including positional encoding,
mask creation functions, vocabulary handling, and dataset generation for
arithmetic tasks.
"""

import torch
import torch.nn as nn
import math
import random


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Adds sinusoidal positional encodings to input embeddings to provide
    position information since transformers lack inherent position awareness.

    Args:
        d_model (int): Model dimension
        max_len (int): Maximum sequence length. Default: 5000
        dropout (float): Dropout probability. Default: 0.1
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # sin for even indices, cos for odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch, seq_len, d_model)

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask for attention mechanism.

    Args:
        seq (torch.Tensor): Input sequence of shape (batch, seq_len)
        pad_idx (int): Padding token index. Default: 0

    Returns:
        torch.Tensor: Boolean mask of shape (batch, 1, 1, seq_len)
                     True for non-padding tokens, False for padding
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len, device="cpu"):
    """
    Create causal (lower triangular) mask for autoregressive attention.

    Prevents attention to future positions in the sequence.

    Args:
        seq_len (int): Sequence length
        device (str): Device to place the mask on. Default: "cpu"

    Returns:
        torch.Tensor: Boolean mask of shape (1, 1, seq_len, seq_len)
                     True for allowed positions, False for masked positions
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    return mask.unsqueeze(0).unsqueeze(0)


class ArithmeticVocab:
    """
    Vocabulary for arithmetic tasks.

    Handles encoding and decoding of arithmetic expressions containing
    digits, operators, and special tokens for sequence-to-sequence tasks.

    Tokens include:
        - Digits: 0-9
        - Operators: +, -, =
        - Special tokens: <PAD>, <EOS>
    """

    def __init__(self):
        # special tokens
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"

        # arithmetic tokens
        self.tokens = [self.pad_token, self.eos_token, "+", "-", "="] + [
            str(i) for i in range(10)
        ]
        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def encode(self, text):
        """
        Encode text string to list of token indices.

        Args:
            text (str): Input text containing arithmetic expression

        Returns:
            list: List of token indices
        """
        # handle special tokens by replacement and tokenization
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+5] == "<EOS>":
                tokens.append(self.token2idx[self.eos_token])
                i += 5
            elif text[i] in self.token2idx:
                tokens.append(self.token2idx[text[i]])
                i += 1
            else:
                # skip unknown characters
                i += 1
        return tokens

    def decode(self, indices):
        """
        Decode list of token indices to text string.

        Args:
            indices (list): List of token indices

        Returns:
            str: Decoded text with padding tokens removed
        """
        # convert list of indices to string
        return "".join(
            [
                self.idx2token[idx]
                for idx in indices
                if idx != self.token2idx[self.pad_token]
            ]
        )

    def __len__(self):
        return len(self.tokens)


def generate_addition_sample(max_num=99):
    """
    Generate single addition problem.

    Args:
        max_num (int): Maximum number to use in the problem. Default: 99

    Returns:
        str: Formatted addition problem like "12+34=46<EOS>"
    """
    a = random.randint(1, max_num)
    b = random.randint(1, max_num)
    result = a + b

    # format: "12+34=68<EOS>"
    problem = f"{a}+{b}={result}<EOS>"
    return problem


def generate_subtraction_sample(max_num=99):
    """
    Generate single subtraction problem.

    Args:
        max_num (int): Maximum number to use in the problem. Default: 99

    Returns:
        str: Formatted subtraction problem like "45-23=22<EOS>"
    """
    a = random.randint(1, max_num)
    b = random.randint(1, max(1, a))  # ensure positive result
    result = a - b

    # format: "45-23=22<EOS>"
    problem = f"{a}-{b}={result}<EOS>"
    return problem


def generate_arithmetic_dataset(num_samples=10000, max_num=99, operations=["add"]):
    """
    Generate arithmetic dataset with specified operations.

    Args:
        num_samples (int): Number of samples to generate. Default: 10000
        max_num (int): Maximum number to use in problems. Default: 99
        operations (list): List of operations ('add', 'subtract'). Default: ['add']

    Returns:
        list: List of formatted arithmetic problems as strings
    """
    samples = []

    for _ in range(num_samples):
        op = random.choice(operations)

        if op == "add":
            sample = generate_addition_sample(max_num)
        elif op == "subtract":
            sample = generate_subtraction_sample(max_num)
        else:
            sample = generate_addition_sample(max_num)

        samples.append(sample)

    return samples


def save_arithmetic_dataset(filename="data/dataset.txt", num_samples=10000):
    """
    Save arithmetic dataset to file.

    Args:
        filename (str): Output filename. Default: "data/dataset.txt"
        num_samples (int): Number of samples to generate. Default: 10000

    Returns:
        None: File is saved to disk
    """
    samples = generate_arithmetic_dataset(num_samples)

    with open(filename, "w") as f:
        for sample in samples:
            f.write(sample + "\n")

    print(f"saved {num_samples} samples to {filename}")


# quick test
if __name__ == "__main__":
    # test vocab
    vocab = ArithmeticVocab()
    print(f"vocab size: {len(vocab)}")
    print(f"tokens: {vocab.tokens}")

    # test encoding/decoding
    text = "12+34=46<EOS>"
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded)
    print(f"\noriginal: {text}")
    print(f"encoded: {encoded}")
    print(f"decoded: {decoded}")

    # generate samples
    print("\nsample problems:")
    for i in range(5):
        print(f"  {generate_addition_sample()}")

    # save dataset
    save_arithmetic_dataset(num_samples=10000)