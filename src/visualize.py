"""
Visualization utilities for transformer attention patterns.

This module provides functions to visualize attention weights from trained
transformer models, helping understand how the model learns to solve
arithmetic problems.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformer import TransformerForSequencePrediction
from utils import ArithmeticVocab
import os
import logging


def plot_attention_head(attention, tokens, layer_idx, head_idx, save_path=None):
    """
    Plot attention weights for a single attention head.

    Args:
        attention (numpy.ndarray): Attention weights of shape (seq_len, seq_len)
        tokens (list): List of token strings for axis labels
        layer_idx (int): Layer index for title
        head_idx (int): Head index for title
        save_path (str, optional): Path to save the plot

    Returns:
        None: Displays or saves the plot
    """
    # attention: (seq_len, seq_len)
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        cbar=True,
        square=True,
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_title(f"layer {layer_idx} - head {head_idx}", fontsize=14)
    ax.set_xlabel("key position", fontsize=12)
    ax.set_ylabel("query position", fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_attention_layer(attention, tokens, layer_idx, save_path=None):
    """
    Plot attention weights for all heads in a single layer.

    Args:
        attention (numpy.ndarray): Attention weights of shape (n_heads, seq_len, seq_len)
        tokens (list): List of token strings for axis labels
        layer_idx (int): Layer index for title
        save_path (str, optional): Path to save the plot

    Returns:
        None: Displays or saves the plot
    """
    # attention: (n_heads, seq_len, seq_len)
    n_heads = attention.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    for head_idx in range(n_heads):
        sns.heatmap(
            attention[head_idx],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            cbar=True,
            square=True,
            ax=axes[head_idx],
            vmin=0,
            vmax=1,
        )
        axes[head_idx].set_title(f"head {head_idx}")
        axes[head_idx].set_xlabel("key")
        axes[head_idx].set_ylabel("query")

    fig.suptitle(f"layer {layer_idx} - all heads", fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_all_layers(attention_weights, tokens, save_dir="outputs/visualizations"):
    """
    Plot attention weights for all layers in the model.

    Args:
        attention_weights (list): List of attention tensors, one per layer
                                 Each tensor has shape (batch, n_heads, seq_len, seq_len)
        tokens (list): List of token strings for axis labels
        save_dir (str): Directory to save the plots. Default: "outputs/visualizations"

    Returns:
        None: Saves plots to specified directory
    """
    # attention_weights: list of (batch, n_heads, seq_len, seq_len)
    n_layers = len(attention_weights)

    for layer_idx, layer_attn in enumerate(attention_weights):
        # take first sample from batch
        layer_attn = layer_attn[0].detach().cpu().numpy()

        save_path = f"{save_dir}/layer_{layer_idx}_all_heads.png"
        plot_attention_layer(layer_attn, tokens, layer_idx, save_path)


def visualize_attention_flow(attention_weights, tokens, save_path=None):
    """
    Visualize attention flow across all layers.

    Creates a single plot showing average attention patterns across all layers,
    helping understand how information flows through the transformer.

    Args:
        attention_weights (list): List of attention tensors, one per layer
                                 Each tensor has shape (batch, n_heads, seq_len, seq_len)
        tokens (list): List of token strings for axis labels
        save_path (str, optional): Path to save the plot

    Returns:
        None: Displays or saves the plot
    """
    # average attention across all heads and layers
    n_layers = len(attention_weights)

    # average across heads for each layer
    avg_attention = []
    for layer_attn in attention_weights:
        # layer_attn: (batch, n_heads, seq_len, seq_len)
        layer_avg = layer_attn[0].mean(dim=0).detach().cpu().numpy()
        avg_attention.append(layer_avg)

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for layer_idx, layer_attn in enumerate(avg_attention):
        sns.heatmap(
            layer_attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            cbar=True,
            square=True,
            ax=axes[layer_idx],
            vmin=0,
            vmax=1,
        )
        axes[layer_idx].set_title(f"layer {layer_idx} (avg heads)", fontsize=12)
        axes[layer_idx].set_xlabel("key", fontsize=10)
        axes[layer_idx].set_ylabel("query", fontsize=10)

    fig.suptitle("attention flow across layers", fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def test_model_examples(model, vocab, device="cpu", num_examples=5):
    """
    Test model on arithmetic problems and visualize attention patterns.

    Generates random arithmetic problems, evaluates the model's performance,
    and creates attention visualizations for the first few examples.

    Args:
        model (TransformerForSequencePrediction): Trained transformer model
        vocab (ArithmeticVocab): Vocabulary for encoding/decoding
        device (str): Device to run inference on. Default: "cpu"
        num_examples (int): Number of test examples to generate. Default: 5

    Returns:
        None: Prints results and saves visualizations
    """
    # Setup basic logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)

    model.eval()

    logger.info("\ntesting arithmetic problems:\n")

    with torch.no_grad():
        for i in range(num_examples):
            # generate random problem
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            correct_result = a + b

            # create input (problem without answer)
            problem = f"{a}+{b}="
            generated = vocab.encode(problem)

            max_gen_len = 10
            for step in range(max_gen_len):
                # pad current sequence to fixed length
                current_len = len(generated)
                padded = generated + [vocab.token2idx[vocab.pad_token]] * (19 - current_len)
                
                input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
                output = model(input_tensor)

                # get prediction at last non-padding position
                next_token_logits = output[0, current_len - 1]
                next_token = next_token_logits.argmax().item()
                
                generated.append(next_token)

                # stop at eos or max length
                if next_token == vocab.token2idx[vocab.eos_token] or len(generated) >= 19:
                    break

            # decode result
            full_text = vocab.decode(generated)

            # extract predicted answer
            if "=" in full_text:
                parts = full_text.split("=")
                if len(parts) > 1:
                    pred_answer = parts[1].replace("<EOS>", "").strip()
                    try:
                        pred_result = int(pred_answer) if pred_answer else "?"
                        is_correct = pred_result == correct_result
                    except:
                        pred_result = pred_answer if pred_answer else "?"
                        is_correct = False
                else:
                    pred_result = "?"
                    is_correct = False
            else:
                pred_result = "?"
                is_correct = False

            status = "✓" if is_correct else "✗"
            logger.info(f"{status} {a} + {b} = {correct_result} | predicted: {pred_result}")

            # visualize attention for first few examples
            if i < 3:
                # get attention for complete sequence
                padded = generated + [vocab.token2idx[vocab.pad_token]] * (
                    19 - len(generated)
                )
                padded = padded[:19]
                input_tensor = torch.tensor([padded], dtype=torch.long).to(device)

                _ = model(input_tensor)
                attention_weights = model.get_attention_weights()

                tokens = [vocab.idx2token.get(idx, "?") for idx in padded]

                # save visualizations
                visualize_attention_flow(
                    attention_weights,
                    tokens,
                    save_path=f"outputs/visualizations/example_{i}_flow.png",
                )

                plot_all_layers(
                    attention_weights, tokens, save_dir="outputs/visualizations"
                )

                logger.info(f"  saved visualization for example {i}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup basic logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)

    # load checkpoint
    checkpoint_path = "outputs/checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        logger.error(f"no checkpoint found at {checkpoint_path}")
        logger.error("train model first using: python src/train.py")
        exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]

    # create model
    model = TransformerForSequencePrediction(
        vocab_size=len(vocab),
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        pad_idx=vocab.token2idx[vocab.pad_token],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"loaded model from {checkpoint_path}")
    logger.info(f"val accuracy: {checkpoint['val_acc']:.2f}%")

    # test and visualize
    test_model_examples(model, vocab, device, num_examples=10)

    logger.info(f"\nvisualizations saved to outputs/visualizations/")