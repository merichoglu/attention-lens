# Attention Lens =

A PyTorch implementation for training transformer models on arithmetic tasks and visualizing their attention patterns. This project helps understand how transformers learn to solve mathematical problems by providing comprehensive attention visualization tools.

## Overview

Attention Lens trains transformer models to solve arithmetic problems (addition, subtraction) and provides detailed visualizations of attention patterns. The project demonstrates how transformers learn to focus on relevant parts of input sequences when performing calculations.

### Key Features

- **Transformer Architecture**: Complete implementation with multi-head attention, positional encoding, and feed-forward networks
- **Arithmetic Dataset Generation**: Configurable dataset creation for training on various arithmetic operations
- **Attention Visualization**: Comprehensive tools to visualize attention patterns across layers and heads
- **Training Pipeline**: Robust training with scheduled sampling, logging, and model checkpointing
- **Interactive Analysis**: Tools to test trained models and analyze their problem-solving strategies

## Requirements

- Python 3.8+
- PyTorch 1.9+
- matplotlib
- seaborn
- numpy

## Quick Start

### 1. Install Dependencies

```bash
pip install torch matplotlib seaborn numpy
```

### 2. Generate Training Data

```bash
python src/generate_data.py
```

This creates training, validation, and test datasets in the `data/` directory with configurable parameters:

- 20,000 training samples
- 2,000 validation samples
- 1,000 test samples
- Numbers range from 1-99
- Addition operations (configurable)

### 3. Train the Model

```bash
python src/train.py
```

The training script will:

- Train a transformer model with scheduled sampling
- Log progress to both console and file (`outputs/logs/`)
- Save the best model checkpoint (`outputs/checkpoints/`)
- Display training metrics and validation accuracy

### 4. Visualize Attention Patterns

```bash
python src/visualize.py
```

This generates attention visualizations showing:

- Individual attention heads per layer
- Attention flow across all layers
- Model predictions on test problems
- Heatmaps saved to `outputs/visualizations/`

## Project Structure

```bash
attention-lens/
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── attention.py              # Attention mechanism implementations
│   ├── transformer.py            # Transformer model architecture
│   ├── utils.py                  # Utility functions and data processing
│   ├── generate_data.py          # Dataset generation script
│   ├── train.py                  # Training pipeline
│   └── visualize.py              # Attention visualization tools
├── data/                         # Generated datasets (created automatically)
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── outputs/                      # Training outputs (created automatically)
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training logs
│   └── visualizations/           # Attention plots
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Architecture Details

### Transformer Model

The model implements a standard transformer encoder architecture:

- **Multi-Head Attention**: Parallel attention heads learning different relationship patterns
- **Positional Encoding**: Sinusoidal encoding to provide position information
- **Feed-Forward Networks**: Two-layer MLPs with GELU activation
- **Layer Normalization**: Applied after attention and feed-forward layers
- **Residual Connections**: Skip connections for stable training

### Training Features

- **Scheduled Sampling**: Gradually transitions from teacher forcing to model predictions
- **Gradient Clipping**: Prevents exploding gradients during training
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rates
- **Comprehensive Logging**: Detailed logs saved to timestamped files
- **Model Checkpointing**: Automatic saving of best models based on validation loss

## =' Configuration

### Dataset Generation

Modify parameters in `src/generate_data.py`:

```python
generate_and_save_dataset(
    train_size=20000,      # Number of training samples
    val_size=2000,         # Number of validation samples
    test_size=1000,        # Number of test samples
    max_num=99,            # Maximum number in problems
    operations=["add"]     # Operations: ["add", "subtract"]
)
```

### Model Architecture

Adjust model parameters in `src/train.py`:

```python
train_model(
    d_model=128,          # Model dimension
    n_heads=4,            # Number of attention heads
    n_layers=3,           # Number of transformer layers
    num_epochs=50,        # Training epochs
    batch_size=64,        # Batch size
    lr=0.001             # Learning rate
)
```

## Understanding the Visualizations

### Attention Head Plots

- **Heatmaps**: Show attention weights between input positions
- **Rows**: Query positions (where attention is coming from)
- **Columns**: Key positions (what is being attended to)
- **Colors**: Attention intensity (darker = higher attention)

### Layer Analysis

- **Early Layers**: Often focus on local patterns and token relationships
- **Middle Layers**: Learn operation-specific patterns (recognizing +, =)
- **Late Layers**: Integrate information for final answer computation

### Attention Flow

- Shows how information flows through the model
- Reveals which parts of the input the model considers important
- Helps identify if the model learns meaningful arithmetic strategies

## Example Output

After training, you might see attention patterns like:

1. **Operator Attention**: Model focuses on the '+' symbol to understand the operation
2. **Operand Collection**: Attention flows from the '=' to both numbers being added
3. **Result Generation**: Sequential attention pattern when generating the answer digits

## Logging

All training runs create detailed logs in `outputs/logs/` with:

- Training progress and metrics
- Model architecture details
- Hyperparameter settings
- Validation results
- Error tracking

Console output provides clean, readable progress updates while files contain comprehensive details for analysis.

## Performance Tips

- Use GPU if available (CUDA support included)
- Adjust batch size based on available memory
- Monitor attention patterns to verify learning
- Experiment with different learning rates
- Use tensorboard integration for advanced monitoring (future enhancement)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) - Implementation guide
