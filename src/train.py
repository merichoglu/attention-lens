"""
Training module for transformer models on arithmetic tasks.

This module contains the dataset class, training functions, and main training
loop for transformer models learning arithmetic sequence prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer import TransformerForSequencePrediction
from utils import ArithmeticVocab
import os
import random
import logging
from datetime import datetime


def setup_logging(log_file=None):
    """
    Setup logging to both file and console.

    Args:
        log_file (str, optional): Path to log file. If None, creates timestamped file.

    Returns:
        logging.Logger: Configured logger instance
    """
    if log_file is None:
        os.makedirs("outputs/logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"outputs/logs/training_{timestamp}.log"

    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # File handler with detailed format
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)

    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class ArithmeticDataset(Dataset):
    """
    Dataset class for arithmetic sequence prediction.

    Loads arithmetic problems from text files and prepares them for
    autoregressive training with proper input/target sequences.

    Args:
        filepath (str): Path to the dataset file
        vocab (ArithmeticVocab): Vocabulary for encoding tokens
        max_len (int): Maximum sequence length. Default: 20
    """
    def __init__(self, filepath, vocab, max_len=20):
        self.vocab = vocab
        self.max_len = max_len

        # load samples from file
        with open(filepath, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]

        logger = logging.getLogger('training')
        logger.info(f"loaded {len(self.samples)} samples from {filepath}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single training example.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (input_seq, target_seq)
                - input_seq: Input tokens for autoregressive prediction
                - target_seq: Target tokens (input shifted by one position)
        """
        sample = self.samples[idx]

        # encode
        tokens = self.vocab.encode(sample)

        # pad to max_len
        if len(tokens) < self.max_len:
            tokens = tokens + [self.vocab.token2idx[self.vocab.pad_token]] * (
                self.max_len - len(tokens)
            )
        else:
            tokens = tokens[: self.max_len]

        tokens = torch.tensor(tokens, dtype=torch.long)

        # input is all tokens except last, target is all tokens except first
        input_seq = tokens[:-1]
        target_seq = tokens[1:]

        return input_seq, target_seq


def train_epoch_teacher_forcing(model, dataloader, criterion, optimizer, device, vocab):
    """
    Train one epoch using standard teacher forcing.

    Args:
        model (TransformerForSequencePrediction): Model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to train on
        vocab (ArithmeticVocab): Vocabulary for token handling

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    logger = logging.getLogger('training')
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # forward pass with teacher forcing
        output = model(inputs)

        # compute loss (ignore padding)
        loss = criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))

        # compute accuracy
        predictions = output.argmax(dim=-1)
        mask = targets != vocab.token2idx[vocab.pad_token]
        correct = ((predictions == targets) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            acc = 100.0 * correct / mask.sum().item() if mask.sum().item() > 0 else 0
            logger.info(f"  batch {batch_idx + 1}/{len(dataloader)}, loss: {loss.item():.4f}, acc: {acc:.2f}%")

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, avg_acc


def train_epoch_scheduled_sampling(model, dataloader, criterion, optimizer, device, vocab, epoch, total_epochs):
    """
    Train one epoch using scheduled sampling.

    Gradually transitions from teacher forcing to using model predictions,
    helping the model learn to handle its own errors during inference.

    Args:
        model (TransformerForSequencePrediction): Model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to train on
        vocab (ArithmeticVocab): Vocabulary for token handling
        epoch (int): Current epoch number
        total_epochs (int): Total number of training epochs

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    logger = logging.getLogger('training')
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    # scheduled sampling ratio: start with teacher forcing, gradually use model predictions
    teacher_forcing_ratio = max(0.0, 1.0 - (epoch / total_epochs) * 0.5)

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        batch_size, seq_len = src.shape

        optimizer.zero_grad()

        # build input sequences with scheduled sampling
        inputs = []
        inputs.append(src[:, :1])  # start with first token
        
        for t in range(1, seq_len):
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # use ground truth
                inputs.append(src[:, :t+1])
            else:
                # use model prediction
                with torch.no_grad():
                    prev_input = torch.cat([inp[:, -1:] for inp in inputs], dim=1)
                    temp_output = model(prev_input)
                    predicted_token = temp_output[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # create new input with predicted token
                new_input = torch.cat([inputs[-1], predicted_token], dim=1)
                inputs.append(new_input)
        
        # final forward pass with constructed input
        model_input = inputs[-1]
        output = model(model_input)

        # compute loss (ignore padding)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))

        # compute accuracy
        predictions = output.argmax(dim=-1)
        mask = tgt != vocab.token2idx[vocab.pad_token]
        correct = ((predictions == tgt) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            acc = 100.0 * correct / mask.sum().item() if mask.sum().item() > 0 else 0
            logger.info(
                f"  batch {batch_idx + 1}/{len(dataloader)}, loss: {loss.item():.4f}, acc: {acc:.2f}%, tf_ratio: {teacher_forcing_ratio:.2f}"
            )

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device, vocab):
    """
    Evaluate model on validation/test data.

    Args:
        model (TransformerForSequencePrediction): Model to evaluate
        dataloader (DataLoader): Evaluation data loader
        criterion (nn.Module): Loss function
        device (str): Device to run evaluation on
        vocab (ArithmeticVocab): Vocabulary for token handling

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))

            predictions = output.argmax(dim=-1)
            mask = tgt != vocab.token2idx[vocab.pad_token]
            correct = ((predictions == tgt) & mask).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, avg_acc


def train_model(
    d_model=128, n_heads=4, n_layers=3, num_epochs=50, batch_size=64, lr=0.001
):
    """
    Complete training pipeline for transformer model on arithmetic tasks.

    Args:
        d_model (int): Model dimension. Default: 128
        n_heads (int): Number of attention heads. Default: 4
        n_layers (int): Number of transformer layers. Default: 3
        num_epochs (int): Number of training epochs. Default: 50
        batch_size (int): Training batch size. Default: 64
        lr (float): Learning rate. Default: 0.001

    Returns:
        tuple: (trained_model, vocabulary)
    """
    # Setup logging
    logger = setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    # create vocab
    vocab = ArithmeticVocab()
    logger.info(f"vocab size: {len(vocab)}")

    # load datasets from files
    train_dataset = ArithmeticDataset("data/train.txt", vocab)
    val_dataset = ArithmeticDataset("data/val.txt", vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # create model
    model = TransformerForSequencePrediction(
        vocab_size=len(vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        pad_idx=vocab.token2idx[vocab.pad_token],
    ).to(device)

    logger.info(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx[vocab.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        logger.info(f"\nepoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch_scheduled_sampling(
            model, train_loader, criterion, optimizer, device, vocab, epoch, num_epochs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, vocab)

        scheduler.step(val_loss)

        logger.info(f"train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
        logger.info(f"val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("outputs/checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "vocab": vocab,
                    "config": {
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "d_ff": d_model * 4,
                    },
                },
                "outputs/checkpoints/best_model.pt",
            )
            logger.info(
                f"saved best model with val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%"
            )

    return model, vocab


if __name__ == "__main__":
    model, vocab = train_model(
        d_model=128, n_heads=4, n_layers=3, num_epochs=50, batch_size=64, lr=0.001
    )