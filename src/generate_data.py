"""
Data generation module for arithmetic datasets.

This module provides functionality to generate and save arithmetic datasets
for training transformer models. It creates train, validation, and test splits
with configurable problem types and difficulty levels.
"""

from utils import ArithmeticVocab, generate_arithmetic_dataset
import os
import logging


def generate_and_save_dataset(
    train_size=20000, val_size=2000, test_size=1000, max_num=99, operations=["add"]
):
    """
    Generate and save arithmetic datasets for training, validation, and testing.

    Creates three dataset files (train.txt, val.txt, test.txt) containing
    arithmetic problems with their solutions in text format.

    Args:
        train_size (int): Number of training samples to generate. Default: 20000
        val_size (int): Number of validation samples to generate. Default: 2000
        test_size (int): Number of test samples to generate. Default: 1000
        max_num (int): Maximum number to use in arithmetic problems. Default: 99
        operations (list): List of operations to include ('add', 'subtract'). Default: ['add']

    Returns:
        None: Files are saved to data/ directory

    Example:
        >>> generate_and_save_dataset(train_size=1000, operations=['add', 'subtract'])
        # Creates data/train.txt, data/val.txt, data/test.txt with mixed operations
    """

    # Setup basic logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)

    logger.info(f"generating datasets...")
    logger.info(f"  train: {train_size} samples")
    logger.info(f"  val: {val_size} samples")
    logger.info(f"  test: {test_size} samples")
    logger.info(f"  operations: {operations}")
    logger.info(f"  max number: {max_num}")

    # create data directory
    os.makedirs("data", exist_ok=True)

    # generate datasets
    train_samples = generate_arithmetic_dataset(train_size, max_num, operations)
    val_samples = generate_arithmetic_dataset(val_size, max_num, operations)
    test_samples = generate_arithmetic_dataset(test_size, max_num, operations)

    # save to files
    with open("data/train.txt", "w") as f:
        for sample in train_samples:
            f.write(sample + "\n")

    with open("data/val.txt", "w") as f:
        for sample in val_samples:
            f.write(sample + "\n")

    with open("data/test.txt", "w") as f:
        for sample in test_samples:
            f.write(sample + "\n")

    logger.info(f"\nsaved datasets to data/")
    logger.info(f"  train.txt: {train_size} samples")
    logger.info(f"  val.txt: {val_size} samples")
    logger.info(f"  test.txt: {test_size} samples")

    # show examples
    logger.info(f"\nexample problems:")
    for i in range(10):
        logger.info(f"  {train_samples[i]}")


if __name__ == "__main__":
    generate_and_save_dataset(
        train_size=20000, val_size=2000, test_size=1000, max_num=99, operations=["add"]
    )
