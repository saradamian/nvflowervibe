"""
Dataset utilities for LLM federated fine-tuning.

Handles loading text datasets, causal LM tokenization (where labels
are the input_ids shifted by one), and partitioning data across
federated clients.

Includes built-in demo text samples so the module runs without
downloading any external data.
"""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from transformers import PreTrainedTokenizerBase

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


def partition_dataset(
    dataset: Dataset,
    num_partitions: int,
    partition_id: int,
) -> Subset:
    """Partition a dataset into non-overlapping IID subsets.

    Splits dataset indices evenly across partitions. If the dataset
    size isn't perfectly divisible, earlier partitions get one extra sample.

    Args:
        dataset: Full dataset to partition.
        num_partitions: Number of partitions to create.
        partition_id: Index of the partition to return (0-based).

    Returns:
        A Subset containing this partition's samples.

    Raises:
        ValueError: If partition_id is out of range or num_partitions < 1.
    """
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id {partition_id} out of range for {num_partitions} partitions"
        )

    n = len(dataset)  # type: ignore[arg-type]
    indices = list(range(n))

    # Split as evenly as possible
    chunk_size = n // num_partitions
    remainder = n % num_partitions

    start = partition_id * chunk_size + min(partition_id, remainder)
    end = start + chunk_size + (1 if partition_id < remainder else 0)

    partition_indices = indices[start:end]
    logger.info(
        f"Partition {partition_id}/{num_partitions}: "
        f"{len(partition_indices)} samples (indices {start}-{end - 1})"
    )
    return Subset(dataset, partition_indices)


def split_train_eval(
    dataset: Dataset,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """Split a dataset into train and eval subsets.

    Args:
        dataset: Dataset to split.
        eval_fraction: Fraction of data to use for evaluation (0.0-0.5).
        seed: Random seed for reproducible split.

    Returns:
        Tuple of (train_subset, eval_subset).
    """
    n = len(dataset)  # type: ignore[arg-type]
    eval_size = max(1, int(n * eval_fraction))
    train_size = n - eval_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    logger.info(f"Split dataset: {train_size} train, {eval_size} eval")
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


# Built-in demo texts for quick testing without downloading anything.
# Short, diverse samples covering different writing styles.
DEMO_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog near the old farmhouse.",
    "Machine learning models learn patterns from data to make predictions.",
    "Federated learning allows training on distributed data without centralizing it.",
    "The sun set behind the mountains, painting the sky in shades of orange and purple.",
    "Python is a versatile programming language used in many scientific domains.",
    "Privacy-preserving machine learning is essential for sensitive applications.",
    "The cat sat on the windowsill, watching birds flutter between the oak trees.",
    "Neural networks are composed of layers of interconnected artificial neurons.",
    "Differential privacy provides mathematical guarantees about information leakage.",
    "The old library contained thousands of books, each telling a unique story.",
    "Gradient descent optimizes model parameters by following the loss landscape.",
    "Secure aggregation ensures that individual updates remain private during training.",
    "The river wound its way through the valley, reflecting the morning light.",
    "Transformers use self-attention to process sequences in parallel efficiently.",
    "Large language models can generate coherent text across many different topics.",
    "The garden was full of colorful flowers that attracted butterflies all summer.",
    "Fine-tuning adapts a pretrained model to a specific downstream task.",
    "Communication efficiency is critical in federated learning across many devices.",
    "The clock tower struck midnight as the town fell into peaceful silence.",
    "Low-rank adaptation reduces the number of trainable parameters significantly.",
]


class CausalLMDataset(Dataset):
    """Text dataset for causal language model fine-tuning.

    Tokenizes text samples for autoregressive language modeling where
    the model predicts each next token. Labels are the input_ids
    themselves -- the HuggingFace CausalLM loss internally shifts
    them so that position i predicts position i+1.

    Args:
        texts: List of text strings.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length after tokenization.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)  # type: ignore[union-attr]
        attention_mask = encoding["attention_mask"].squeeze(0)  # type: ignore[union-attr]

        # For causal LM: labels = input_ids. The model's loss function
        # internally shifts so position i predicts position i+1.
        # Set labels to -100 where attention_mask is 0 (padding) so
        # padding tokens don't contribute to the loss.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_demo_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> CausalLMDataset:
    """Load the built-in demo text dataset.

    Args:
        tokenizer: Tokenizer for the causal LM.
        max_length: Maximum sequence length.

    Returns:
        CausalLMDataset with demo texts.
    """
    logger.info(f"Loading demo dataset with {len(DEMO_TEXTS)} text samples")
    return CausalLMDataset(
        texts=DEMO_TEXTS,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def load_dataset_from_hub(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 128,
) -> CausalLMDataset:
    """Load a text dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "wikitext").
        tokenizer: Tokenizer for the causal LM.
        text_column: Name of the column containing text.
        split: Dataset split to use.
        max_samples: Maximum number of samples to load (None = all).
        max_length: Maximum sequence length.

    Returns:
        CausalLMDataset with loaded texts.
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
    ds = load_dataset(dataset_name, split=split)

    texts = ds[text_column]
    # Filter out empty strings which are common in text datasets
    texts = [t for t in texts if t and t.strip()]
    if max_samples is not None:
        texts = texts[:max_samples]

    logger.info(f"Loaded {len(texts)} text samples from {dataset_name}")
    return CausalLMDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )
