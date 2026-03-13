"""
Dataset utilities for ESM2 federated learning.

Handles loading protein sequence datasets, tokenization, masking for MLM,
and partitioning data across federated clients.
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
        f"{len(partition_indices)} samples (indices {start}–{end - 1})"
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

# Representative protein sequences for demo/testing when no external dataset
# is available. These are short, real protein family motifs.
DEMO_SEQUENCES: List[str] = [
    "MKTLLILAVLCLGFAQAKIPYEEGP",
    "MVLSPADKTNVKAAWGKVGAHAGEY",
    "MNIFEMLRIDEGLRLKIYKDTE",
    "GKDGQPKISSVTCYLCGSDEMFE",
    "MSTAKLISQWFIFGHTFQNAYEH",
    "PKYVKQNTLKLATGMRNVPEKQTRGIF",
    "MLKKIMLSLVCAFPFAVTGAQIE",
    "MGLSDGEWQLVLNVWGKVEADIPGHG",
    "MKFLILLFNILCLFPVLAADNHVVV",
    "FVNQHLCGSHLVEALYLVCGERGFF",
    "MVHLTPEEKSAVTALWGKVNV",
    "MKTIQILVAYVIFLTSGFAHS",
    "MPHSSALTPETGEIFYYDIANK",
    "MKLLILTCLVAVALARPKHPIKHQGL",
    "GIVEQCCTSICSLYQLENYCN",
    "MALWMRLLPLLALLALWGPD",
    "MVKVYAPASSANMVHSSKTSIS",
    "MKWVTFISLLFLFSSAYSRGVFRR",
    "APRLICDSRVLERYLLEAKEAE",
    "MDSKGSSQKGSRLLLLLVVSNL",
    "MNFLLSWVHWSLALLLYLHHAK",
    "MEFSSPSREECPKPLSRVHFG",
    "MASRLLLLLLLLLLCGAQAIVEE",
    "MDSLAHMCFFFFLCSFVFAK",
    "MKIVILLLLCLLGITTQQPVLT",
    "MALSWRFLTILSGLLVLQVEGSK",
    "MYSFVSEETGTLIVNSVLL",
    "MARGSLALLLLLLAGCSRPIQI",
    "MKLLMILLGLTLASGCQANSND",
    "MELSVLLFLALLTGLVSCLGQR",
    "MPIFSDRVTLLASAVALLSAGLCS",
    "MKSILLGLAVYLLAGSCSVEQK",
]


class ProteinMLMDataset(Dataset):
    """Protein sequence dataset for masked language modeling.

    Tokenizes protein sequences and applies random masking for MLM training,
    mirroring the ESM2 pretraining objective.

    Args:
        sequences: List of protein sequences (amino acid strings).
        tokenizer: HuggingFace tokenizer for ESM2.
        max_length: Maximum sequence length after tokenization.
        mask_probability: Fraction of tokens to mask for MLM.
    """

    def __init__(
        self,
        sequences: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mask_probability: float = 0.15,
    ) -> None:
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]

        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)  # type: ignore[union-attr]
        attention_mask = encoding["attention_mask"].squeeze(0)  # type: ignore[union-attr]

        # Create labels (copy of input_ids)
        labels = input_ids.clone()

        # Create mask: only mask actual tokens (not special tokens or padding)
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            ),
            dtype=torch.bool,
        )
        probability_matrix = torch.full(labels.shape, self.mask_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels: -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of masked tokens → [MASK], 10% → random, 10% → unchanged
        indices_replaced = masked_indices & (torch.rand(labels.shape) < 0.8)
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            masked_indices & ~indices_replaced & (torch.rand(labels.shape) < 0.5)
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_demo_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    mask_probability: float = 0.15,
) -> ProteinMLMDataset:
    """Load the built-in demo protein sequence dataset.

    Args:
        tokenizer: ESM2 tokenizer.
        max_length: Maximum sequence length.
        mask_probability: MLM mask probability.

    Returns:
        ProteinMLMDataset with demo sequences.
    """
    logger.info(f"Loading demo dataset with {len(DEMO_SEQUENCES)} sequences")
    return ProteinMLMDataset(
        sequences=DEMO_SEQUENCES,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_probability=mask_probability,
    )


def load_dataset_from_hub(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    sequence_column: str = "sequence",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 128,
    mask_probability: float = 0.15,
) -> ProteinMLMDataset:
    """Load a protein dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        tokenizer: ESM2 tokenizer.
        sequence_column: Name of the column containing protein sequences.
        split: Dataset split to use.
        max_samples: Maximum number of samples to load (None = all).
        max_length: Maximum sequence length.
        mask_probability: MLM mask probability.

    Returns:
        ProteinMLMDataset with loaded sequences.
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
    ds = load_dataset(dataset_name, split=split)

    sequences = ds[sequence_column]
    if max_samples is not None:
        sequences = sequences[:max_samples]

    logger.info(f"Loaded {len(sequences)} sequences from {dataset_name}")
    return ProteinMLMDataset(
        sequences=sequences,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_probability=mask_probability,
    )
