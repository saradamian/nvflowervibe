"""
Dataset utilities for ESM2 federated learning.

Handles loading protein sequence datasets, tokenization, masking for MLM,
and partitioning data across federated clients.
"""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

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
        tokenizer: AutoTokenizer,
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

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

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
    tokenizer: AutoTokenizer,
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
    tokenizer: AutoTokenizer,
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


def partition_dataset(
    dataset: ProteinMLMDataset,
    num_partitions: int,
    partition_id: int,
) -> Subset:
    """Partition a dataset for federated learning (IID split).

    Divides the dataset into roughly equal, non-overlapping partitions.

    Args:
        dataset: Full dataset to partition.
        num_partitions: Total number of FL clients.
        partition_id: This client's partition index (0-based).

    Returns:
        Subset of the dataset for this client.

    Raises:
        ValueError: If partition_id is out of range.
    """
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id {partition_id} out of range [0, {num_partitions})"
        )

    total = len(dataset)
    partition_size = total // num_partitions
    remainder = total % num_partitions

    # Distribute remainder across first partitions
    start = partition_id * partition_size + min(partition_id, remainder)
    end = start + partition_size + (1 if partition_id < remainder else 0)

    indices = list(range(start, end))
    logger.info(
        f"Partition {partition_id}/{num_partitions}: "
        f"{len(indices)} samples (indices {start}–{end - 1})"
    )
    return Subset(dataset, indices)
