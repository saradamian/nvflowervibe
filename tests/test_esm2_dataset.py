"""
Tests for ESM2 dataset utilities.

Covers ProteinMLMDataset construction, tokenization & masking logic,
demo dataset loading, and IID partitioning.
"""

import pytest
import torch

from sfl.esm2.dataset import (
    DEMO_SEQUENCES,
    ProteinMLMDataset,
    load_demo_dataset,
    partition_dataset,
)
from sfl.esm2.model import load_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def demo_dataset(tokenizer):
    return load_demo_dataset(tokenizer, max_length=64)


class TestProteinMLMDataset:

    def test_length(self, tokenizer):
        seqs = ["ACDEFG", "HIKLMN", "PQRSTV"]
        ds = ProteinMLMDataset(seqs, tokenizer, max_length=32)
        assert len(ds) == 3

    def test_getitem_keys(self, tokenizer):
        ds = ProteinMLMDataset(["ACDEFG"], tokenizer, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_shapes(self, tokenizer):
        ds = ProteinMLMDataset(["ACDEFG"], tokenizer, max_length=32)
        item = ds[0]
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["labels"].shape == (32,)

    def test_getitem_returns_tensors(self, tokenizer):
        ds = ProteinMLMDataset(["ACDEFG"], tokenizer, max_length=32)
        item = ds[0]
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_labels_have_ignored_positions(self, tokenizer):
        """Non-masked positions should have label -100."""
        ds = ProteinMLMDataset(["ACDEFGHIKLMNPQRSTVWY"], tokenizer, max_length=32)
        item = ds[0]
        # At least some labels should be -100 (non-masked)
        assert (item["labels"] == -100).any()

    def test_special_tokens_not_masked(self, tokenizer):
        """Padding positions should always have label -100."""
        ds = ProteinMLMDataset(
            ["ACDEFG"], tokenizer, max_length=32, mask_probability=1.0
        )
        item = ds[0]
        labels = item["labels"].tolist()
        for i in range(len(labels)):
            if item["attention_mask"][i] == 0:
                # Padding tokens should never be prediction targets
                assert labels[i] == -100

    def test_zero_mask_probability_no_masking(self, tokenizer):
        """With mask_probability=0, no tokens should be masked."""
        ds = ProteinMLMDataset(
            ["ACDEFGHIKLMNPQRSTVWY"], tokenizer, max_length=32, mask_probability=0.0
        )
        item = ds[0]
        # All labels should be -100 (nothing masked)
        assert (item["labels"] == -100).all()


class TestLoadDemoDataset:

    def test_returns_dataset(self, demo_dataset):
        assert isinstance(demo_dataset, ProteinMLMDataset)

    def test_correct_length(self, demo_dataset):
        assert len(demo_dataset) == len(DEMO_SEQUENCES)

    def test_sequences_stored(self, demo_dataset):
        assert demo_dataset.sequences is DEMO_SEQUENCES


class TestPartitionDataset:

    def test_partition_sizes_balanced(self, demo_dataset):
        n = len(demo_dataset)
        parts = [partition_dataset(demo_dataset, 4, i) for i in range(4)]
        sizes = [len(p) for p in parts]
        assert sum(sizes) == n
        # Sizes should differ by at most 1
        assert max(sizes) - min(sizes) <= 1

    def test_all_indices_covered(self, demo_dataset):
        n = len(demo_dataset)
        all_indices = []
        for i in range(4):
            p = partition_dataset(demo_dataset, 4, i)
            all_indices.extend(p.indices)
        assert sorted(all_indices) == list(range(n))

    def test_no_overlap(self, demo_dataset):
        all_indices = []
        for i in range(4):
            p = partition_dataset(demo_dataset, 4, i)
            all_indices.extend(p.indices)
        assert len(set(all_indices)) == len(all_indices)

    def test_invalid_partition_id_raises(self, demo_dataset):
        with pytest.raises(ValueError, match="out of range"):
            partition_dataset(demo_dataset, 4, 4)
        with pytest.raises(ValueError, match="out of range"):
            partition_dataset(demo_dataset, 4, -1)

    def test_single_partition(self, demo_dataset):
        p = partition_dataset(demo_dataset, 1, 0)
        assert len(p) == len(demo_dataset)
