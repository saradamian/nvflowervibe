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

    def test_getitem_keys_shapes_types(self, tokenizer):
        """Items have correct keys, shapes, and tensor types."""
        ds = ProteinMLMDataset(["ACDEFG"], tokenizer, max_length=32)
        item = ds[0]
        for key in ("input_ids", "attention_mask", "labels"):
            assert key in item
            assert isinstance(item[key], torch.Tensor)
            assert item[key].shape == (32,)

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

    def test_loads_all_demo_sequences(self, demo_dataset):
        """Demo dataset contains all built-in sequences."""
        assert isinstance(demo_dataset, ProteinMLMDataset)
        assert len(demo_dataset) == len(DEMO_SEQUENCES)


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
