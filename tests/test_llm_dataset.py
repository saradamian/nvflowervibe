"""
Tests for LLM dataset utilities.

Covers CausalLMDataset construction, tokenization, demo dataset loading,
train/eval splitting, and IID partitioning.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.slow

from sfl.llm.dataset import (
    DEMO_TEXTS,
    CausalLMDataset,
    load_demo_dataset,
    partition_dataset,
    split_train_eval,
)
from sfl.llm.model import load_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("gpt2")


@pytest.fixture(scope="module")
def demo_dataset(tokenizer):
    return load_demo_dataset(tokenizer, max_length=32)


class TestCausalLMDataset:

    def test_length(self, tokenizer):
        texts = ["Hello world.", "Testing one two three."]
        ds = CausalLMDataset(texts, tokenizer, max_length=32)
        assert len(ds) == 2

    def test_getitem_keys_shapes_types(self, tokenizer):
        """Items have correct keys, shapes, and tensor types."""
        ds = CausalLMDataset(["Hello world."], tokenizer, max_length=32)
        item = ds[0]
        for key in ("input_ids", "attention_mask", "labels"):
            assert key in item
            assert isinstance(item[key], torch.Tensor)
            assert item[key].shape == (32,)

    def test_labels_ignore_padding(self, tokenizer):
        """Padding positions should have label -100."""
        ds = CausalLMDataset(["Hi"], tokenizer, max_length=32)
        item = ds[0]
        # Where attention_mask is 0, labels must be -100
        padding_mask = item["attention_mask"] == 0
        if padding_mask.any():
            assert (item["labels"][padding_mask] == -100).all()

    def test_labels_match_input_ids_at_non_padding(self, tokenizer):
        """Non-padding label positions should equal input_ids (causal LM)."""
        ds = CausalLMDataset(["The quick brown fox."], tokenizer, max_length=32)
        item = ds[0]
        non_padding = item["attention_mask"] == 1
        assert (item["labels"][non_padding] == item["input_ids"][non_padding]).all()


class TestLoadDemoDataset:

    def test_loads_all_demo_texts(self, demo_dataset):
        """Demo dataset contains all built-in text samples."""
        assert isinstance(demo_dataset, CausalLMDataset)
        assert len(demo_dataset) == len(DEMO_TEXTS)

    def test_demo_texts_not_empty(self):
        """Built-in DEMO_TEXTS should have a reasonable number of samples."""
        assert len(DEMO_TEXTS) >= 10


class TestSplitTrainEval:

    def test_split_sizes(self, demo_dataset):
        """Train + eval should cover the full dataset."""
        train, eval_ = split_train_eval(demo_dataset, eval_fraction=0.2, seed=42)
        assert len(train) + len(eval_) == len(demo_dataset)

    def test_eval_has_at_least_one(self, demo_dataset):
        """Eval split should have at least 1 sample."""
        _, eval_ = split_train_eval(demo_dataset, eval_fraction=0.1, seed=42)
        assert len(eval_) >= 1

    def test_deterministic_with_seed(self, demo_dataset):
        """Same seed should produce the same split."""
        train1, eval1 = split_train_eval(demo_dataset, seed=123)
        train2, eval2 = split_train_eval(demo_dataset, seed=123)
        assert train1.indices == train2.indices
        assert eval1.indices == eval2.indices


class TestPartitionDataset:

    def test_partition_sizes_balanced(self, demo_dataset):
        n = len(demo_dataset)
        parts = [partition_dataset(demo_dataset, 4, i) for i in range(4)]
        sizes = [len(p) for p in parts]
        assert sum(sizes) == n
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
