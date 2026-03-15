"""
Tests for LLM model utilities.

Covers model/tokenizer loading, parameter extraction and injection,
roundtrip serialization, and LoRA adapter management.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.slow

from sfl.llm.model import (
    DEFAULT_MODEL_NAME,
    get_parameters,
    load_model,
    load_tokenizer,
    set_parameters,
)


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 model once for all tests in this module."""
    return load_model("gpt2")


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return load_tokenizer("gpt2")


class TestLoadModel:

    def test_returns_pretrained_model(self, gpt2_model):
        from transformers import PreTrainedModel
        assert isinstance(gpt2_model, PreTrainedModel)

    def test_default_model_name(self):
        assert DEFAULT_MODEL_NAME == "gpt2"


class TestLoadTokenizer:

    def test_returns_tokenizer(self, gpt2_tokenizer):
        from transformers import PreTrainedTokenizerBase
        assert isinstance(gpt2_tokenizer, PreTrainedTokenizerBase)

    def test_pad_token_set(self, gpt2_tokenizer):
        """GPT-2 tokenizer should have pad_token assigned."""
        assert gpt2_tokenizer.pad_token is not None


class TestGetParameters:

    def test_parameter_count_matches_state_dict(self, gpt2_model):
        params = get_parameters(gpt2_model)
        state_dict = gpt2_model.state_dict()
        assert len(params) == len(state_dict)

    def test_parameter_shapes_match(self, gpt2_model):
        params = get_parameters(gpt2_model)
        for param_np, (_, tensor) in zip(params, gpt2_model.state_dict().items()):
            assert isinstance(param_np, np.ndarray)
            assert param_np.shape == tuple(tensor.shape)


class TestSetParameters:

    def test_roundtrip(self, gpt2_model):
        """get -> set -> get should produce identical arrays."""
        original = get_parameters(gpt2_model)
        noisy = [p + 0.01 for p in original]
        set_parameters(gpt2_model, noisy)
        after_noise = get_parameters(gpt2_model)
        for a, b in zip(after_noise, noisy):
            np.testing.assert_array_almost_equal(a, b)
        # Restore original
        set_parameters(gpt2_model, original)

    def test_count_mismatch_raises(self, gpt2_model):
        params = get_parameters(gpt2_model)
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            set_parameters(gpt2_model, params[:-1])

    def test_updates_weights(self):
        """Setting parameters actually changes model weights."""
        model = load_model("gpt2")
        original = get_parameters(model)
        zeroed = [np.zeros_like(p) for p in original]
        set_parameters(model, zeroed)
        after = get_parameters(model)
        for p in after:
            np.testing.assert_array_equal(p, np.zeros_like(p))


class TestLoRA:
    """Test LoRA adapter application and parameter extraction."""

    @pytest.fixture(scope="class")
    def peft(self):
        return pytest.importorskip("peft")

    @pytest.fixture(scope="class")
    def lora_model(self, peft):
        from sfl.llm.model import apply_lora
        model = load_model("gpt2")
        return apply_lora(model, r=4, alpha=8)

    def test_apply_lora_freezes_base(self, lora_model):
        """Base model parameters should be frozen after LoRA."""
        from sfl.llm.model import get_lora_parameters
        # Total params > LoRA params means base is frozen
        total = sum(p.numel() for p in lora_model.parameters())
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        assert trainable < total

    def test_get_lora_parameters_subset(self, lora_model):
        """get_lora_parameters returns fewer arrays than get_parameters."""
        from sfl.llm.model import get_lora_parameters
        lora_params = get_lora_parameters(lora_model)
        full_params = get_parameters(lora_model)
        assert len(lora_params) > 0
        assert len(lora_params) < len(full_params)

    def test_set_lora_parameters_roundtrip(self, lora_model):
        """get_lora -> set_lora -> get_lora should produce identical arrays."""
        from sfl.llm.model import get_lora_parameters, set_lora_parameters
        original = get_lora_parameters(lora_model)
        noisy = [p + 0.01 for p in original]
        set_lora_parameters(lora_model, noisy)
        after = get_lora_parameters(lora_model)
        for a, b in zip(after, noisy):
            np.testing.assert_array_almost_equal(a, b)
        # Restore
        set_lora_parameters(lora_model, original)

    def test_set_lora_count_mismatch_raises(self, lora_model):
        from sfl.llm.model import get_lora_parameters, set_lora_parameters
        params = get_lora_parameters(lora_model)
        with pytest.raises(ValueError, match="LoRA parameter count mismatch"):
            set_lora_parameters(lora_model, params[:-1])
