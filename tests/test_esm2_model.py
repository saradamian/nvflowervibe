"""
Tests for ESM2 model utilities.

Covers model/tokenizer loading, parameter extraction and injection,
and roundtrip serialization. Uses the real ESM2 model (smallest variant).
"""

import numpy as np
import pytest
import torch

from sfl.esm2.model import (
    DEFAULT_MODEL_NAME,
    get_parameters,
    load_model,
    load_tokenizer,
    set_parameters,
)


@pytest.fixture(scope="module")
def esm2_model():
    """Load ESM2 model once for all tests in this module."""
    return load_model(DEFAULT_MODEL_NAME)


@pytest.fixture(scope="module")
def esm2_tokenizer():
    """Load ESM2 tokenizer once for all tests in this module."""
    return load_tokenizer(DEFAULT_MODEL_NAME)


class TestLoadModel:

    def test_returns_model(self, esm2_model):
        assert esm2_model is not None
        assert hasattr(esm2_model, "parameters")

    def test_model_has_parameters(self, esm2_model):
        params = list(esm2_model.parameters())
        assert len(params) > 0

    def test_model_is_on_cpu(self, esm2_model):
        first_param = next(esm2_model.parameters())
        assert first_param.device.type == "cpu"


class TestLoadTokenizer:

    def test_returns_tokenizer(self, esm2_tokenizer):
        assert esm2_tokenizer is not None

    def test_tokenizer_can_encode(self, esm2_tokenizer):
        tokens = esm2_tokenizer("ACDEFGHIKLMNPQRSTVWY", return_tensors="pt")
        assert "input_ids" in tokens
        assert tokens["input_ids"].shape[0] == 1

    def test_tokenizer_has_mask_token(self, esm2_tokenizer):
        assert esm2_tokenizer.mask_token_id is not None


class TestGetParameters:

    def test_returns_list_of_numpy(self, esm2_model):
        params = get_parameters(esm2_model)
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_parameter_count_matches_state_dict(self, esm2_model):
        params = get_parameters(esm2_model)
        assert len(params) == len(esm2_model.state_dict())

    def test_parameter_shapes_match(self, esm2_model):
        params = get_parameters(esm2_model)
        for param_np, (_, tensor) in zip(params, esm2_model.state_dict().items()):
            assert param_np.shape == tuple(tensor.shape)


class TestSetParameters:

    def test_roundtrip(self, esm2_model):
        """get → set → get should produce identical arrays."""
        original = get_parameters(esm2_model)
        # Perturb, then restore
        noisy = [p + 0.01 for p in original]
        set_parameters(esm2_model, noisy)
        after_noise = get_parameters(esm2_model)
        # Should match the noisy version
        for a, b in zip(after_noise, noisy):
            np.testing.assert_array_almost_equal(a, b)
        # Restore original
        set_parameters(esm2_model, original)

    def test_count_mismatch_raises(self, esm2_model):
        params = get_parameters(esm2_model)
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            set_parameters(esm2_model, params[:-1])

    def test_updates_weights(self):
        """Setting parameters actually changes model weights."""
        model = load_model()
        original = get_parameters(model)
        zeroed = [np.zeros_like(p) for p in original]
        set_parameters(model, zeroed)
        after = get_parameters(model)
        for p in after:
            np.testing.assert_array_equal(p, np.zeros_like(p))
