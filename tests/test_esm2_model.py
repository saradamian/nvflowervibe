"""
Tests for ESM2 model utilities.

Covers model/tokenizer loading, parameter extraction and injection,
and roundtrip serialization. Uses the real ESM2 model (smallest variant).
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

from sfl.esm2.model import (
    DEFAULT_MODEL_NAME,
    get_parameters,
    load_model,
    set_parameters,
)


@pytest.fixture(scope="module")
def esm2_model():
    """Load ESM2 model once for all tests in this module."""
    return load_model(DEFAULT_MODEL_NAME)


class TestGetParameters:

    def test_parameter_count_and_shapes_match_state_dict(self, esm2_model):
        """get_parameters returns numpy arrays matching the model state dict."""
        params = get_parameters(esm2_model)
        state_dict = esm2_model.state_dict()
        assert len(params) == len(state_dict)
        for param_np, (_, tensor) in zip(params, state_dict.items()):
            assert isinstance(param_np, np.ndarray)
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
