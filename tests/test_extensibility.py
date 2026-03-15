"""Tests for extensibility features: InferenceClient, params, adapter mask mod."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from sfl.client.inference import BaseInferenceClient, InferenceResult
from sfl.utils.params import downcast_parameters, upcast_parameters


# ── BaseInferenceClient ──────────────────────────────────────────────


class DummyInferenceClient(BaseInferenceClient):
    """Concrete implementation for testing."""

    def compute_predictions(self, parameters, config):
        return {"preds": [1, 2, 3]}, 3, {"accuracy": 0.95}


class TestBaseInferenceClient:
    def test_get_parameters_returns_empty(self):
        client = DummyInferenceClient()
        assert client.get_parameters({}) == []

    def test_evaluate_delegates_to_compute_predictions(self):
        client = DummyInferenceClient()
        params = [np.array([1.0, 2.0])]
        loss, n, metrics = client.evaluate(params, {})
        assert n == 3
        assert metrics["accuracy"] == 0.95

    def test_evaluate_extracts_loss_from_metrics(self):
        class LossClient(BaseInferenceClient):
            def compute_predictions(self, parameters, config):
                return None, 10, {"loss": 0.42}

        client = LossClient()
        loss, n, metrics = client.evaluate([np.zeros(5)], {})
        assert loss == 0.42
        assert n == 10

    def test_evaluate_defaults_loss_to_zero(self):
        client = DummyInferenceClient()
        loss, _, _ = client.evaluate([np.zeros(5)], {})
        assert loss == 0.0

    def test_fit_warns_and_delegates(self):
        client = DummyInferenceClient()
        params = [np.array([1.0])]
        # fit() should still work (returns evaluate results)
        result = client.fit(params, {})
        assert len(result) == 3

    def test_inference_result_type_alias(self):
        # InferenceResult should be a valid tuple type
        result: InferenceResult = ({"preds": []}, 0, {"metric": 1.0})
        assert len(result) == 3

    def test_exported_from_sfl_init(self):
        import sfl
        assert hasattr(sfl, "BaseInferenceClient")
        assert hasattr(sfl, "InferenceResult")

    def test_exported_from_client_init(self):
        from sfl.client import BaseInferenceClient as BIC
        assert BIC is BaseInferenceClient


# ── downcast/upcast parameters ──────────────────────────────────────


class TestParamsUtility:
    def test_downcast_reduces_precision(self):
        params = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        downcasted = downcast_parameters(params)
        assert downcasted[0].dtype == np.float16

    def test_upcast_restores_precision(self):
        params = [np.array([1.0, 2.0], dtype=np.float16)]
        upcasted = upcast_parameters(params)
        assert upcasted[0].dtype == np.float32

    def test_roundtrip_preserves_values(self):
        params = [np.array([1.0, 0.5, -0.25], dtype=np.float32)]
        roundtripped = upcast_parameters(downcast_parameters(params))
        np.testing.assert_allclose(roundtripped[0], params[0], atol=1e-3)

    def test_downcast_custom_dtype(self):
        params = [np.array([1.0], dtype=np.float64)]
        downcasted = downcast_parameters(params, dtype=np.float32)
        assert downcasted[0].dtype == np.float32

    def test_upcast_custom_dtype(self):
        params = [np.array([1.0], dtype=np.float16)]
        upcasted = upcast_parameters(params, dtype=np.float64)
        assert upcasted[0].dtype == np.float64

    def test_multiple_arrays(self):
        params = [
            np.ones((10, 10), dtype=np.float32),
            np.zeros((5,), dtype=np.float32),
        ]
        downcasted = downcast_parameters(params)
        assert len(downcasted) == 2
        assert all(p.dtype == np.float16 for p in downcasted)

    def test_exported_from_utils_init(self):
        from sfl.utils import downcast_parameters, upcast_parameters
        assert callable(downcast_parameters)
        assert callable(upcast_parameters)


# ── Adapter mask mod ────────────────────────────────────────────────


class TestAdapterMaskMod:
    def test_import(self):
        from sfl.privacy.filters import make_adapter_mask_mod
        assert callable(make_adapter_mask_mod)

    def test_exported_from_privacy_init(self):
        from sfl.privacy import make_adapter_mask_mod
        assert callable(make_adapter_mask_mod)

    def test_create_with_indices(self):
        from sfl.privacy.filters import make_adapter_mask_mod
        mod = make_adapter_mask_mod(adapter_indices=[0, 2])
        assert callable(mod)

    def test_create_with_pattern(self):
        from sfl.privacy.filters import make_adapter_mask_mod
        mod = make_adapter_mask_mod(
            adapter_pattern=r"lora_",
            param_names=["base.weight", "lora_A.weight", "base.bias", "lora_B.weight"],
        )
        assert callable(mod)
