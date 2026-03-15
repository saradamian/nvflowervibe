"""
Tests for LLM federated client.

Covers LLMClient initialization, inheritance from BaseFederatedClient,
compute_update (training), evaluate, get_parameters, and client_fn factory.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.slow

from unittest.mock import MagicMock

from sfl.client.base import BaseFederatedClient
from sfl.llm.client import LLMClient, client_fn
from sfl.llm.model import get_parameters


@pytest.fixture(scope="module")
def llm_client():
    """Create an LLMClient on CPU with tiny config for testing."""
    return LLMClient(
        client_id=0,
        partition_id=0,
        num_partitions=2,
        model_name="gpt2",
        device="cpu",
        local_epochs=1,
        batch_size=2,
        max_length=32,
    )


class TestLLMClientInit:

    def test_extends_base_and_has_data(self, llm_client):
        """Client inherits from BaseFederatedClient and loads data."""
        assert isinstance(llm_client, BaseFederatedClient)
        assert llm_client.device == "cpu"
        assert llm_client.model is not None
        assert len(llm_client.train_data) > 0

    def test_has_eval_data(self, llm_client):
        """Client should have an eval split."""
        # eval_data may be empty for very small partitions, but should exist
        assert hasattr(llm_client, "eval_data")


class TestLLMClientMethods:

    def test_get_parameters_returns_numpy(self, llm_client):
        params = llm_client.get_parameters({})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_compute_update_returns_triple(self, llm_client):
        """compute_update should return (params, num_examples, metrics)."""
        params = llm_client.get_parameters({})
        result = llm_client.compute_update(params, {})
        assert len(result) == 3
        updated_params, num_examples, metrics = result
        assert isinstance(updated_params, list)
        assert isinstance(num_examples, int)
        assert isinstance(metrics, dict)

    def test_compute_update_metrics(self, llm_client):
        params = llm_client.get_parameters({})
        _, _, metrics = llm_client.compute_update(params, {})
        assert "client_id" in metrics
        assert "train_loss" in metrics
        assert "num_examples" in metrics

    def test_evaluate_returns_loss(self, llm_client):
        params = llm_client.get_parameters({})
        loss, num_examples, metrics = llm_client.evaluate(params, {})
        assert isinstance(loss, float)
        assert loss > 0
        assert num_examples > 0
        assert "eval_loss" in metrics

    def test_training_changes_parameters(self):
        """A training round should produce different parameters."""
        client = LLMClient(
            client_id=0,
            partition_id=0,
            num_partitions=1,
            model_name="gpt2",
            device="cpu",
            local_epochs=1,
            batch_size=2,
            max_length=32,
        )
        params_before = [p.copy() for p in client.get_parameters({})]
        updated_params, _, _ = client.compute_update(params_before, {})
        changed = any(
            not np.array_equal(a, b)
            for a, b in zip(params_before, updated_params)
        )
        assert changed


class TestClientFn:

    def test_returns_client(self):
        """client_fn should return a Flower Client."""
        from flwr.client import Client
        context = MagicMock()
        context.node_config = {"partition-id": 0, "num-partitions": 1}
        context.node_id = 0
        context.run_config = {"max-length": 32}
        result = client_fn(context)
        assert isinstance(result, Client)

    def test_reads_partition_from_context(self):
        context = MagicMock()
        context.node_config = {"partition-id": 1, "num-partitions": 2}
        context.node_id = 1
        context.run_config = {"max-length": 32}
        # Should not raise -- partition 1 of 2 is valid
        client_fn(context)
