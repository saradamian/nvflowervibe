"""
Tests for ESM2 federated client.

Covers ESM2Client initialization, inheritance from BaseFederatedClient,
compute_update (training), evaluate, get_parameters, and client_fn factory.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from sfl.client.base import BaseFederatedClient
from sfl.esm2.client import ESM2Client, client_fn
from sfl.esm2.model import get_parameters, load_model, DEFAULT_MODEL_NAME


@pytest.fixture(scope="module")
def esm2_client():
    """Create an ESM2Client on CPU for testing."""
    return ESM2Client(
        client_id=0,
        partition_id=0,
        num_partitions=2,
        device="cpu",
        local_epochs=1,
        batch_size=4,
    )


class TestESM2ClientInit:

    def test_extends_base_and_has_data(self, esm2_client):
        """Client inherits from BaseFederatedClient and loads data."""
        assert isinstance(esm2_client, BaseFederatedClient)
        assert esm2_client.device == "cpu"
        assert esm2_client.model is not None
        assert len(esm2_client.train_data) > 0


class TestESM2ClientMethods:

    def test_get_parameters_returns_numpy(self, esm2_client):
        params = esm2_client.get_parameters({})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_compute_update_returns_triple(self, esm2_client):
        """compute_update should return (params, num_examples, metrics)."""
        params = esm2_client.get_parameters({})
        result = esm2_client.compute_update(params, {})
        assert len(result) == 3
        updated_params, num_examples, metrics = result
        assert isinstance(updated_params, list)
        assert isinstance(num_examples, int)
        assert isinstance(metrics, dict)

    def test_compute_update_metrics(self, esm2_client):
        params = esm2_client.get_parameters({})
        _, _, metrics = esm2_client.compute_update(params, {})
        assert "client_id" in metrics
        assert "train_loss" in metrics
        assert "num_examples" in metrics

    def test_fit_delegates_to_compute_update(self, esm2_client):
        """fit() (from BaseFederatedClient) should call compute_update()."""
        params = esm2_client.get_parameters({})
        result = esm2_client.fit(params, {})
        assert len(result) == 3
        updated_params, num_examples, metrics = result
        assert isinstance(updated_params, list)
        assert num_examples > 0

    def test_evaluate_returns_loss(self, esm2_client):
        params = esm2_client.get_parameters({})
        loss, num_examples, metrics = esm2_client.evaluate(params, {})
        assert isinstance(loss, float)
        assert loss > 0  # MLM on random data should have some loss
        assert num_examples > 0
        assert "eval_loss" in metrics

    def test_training_changes_parameters(self):
        """A training round should produce different parameters."""
        client = ESM2Client(
            client_id=0, partition_id=0, num_partitions=1,
            device="cpu", local_epochs=1, batch_size=4,
        )
        params_before = [p.copy() for p in client.get_parameters({})]
        updated_params, _, _ = client.compute_update(params_before, {})
        # At least some parameters should change after training
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
        context.run_config = {}
        result = client_fn(context)
        assert isinstance(result, Client)

    def test_reads_partition_from_context(self):
        context = MagicMock()
        context.node_config = {"partition-id": 1, "num-partitions": 2}
        context.node_id = 1
        context.run_config = {}
        # Should not raise — partition 1 of 2 is valid
        client_fn(context)

    def test_string_partition_id_handled(self):
        """partition-id might come as a string from NVFlare."""
        context = MagicMock()
        context.node_config = {"partition-id": "0", "num-partitions": "1"}
        context.node_id = 0
        context.run_config = {}
        client_fn(context)
