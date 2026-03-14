"""
Integration tests for the SFL core federated learning pipeline.

These tests verify end-to-end functionality: config → client → server
strategy → aggregation. They ensure the FL loop produces correct results
without starting a full simulation (no Ray, no network).

Run automatically on PR to main via CI.
"""

from typing import List, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
from flwr.common import (
    FitRes,
    Parameters,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from sfl.client.sum_client import SumClient
from sfl.server.strategy import SumFedAvg
from sfl.utils.config import load_config, reset_config


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_fit_result(value: float) -> FitRes:
    """Build a FitRes with a single scalar parameter."""
    params = ndarrays_to_parameters([np.array([value], dtype=np.float32)])
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=1,
        metrics={"client_secret": value},
    )


def _make_client_proxy(cid: str) -> ClientProxy:
    """Build a mock ClientProxy with a given client id."""
    proxy = MagicMock(spec=ClientProxy)
    proxy.cid = cid
    return proxy  # type: ignore[return-value]


# ── Strategy aggregation ───────────────────────────────────────────────────

class TestSumAggregation:
    """Test that SumFedAvg correctly sums client values."""

    def _run_aggregation(self, client_values: list[float]):
        """Helper: feed values through SumFedAvg and return metrics."""
        initial = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        strategy = SumFedAvg(
            initial_parameters=initial,
            min_fit_clients=len(client_values),
            min_available_clients=len(client_values),
        )

        results = [
            (_make_client_proxy(str(i)), _make_fit_result(v))
            for i, v in enumerate(client_values)
        ]

        aggregated = strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )
        assert aggregated is not None
        _, metrics = aggregated
        return metrics

    def test_two_client_sum(self):
        metrics = self._run_aggregation([7.0, 8.0])
        assert metrics["federated_sum"] == pytest.approx(15.0)
        assert metrics["num_clients"] == 2

    def test_negative_values(self):
        metrics = self._run_aggregation([-5.0, 3.0])
        assert metrics["federated_sum"] == pytest.approx(-2.0)

    def test_aggregation_returns_parameters(self):
        initial = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        strategy = SumFedAvg(initial_parameters=initial)

        results = [
            (_make_client_proxy("0"), _make_fit_result(7.0)),
            (_make_client_proxy("1"), _make_fit_result(8.0)),
        ]

        aggregated = strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )
        assert aggregated is not None
        params, _ = aggregated
        arrays = parameters_to_ndarrays(params)
        assert len(arrays) == 1
        assert arrays[0].dtype == np.float32


# ── End-to-end: client → strategy ──────────────────────────────────────────

class TestEndToEndPipeline:
    """Test the full pipeline without Flower simulation runtime."""

    def test_clients_through_strategy(self):
        """Create real clients, feed their output through strategy."""
        clients = [
            SumClient(client_id=0, secret=7.0),
            SumClient(client_id=1, secret=8.0),
        ]

        # Simulate client fit calls
        client_outputs = [c.fit([], {}) for c in clients]

        # Build FitRes from client outputs
        results = []
        for i, (params, num_examples, metrics) in enumerate(client_outputs):
            fit_res = FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(params),
                num_examples=num_examples,
                metrics=metrics,
            )
            results.append((_make_client_proxy(str(i)), fit_res))

        # Run aggregation
        initial = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        strategy = SumFedAvg(initial_parameters=initial)
        aggregated = strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )

        assert aggregated is not None
        _, metrics = aggregated
        assert metrics["federated_sum"] == pytest.approx(15.0)
        assert metrics["num_clients"] == 2

    def test_pipeline_with_failures(self):
        """Strategy handles partial failures gracefully."""
        initial = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        strategy = SumFedAvg(
            initial_parameters=initial,
            min_fit_clients=1,
            min_available_clients=1,
        )

        results = [
            (_make_client_proxy("0"), _make_fit_result(7.0)),
        ]
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = [
            BaseException("client 1 failed"),
        ]

        aggregated = strategy.aggregate_fit(
            server_round=1, results=results, failures=failures
        )
        assert aggregated is not None
        _, metrics = aggregated
        assert metrics["federated_sum"] == pytest.approx(7.0)
        assert metrics["num_clients"] == 1


# ── Config → server_fn integration ─────────────────────────────────────────

class TestConfigIntegration:
    """Test that config flows correctly into the FL components."""

    def setup_method(self):
        reset_config()

    def test_default_config_produces_valid_federation(self):
        config = load_config()
        fed = config.federation
        assert fed.num_clients >= 1
        assert fed.num_rounds >= 1
        assert fed.min_available_clients <= fed.num_clients

    def test_config_validation_rejects_bad_values(self):
        with pytest.raises(ValueError):
            load_config(cli_overrides={"federation": {"num_clients": 0}})


# ── Strategy edge cases ────────────────────────────────────────────────────

class TestStrategyEdgeCases:
    """Test strategy robustness against unusual inputs."""

    def test_empty_results_sums_to_zero(self):
        initial = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        strategy = SumFedAvg(
            initial_parameters=initial,
            min_fit_clients=0,
            min_available_clients=0,
        )

        aggregated = strategy.aggregate_fit(
            server_round=1, results=[], failures=[]
        )
        # With no clients, sum is 0
        assert aggregated is not None
        _, metrics = aggregated
        assert metrics["federated_sum"] == 0
        assert metrics["num_clients"] == 0


