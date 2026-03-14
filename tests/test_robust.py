"""
Tests for Byzantine-robust aggregation strategies.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from sfl.server.robust import MultiKrumFedAvg, TrimmedMeanFedAvg


def _make_result(values, num_examples=1):
    """Build a (ClientProxy, FitRes) tuple from a list of arrays."""
    proxy = MagicMock(spec=ClientProxy)
    params = ndarrays_to_parameters([np.array(v, dtype=np.float32) for v in values])
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=num_examples,
        metrics={},
    )
    return (proxy, fit_res)


# ── Multi-Krum ──────────────────────────────────────────────────────────────

class TestMultiKrumFedAvg:

    def test_excludes_outlier(self):
        """With one clearly adversarial client, Krum should exclude it."""
        # 4 honest clients near [1,1,1], 1 adversary at [100,100,100]
        honest = [[1.0, 1.0, 1.0]] * 4
        adversary = [[100.0, 100.0, 100.0]]
        results = [_make_result([v]) for v in honest + adversary]

        strategy = MultiKrumFedAvg(
            num_byzantine=1,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None
        aggregated = parameters_to_ndarrays(params)
        # Result should be near [1,1,1], not pulled toward 100
        np.testing.assert_allclose(aggregated[0], [1.0, 1.0, 1.0], atol=0.01)

    def test_fallback_on_small_n(self):
        """With too few clients, falls back to FedAvg."""
        results = [_make_result([[2.0]]), _make_result([[4.0]])]
        strategy = MultiKrumFedAvg(
            num_byzantine=1,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None
        # FedAvg fallback: average of 2 and 4 = 3
        aggregated = parameters_to_ndarrays(params)
        np.testing.assert_allclose(aggregated[0], [3.0], atol=0.1)

    def test_matches_fedavg_no_adversaries(self):
        """With num_byzantine=0 and all identical clients, result equals FedAvg."""
        results = [_make_result([[5.0]])] * 5
        strategy = MultiKrumFedAvg(
            num_byzantine=0,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        aggregated = parameters_to_ndarrays(params)
        np.testing.assert_allclose(aggregated[0], [5.0], atol=0.01)


# ── Trimmed Mean ────────────────────────────────────────────────────────────

class TestTrimmedMeanFedAvg:

    def test_trims_extremes(self):
        """Trimmed mean should exclude the highest and lowest values."""
        # 5 clients, trim_ratio=0.2 → trim 1 from each side
        values = [[1.0], [2.0], [3.0], [4.0], [100.0]]
        results = [_make_result([v]) for v in values]

        strategy = TrimmedMeanFedAvg(
            trim_ratio=0.2,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None
        aggregated = parameters_to_ndarrays(params)
        # After trimming 1.0 and 100.0: mean(2,3,4) = 3.0
        np.testing.assert_allclose(aggregated[0], [3.0], atol=0.01)

    def test_fallback_on_small_n(self):
        """With only 2 clients, falls back to FedAvg."""
        results = [_make_result([[2.0]]), _make_result([[4.0]])]
        strategy = TrimmedMeanFedAvg(
            trim_ratio=0.1,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        aggregated = parameters_to_ndarrays(params)
        np.testing.assert_allclose(aggregated[0], [3.0], atol=0.1)

    def test_invalid_trim_ratio(self):
        """trim_ratio must be in [0, 0.5)."""
        with pytest.raises(ValueError, match="trim_ratio"):
            TrimmedMeanFedAvg(trim_ratio=0.5)

    def test_multidim_parameters(self):
        """Works with multi-dimensional parameter arrays."""
        # 5 clients, each with a 2x2 array
        values = [
            [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            [np.array([[99, 99], [99, 99]], dtype=np.float32)],
            [np.array([[-99, -99], [-99, -99]], dtype=np.float32)],
        ]
        results = [
            _make_result(v) for v in values
        ]
        strategy = TrimmedMeanFedAvg(
            trim_ratio=0.2,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        aggregated = parameters_to_ndarrays(params)
        np.testing.assert_allclose(
            aggregated[0], [[1, 2], [3, 4]], atol=0.01,
        )
