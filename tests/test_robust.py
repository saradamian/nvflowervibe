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

from sfl.server.robust import MultiKrumFedAvg, TrimmedMeanFedAvg, FoundationFLFedAvg


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

    def test_excludes_outlier_high_dim(self):
        """Krum with JL projection should still exclude outliers (C3)."""
        from sfl.server.robust import _KRUM_DIM_THRESHOLD
        d = _KRUM_DIM_THRESHOLD + 100  # force projection
        honest = [np.ones(d, dtype=np.float32)] * 4
        adversary = [np.ones(d, dtype=np.float32) * 100.0]
        results = [_make_result([v]) for v in honest + adversary]

        strategy = MultiKrumFedAvg(
            num_byzantine=1,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None
        aggregated = parameters_to_ndarrays(params)
        # Should still be near 1.0, not pulled toward 100
        np.testing.assert_allclose(aggregated[0][:5], [1.0] * 5, atol=0.01)

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


# ── FoundationFL ──────────────────────────────────────────────────────────

class TestFoundationFLFedAvg:
    """Tests for FoundationFLFedAvg (NDSS 2025)."""

    def test_excludes_adversary_with_root(self):
        """With a root update, adversarial clients should be excluded."""
        honest = [[1.0, 1.0, 1.0, 1.0, 1.0]] * 5
        adversary = [[-100.0, -100.0, -100.0, -100.0, -100.0]]
        results = [_make_result([v]) for v in honest + adversary]

        root = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        strategy = FoundationFLFedAvg(
            root_update=root,
            trust_threshold=0.0,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, metrics = strategy.aggregate_fit(1, results, [])
        assert params is not None
        aggregated = parameters_to_ndarrays(params)
        # Result should be near [1,1,1,1,1], not pulled toward -100
        np.testing.assert_allclose(aggregated[0], [1.0] * 5, atol=0.5)

    def test_keeps_all_when_honest(self):
        """When all clients are similar and aligned with root, none removed."""
        values = [[1.0, 2.0, 3.0]] * 6
        results = [_make_result([v]) for v in values]

        root = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        strategy = FoundationFLFedAvg(
            root_update=root,
            trust_threshold=0.0,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None
        aggregated = parameters_to_ndarrays(params)
        np.testing.assert_allclose(aggregated[0], [1.0, 2.0, 3.0], atol=0.01)

    def test_fallback_without_root(self):
        """Without root update, uses client mean as reference and still aggregates."""
        # All clients similar — no root needed, mean-based reference works
        values = [[1.0, 2.0]] * 4
        results = [_make_result([v]) for v in values]

        strategy = FoundationFLFedAvg(
            root_update=None,
            trust_threshold=0.0,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None
        aggregated = parameters_to_ndarrays(params)
        np.testing.assert_allclose(aggregated[0], [1.0, 2.0], atol=0.1)

    def test_weighted_averaging(self):
        """With weighted=True, more similar clients should have higher weight."""
        # Two clients: one closer to root, one farther
        results = [
            _make_result([[1.0, 1.0, 1.0]]),  # very similar to root
            _make_result([[0.5, 0.5, 0.5]]),  # similar direction, smaller
        ]
        root = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        strategy = FoundationFLFedAvg(
            root_update=root,
            trust_threshold=-1.0,  # keep all
            weighted=True,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None

    def test_binary_filtering(self):
        """With weighted=False, kept clients get equal weight."""
        results = [
            _make_result([[2.0, 2.0]]),
            _make_result([[4.0, 4.0]]),
        ]
        root = np.array([1.0, 1.0], dtype=np.float32)
        strategy = FoundationFLFedAvg(
            root_update=root,
            trust_threshold=-1.0,
            weighted=False,
            min_fit_clients=1,
            min_available_clients=1,
        )
        params, _ = strategy.aggregate_fit(1, results, [])
        assert params is not None
        aggregated = parameters_to_ndarrays(params)
        np.testing.assert_allclose(aggregated[0], [3.0, 3.0], atol=0.1)

    def test_set_root_update(self):
        """set_root_update should update the reference vector."""
        strategy = FoundationFLFedAvg(
            root_update=None,
            min_fit_clients=1,
            min_available_clients=1,
        )
        root = np.array([1.0, 2.0], dtype=np.float32)
        strategy.set_root_update(root)
        assert strategy._root_update is not None
        np.testing.assert_array_equal(strategy._root_update, root)


# ── Norm Verification (H4) ─────────────────────────────────────────────────

class TestVerifyUpdateNorms:
    """Tests for verify_update_norms (H4)."""

    def test_rejects_oversized_update(self):
        """Updates exceeding max_norm should be filtered out."""
        from sfl.server.robust import verify_update_norms

        small = _make_result([[1.0, 0.0, 0.0]])  # norm = 1
        big = _make_result([[100.0, 100.0, 100.0]])  # norm ≈ 173

        results = [small, big]
        filtered = verify_update_norms(results, max_norm=10.0)
        assert len(filtered) == 1

    def test_keeps_valid_updates(self):
        """Updates within max_norm should be kept."""
        from sfl.server.robust import verify_update_norms

        results = [_make_result([[1.0, 1.0]]), _make_result([[2.0, 2.0]])]
        filtered = verify_update_norms(results, max_norm=100.0)
        assert len(filtered) == 2

    def test_empty_on_all_rejected(self):
        """If all updates exceed max_norm, result should be empty."""
        from sfl.server.robust import verify_update_norms

        results = [_make_result([[100.0] * 10])]
        filtered = verify_update_norms(results, max_norm=1.0)
        assert len(filtered) == 0
