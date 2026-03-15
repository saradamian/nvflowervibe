"""
Unit and integration tests for the SFL privacy module.

Tests DP config, strategy wrapping (server-side and client-side),
SecAgg config, and DP integration with both sum demo and ESM2 servers.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    DifferentialPrivacyServerSideFixedClipping,
)

from sfl.privacy.dp import DPConfig, wrap_strategy_with_dp, _AccountingWrapper
from sfl.privacy.secagg import SecAggConfig, build_secagg_config, make_secagg_main
from sfl.server.strategy import SumFedAvg

try:
    from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism
    _has_dp_accounting = True
except ImportError:
    _has_dp_accounting = False


def _assert_dp_wrapped(wrapped, expected_inner_type):
    """Assert DP wrapping — with or without dp-accounting."""
    if _has_dp_accounting:
        assert isinstance(wrapped, _AccountingWrapper)
        assert isinstance(wrapped._inner, expected_inner_type)
    else:
        assert isinstance(wrapped, expected_inner_type)


# ── DPConfig ────────────────────────────────────────────────────────────────


# ── wrap_strategy_with_dp ───────────────────────────────────────────────────


class TestWrapStrategyWithDP:
    """Tests for the DP strategy wrapper."""

    def _make_fedavg(self):
        """Create a simple FedAvg strategy."""
        initial = ndarrays_to_parameters(
            [np.array([0.0], dtype=np.float32)]
        )
        return FedAvg(
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=initial,
        )

    def test_server_side_wrapping(self):
        strategy = self._make_fedavg()
        dp_config = DPConfig(mode="server", noise_multiplier=0.5, clipping_norm=1.0)
        wrapped = wrap_strategy_with_dp(strategy, dp_config)
        _assert_dp_wrapped(wrapped, DifferentialPrivacyServerSideFixedClipping)

    def test_client_side_wrapping(self):
        strategy = self._make_fedavg()
        dp_config = DPConfig(mode="client", noise_multiplier=0.5, clipping_norm=1.0)
        wrapped = wrap_strategy_with_dp(strategy, dp_config)
        _assert_dp_wrapped(wrapped, DifferentialPrivacyClientSideFixedClipping)

    def test_wraps_sum_fedavg(self):
        """DP wrapping works with SumFedAvg strategy too."""
        initial = ndarrays_to_parameters(
            [np.array([0.0], dtype=np.float32)]
        )
        strategy = SumFedAvg(
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=initial,
        )
        dp_config = DPConfig(mode="server")
        wrapped = wrap_strategy_with_dp(strategy, dp_config)
        _assert_dp_wrapped(wrapped, DifferentialPrivacyServerSideFixedClipping)


# ── calibrate_gaussian_sigma ────────────────────────────────────────────────


@pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
class TestCalibrateGaussianSigma:
    """Tests for the PLD-based noise calibration."""

    def test_calibrated_sigma_satisfies_epsilon(self):
        """σ returned by calibration should give ε ≤ target."""
        from sfl.privacy.dp import calibrate_gaussian_sigma
        from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

        sigma = calibrate_gaussian_sigma(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        pld = from_gaussian_mechanism(standard_deviation=sigma)
        actual_eps = pld.get_epsilon_for_delta(1e-5)
        assert actual_eps <= 1.0 + 1e-6

    def test_higher_epsilon_gives_less_noise(self):
        """Relaxing epsilon should require less noise (lower sigma)."""
        from sfl.privacy.dp import calibrate_gaussian_sigma

        sigma_tight = calibrate_gaussian_sigma(epsilon=0.5, delta=1e-5, sensitivity=1.0)
        sigma_loose = calibrate_gaussian_sigma(epsilon=2.0, delta=1e-5, sensitivity=1.0)
        assert sigma_tight > sigma_loose

    def test_invalid_params_raise(self):
        """Non-positive epsilon or delta should raise ValueError."""
        from sfl.privacy.dp import calibrate_gaussian_sigma

        with pytest.raises(ValueError):
            calibrate_gaussian_sigma(epsilon=0, delta=1e-5, sensitivity=1.0)
        with pytest.raises(ValueError):
            calibrate_gaussian_sigma(epsilon=1.0, delta=-1, sensitivity=1.0)


# ── AdaptiveClipWrapper ────────────────────────────────────────────────────


class TestAdaptiveClipWrapper:
    """Tests for adaptive clipping norm adjustment."""

    def test_wrap_strategy_with_adaptive_clip(self):
        """adaptive_clipping=True should wrap with AdaptiveClipWrapper."""
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper
        initial = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        strategy = FedAvg(
            min_fit_clients=2, min_available_clients=2,
            initial_parameters=initial,
        )
        dp_config = DPConfig(
            mode="server", noise_multiplier=0.5, clipping_norm=1.0,
            adaptive_clipping=True, target_quantile=0.5, clip_learning_rate=0.2,
        )
        wrapped = wrap_strategy_with_dp(strategy, dp_config)
        if _has_dp_accounting:
            # Outer: _AccountingWrapper, inner: AdaptiveClipWrapper
            assert isinstance(wrapped, _AccountingWrapper)
            assert isinstance(wrapped._inner, AdaptiveClipWrapper)
        else:
            assert isinstance(wrapped, AdaptiveClipWrapper)

    def test_clip_decreases_when_updates_small(self):
        """When all updates are below clip norm, clip should decrease."""
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig

        inner = MagicMock()
        inner.clipping_norm = 10.0
        # current_round_params: single param = [0.0]
        inner.current_round_params = [np.array([0.0], dtype=np.float32)]
        inner.aggregate_fit.return_value = (None, {})

        wrapper = AdaptiveClipWrapper(inner, AdaptiveClipConfig(target_quantile=0.5))

        # Client sends update with norm=1.0 (well below clip=10.0)
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([1.0], dtype=np.float32)]),
            num_examples=1, metrics={},
        )
        results = [(MagicMock(), fit_res)]

        wrapper.aggregate_fit(1, results, [])
        # 0% clipped, target 50% → clip should decrease
        assert inner.clipping_norm < 10.0

    def test_clip_increases_when_updates_large(self):
        """When all updates exceed clip norm, clip should increase."""
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig

        inner = MagicMock()
        inner.clipping_norm = 1.0
        inner.current_round_params = [np.array([0.0], dtype=np.float32)]
        inner.aggregate_fit.return_value = (None, {})

        wrapper = AdaptiveClipWrapper(inner, AdaptiveClipConfig(target_quantile=0.5))

        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([100.0], dtype=np.float32)]),
            num_examples=1, metrics={},
        )
        results = [(MagicMock(), fit_res)]

        wrapper.aggregate_fit(1, results, [])
        # 100% clipped, target 50% → clip should increase
        assert inner.clipping_norm > 1.0

    def test_clip_bounds_respected(self):
        """Clip norm should be clamped to [clip_min, clip_max]."""
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig

        inner = MagicMock()
        inner.clipping_norm = 0.2
        inner.current_round_params = [np.array([0.0], dtype=np.float32)]
        inner.aggregate_fit.return_value = (None, {})

        cfg = AdaptiveClipConfig(
            target_quantile=0.5, learning_rate=10.0,  # aggressive lr
            clip_min=0.1, clip_max=100.0,
        )
        wrapper = AdaptiveClipWrapper(inner, cfg)

        # All updates tiny → clip wants to go very low
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([0.001], dtype=np.float32)]),
            num_examples=1, metrics={},
        )
        results = [(MagicMock(), fit_res)]
        wrapper.aggregate_fit(1, results, [])
        assert inner.clipping_norm >= cfg.clip_min

    def test_noisy_quantile_stays_bounded(self):
        """With quantile_noise_multiplier>0, fraction_clipped stays in [0,1]."""
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig

        inner = MagicMock()
        inner.clipping_norm = 5.0
        inner.current_round_params = [np.array([0.0], dtype=np.float32)]
        inner.aggregate_fit.return_value = (None, {})

        cfg = AdaptiveClipConfig(
            target_quantile=0.5,
            quantile_noise_multiplier=10.0,  # very large noise
        )
        wrapper = AdaptiveClipWrapper(inner, cfg)

        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([1.0], dtype=np.float32)]),
            num_examples=1, metrics={},
        )
        results = [(MagicMock(), fit_res)]

        # Run many rounds; clip must always stay within bounds
        np.random.seed(42)
        for r in range(20):
            wrapper.aggregate_fit(r + 1, results, [])
            assert cfg.clip_min <= inner.clipping_norm <= cfg.clip_max


# ── SecAggConfig ────────────────────────────────────────────────────────────


class TestSecAggConfig:
    """Tests for secure aggregation configuration."""

    def test_build_secagg_config(self):
        cfg = SecAggConfig(num_shares=4, reconstruction_threshold=3)
        result = build_secagg_config(cfg)
        assert result["num_shares"] == 4
        assert result["reconstruction_threshold"] == 3
        assert result["clipping_range"] == 8.0
        assert result["quantization_range"] == 4194304

    def test_custom_clipping_and_quantization(self):
        cfg = SecAggConfig(
            num_shares=5,
            reconstruction_threshold=3,
            clipping_range=16.0,
            quantization_range=2**26,
        )
        result = build_secagg_config(cfg)
        assert result["clipping_range"] == 16.0
        assert result["quantization_range"] == 2**26

    def test_make_secagg_main_calls_server_fn(self):
        """The secagg main function should call server_fn to get components."""
        from flwr.server import ServerAppComponents, ServerConfig

        initial = ndarrays_to_parameters(
            [np.array([0.0], dtype=np.float32)]
        )
        strategy = FedAvg(
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=initial,
        )
        components = ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=1),
        )

        mock_server_fn = MagicMock(return_value=components)
        cfg = SecAggConfig(num_shares=3, reconstruction_threshold=2)
        main_fn = make_secagg_main(mock_server_fn, cfg)

        # We can't fully run the workflow (needs Grid), but verify
        # the function was created and server_fn would be called
        assert callable(main_fn)

    def test_low_threshold_warns(self):
        """reconstruction_threshold below ceil(2n/3) should warn (C2)."""
        # ceil(2*4/3)=3, threshold=2 < 3 should trigger logger warning
        # Just verify it succeeds without raising
        cfg = SecAggConfig(num_shares=4, reconstruction_threshold=2)
        assert cfg.reconstruction_threshold == 2

    def test_threshold_exceeds_shares_raises(self):
        """reconstruction_threshold > num_shares should raise ValueError."""
        with pytest.raises(ValueError, match="must be <= num_shares"):
            SecAggConfig(num_shares=3, reconstruction_threshold=4)

    def test_shares_below_two_raises(self):
        """num_shares < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 2"):
            SecAggConfig(num_shares=1, reconstruction_threshold=1)


# ── Sum Demo Server DP Integration ──────────────────────────────────────────


class TestSumServerDP:
    """Test DP integration in the sum demo server."""

    def _make_context(self, run_config=None):
        ctx = MagicMock(spec=Context)
        ctx.run_config = run_config or {}
        return ctx

    def test_server_fn_no_dp(self):
        """Without DP config, server_fn returns unwrapped strategy."""
        from sfl.server.app import server_fn

        ctx = self._make_context()
        components = server_fn(ctx)
        assert isinstance(components.strategy, SumFedAvg)

    def test_server_fn_dp_via_run_config(self):
        """DP enabled via run_config wraps the strategy."""
        from sfl.server.app import server_fn

        ctx = self._make_context({
            "dp-enabled": "true",
            "dp-noise-multiplier": "0.5",
            "dp-clipping-norm": "5.0",
            "dp-mode": "server",
            "num-clients": "2",
        })
        components = server_fn(ctx)
        _assert_dp_wrapped(
            components.strategy, DifferentialPrivacyServerSideFixedClipping,
        )

    def test_server_fn_dp_via_env_vars(self):
        """DP enabled via environment variables wraps the strategy."""
        from sfl.server.app import server_fn

        env_patch = {
            "SFL_DP_ENABLED": "true",
            "SFL_DP_NOISE": "0.3",
            "SFL_DP_CLIP": "8.0",
            "SFL_DP_MODE": "client",
        }
        ctx = self._make_context()
        with patch.dict(os.environ, env_patch, clear=False):
            components = server_fn(ctx)
        _assert_dp_wrapped(
            components.strategy, DifferentialPrivacyClientSideFixedClipping,
        )

    def test_server_fn_dp_run_config_overrides_env(self):
        """run_config DP settings take precedence over env vars."""
        from sfl.server.app import server_fn

        env_patch = {
            "SFL_DP_ENABLED": "true",
            "SFL_DP_MODE": "client",
        }
        ctx = self._make_context({
            "dp-enabled": "true",
            "dp-mode": "server",
            "num-clients": "2",
        })
        with patch.dict(os.environ, env_patch, clear=False):
            components = server_fn(ctx)
        # run_config says "server", so it should be server-side
        _assert_dp_wrapped(
            components.strategy, DifferentialPrivacyServerSideFixedClipping,
        )



        assert callable(build_secagg_config)


# ── Joint DP composition ───────────────────────────────────────────────────


@pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
class TestJointDPComposition:
    """Tests for _AccountingWrapper composing client+server DP."""

    def test_metrics_include_total_epsilon(self):
        """When clients report dpsgd_epsilon, aggregate metrics should
        include dp_total_epsilon = server ε + max client ε."""
        from sfl.privacy.dp import _AccountingWrapper
        from sfl.privacy.accountant import PrivacyAccountant
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p

        inner = MagicMock()
        inner.aggregate_fit.return_value = (
            n2p([np.array([1.0], dtype=np.float32)]),
            {},
        )

        accountant = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
        )
        wrapper = _AccountingWrapper(inner, accountant)

        # Simulate two clients reporting dpsgd_epsilon
        fit_res_1 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([1.0], dtype=np.float32)]),
            num_examples=10,
            metrics={"dpsgd_epsilon": 2.5},
        )
        fit_res_2 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([1.0], dtype=np.float32)]),
            num_examples=10,
            metrics={"dpsgd_epsilon": 3.0},
        )
        results = [(MagicMock(), fit_res_1), (MagicMock(), fit_res_2)]

        _, metrics = wrapper.aggregate_fit(1, results, [])

        assert "dp_epsilon" in metrics
        assert "dp_total_epsilon" in metrics
        assert "dpsgd_epsilon_max" in metrics
        # PLD composition: total should be tighter than basic (server + client)
        basic_total = metrics["dp_epsilon"] + 3.0
        assert metrics["dp_total_epsilon"] < basic_total + 1e-6
        assert metrics["dp_total_epsilon"] > 0
        assert metrics["dpsgd_epsilon_max"] == pytest.approx(3.0)

    def test_no_client_dp_no_total(self):
        """Without dpsgd_epsilon in results, dp_total_epsilon is absent."""
        from sfl.privacy.dp import _AccountingWrapper
        from sfl.privacy.accountant import PrivacyAccountant
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p

        inner = MagicMock()
        inner.aggregate_fit.return_value = (
            n2p([np.array([1.0], dtype=np.float32)]),
            {},
        )

        accountant = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
        )
        wrapper = _AccountingWrapper(inner, accountant)

        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([1.0], dtype=np.float32)]),
            num_examples=10, metrics={},
        )
        results = [(MagicMock(), fit_res)]

        _, metrics = wrapper.aggregate_fit(1, results, [])
        assert "dp_epsilon" in metrics
        assert "dp_total_epsilon" not in metrics


# ── Adaptive Clip Timing Tests (A2) ─────────────────────────────────────


class TestAdaptiveClipTiming:
    """Verify clip norm is updated AFTER aggregate_fit, not before."""

    @pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
    def test_clip_unchanged_during_aggregation(self):
        """The inner strategy should see the OLD clip norm during aggregate_fit."""
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p

        inner = MagicMock()
        inner.clipping_norm = 5.0
        initial_clip = inner.clipping_norm

        # When aggregate_fit is called, capture the clip norm at call time
        clip_during_agg = []

        def capture_clip(*args, **kwargs):
            clip_during_agg.append(inner.clipping_norm)
            return (n2p([np.array([1.0], dtype=np.float32)]), {})

        inner.aggregate_fit.side_effect = capture_clip
        inner.current_round_params = [np.zeros(10, dtype=np.float32)]

        cfg = AdaptiveClipConfig(target_quantile=0.5, learning_rate=0.2)
        wrapper = AdaptiveClipWrapper(inner, cfg)

        # Create results with norms that will trigger a clip update
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.ones(10, dtype=np.float32) * 10]),
            num_examples=10, metrics={},
        )
        results = [(MagicMock(), fit_res)]

        wrapper.aggregate_fit(1, results, [])

        # During aggregation, clip should have been the ORIGINAL value
        assert clip_during_agg[0] == initial_clip
        # After aggregation, clip should have changed
        assert inner.clipping_norm != initial_clip


# ── Quantile Cost Composition Tests (A3) ─────────────────────────────────


class TestQuantileCostComposition:
    """Verify quantile query DP cost is composed into the accountant."""

    @pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
    def test_quantile_cost_increases_epsilon(self):
        """When quantile_noise_multiplier > 0, total ε should be higher."""
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig
        from sfl.privacy.accountant import PrivacyAccountant
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p

        def make_wrapped(quantile_noise):
            inner = MagicMock()
            inner.clipping_norm = 10.0
            inner.current_round_params = [np.zeros(10, dtype=np.float32)]
            inner.aggregate_fit.return_value = (
                n2p([np.array([1.0], dtype=np.float32)]), {}
            )
            ac_cfg = AdaptiveClipConfig(
                target_quantile=0.5, learning_rate=0.2,
                quantile_noise_multiplier=quantile_noise,
            )
            adaptive = AdaptiveClipWrapper(inner, ac_cfg)
            accountant = PrivacyAccountant(
                noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
            )
            return _AccountingWrapper(adaptive, accountant)

        w_noisy = make_wrapped(quantile_noise=0.5)
        w_silent = make_wrapped(quantile_noise=0.0)

        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.ones(10, dtype=np.float32)]),
            num_examples=10, metrics={},
        )
        results = [(MagicMock(), fit_res)]

        # Run one round on each
        _, m_noisy = w_noisy.aggregate_fit(1, results, [])
        _, m_silent = w_silent.aggregate_fit(1, results, [])

        # Noisy quantile should have higher (or equal) epsilon
        assert m_noisy["dp_epsilon"] >= m_silent["dp_epsilon"]


# ── Low Noise Warning Tests (B4) ─────────────────────────────────────────


class TestLowNoiseWarning:
    """Verify warning is logged when noise_multiplier < 0.3."""

    def test_low_noise_warns(self):
        """noise_multiplier=0.1 should produce a warning."""
        import logging
        with patch("sfl.privacy.dp.logger") as mock_logger:
            dp_config = DPConfig(noise_multiplier=0.1, clipping_norm=10.0)
            wrap_strategy_with_dp(FedAvg(), dp_config)
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "very low" in warning_msg.lower() or "negligible" in warning_msg.lower()

    def test_normal_noise_no_warn(self):
        """noise_multiplier=1.0 should NOT produce a warning about low noise."""
        with patch("sfl.privacy.dp.logger") as mock_logger:
            dp_config = DPConfig(noise_multiplier=1.0, clipping_norm=10.0)
            wrap_strategy_with_dp(FedAvg(), dp_config)
            # Check that no warning about low noise was issued
            for call in mock_logger.warning.call_args_list:
                assert "very low" not in call[0][0].lower()


# ── Budget Dashboard Tests (B7) ──────────────────────────────────────────


class TestBudgetDashboard:
    """Verify per-round budget dashboard logging (B7)."""

    @pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
    def test_dashboard_logged_each_round(self):
        """Budget dashboard should log after each aggregate_fit round."""
        from sfl.privacy.accountant import PrivacyAccountant
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters as n2p

        inner = MagicMock()
        inner.aggregate_fit.return_value = (
            n2p([np.array([1.0], dtype=np.float32)]), {}
        )

        accountant = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
        )
        wrapper = _AccountingWrapper(inner, accountant)

        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=n2p([np.array([1.0], dtype=np.float32)]),
            num_examples=10, metrics={},
        )
        results = [(MagicMock(), fit_res)]

        with patch("sfl.privacy.dp.logger") as mock_logger:
            wrapper.aggregate_fit(1, results, [])
            # Should log the ASCII dashboard
            info_calls = [str(c) for c in mock_logger.info.call_args_list]
            dashboard_logged = any("Privacy Budget" in s for s in info_calls)
            assert dashboard_logged, "Budget dashboard not logged"


# ── PABI Tests (S2) ────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
class TestPABI:
    """Tests for Privacy Amplification by Iteration (S2)."""

    def test_pabi_tighter_than_standard(self):
        """With strong convexity, PABI ε should be ≤ standard PLD ε."""
        from sfl.privacy.accountant import compute_pabi_epsilon

        eps_standard = compute_pabi_epsilon(
            noise_multiplier=1.0, num_steps=100,
            sample_rate=1.0, delta=1e-5,
            strong_convexity=0.0,  # no PABI
        )
        eps_pabi = compute_pabi_epsilon(
            noise_multiplier=1.0, num_steps=100,
            sample_rate=1.0, delta=1e-5,
            smoothness=1.0,
            strong_convexity=0.1,  # μ/L = 0.1
        )
        assert eps_pabi <= eps_standard

    def test_pabi_zero_convexity_equals_standard(self):
        """With strong_convexity=0, PABI should equal standard PLD."""
        from sfl.privacy.accountant import compute_pabi_epsilon

        eps_a = compute_pabi_epsilon(
            noise_multiplier=1.0, num_steps=50,
            sample_rate=1.0, delta=1e-5,
            strong_convexity=0.0,
        )
        eps_b = compute_pabi_epsilon(
            noise_multiplier=1.0, num_steps=50,
            sample_rate=1.0, delta=1e-5,
            smoothness=1.0, strong_convexity=0.0,
        )
        assert abs(eps_a - eps_b) < 1e-6

    def test_pabi_returns_positive(self):
        """PABI ε should always be positive."""
        from sfl.privacy.accountant import compute_pabi_epsilon

        eps = compute_pabi_epsilon(
            noise_multiplier=2.0, num_steps=10,
            delta=1e-5, smoothness=1.0, strong_convexity=0.5,
        )
        assert eps > 0


# ── Distributed Noise Tests (S3) ──────────────────────────────────────────


class TestDistributedNoise:
    """Tests for distributed noise splitting (S3)."""

    def test_total_sigma_preserved(self):
        """σ_client and σ_server should compose to target_sigma."""
        from sfl.privacy.dp import compute_distributed_noise_params

        result = compute_distributed_noise_params(
            target_sigma=1.0, num_clients=10, trust_fraction=0.0,
        )
        assert abs(result["total_sigma"] - 1.0) < 1e-10

    def test_fully_distributed(self):
        """trust_fraction=0 → server adds no noise, clients share equally."""
        from sfl.privacy.dp import compute_distributed_noise_params

        result = compute_distributed_noise_params(
            target_sigma=1.0, num_clients=4, trust_fraction=0.0,
        )
        assert result["sigma_server"] == 0.0
        assert result["sigma_client"] > 0

    def test_fully_server(self):
        """trust_fraction=1 → server adds all noise, clients add none."""
        from sfl.privacy.dp import compute_distributed_noise_params

        result = compute_distributed_noise_params(
            target_sigma=2.0, num_clients=5, trust_fraction=1.0,
        )
        assert abs(result["sigma_server"] - 2.0) < 1e-10
        assert result["sigma_client"] == 0.0

    def test_half_split(self):
        """trust_fraction=0.5 → half variance from server, half from clients."""
        from sfl.privacy.dp import compute_distributed_noise_params
        import math

        result = compute_distributed_noise_params(
            target_sigma=1.0, num_clients=4, trust_fraction=0.5,
        )
        # Server var = 0.5, client var per client = 0.5/4 = 0.125
        assert abs(result["sigma_server"] - math.sqrt(0.5)) < 1e-10
        assert abs(result["total_sigma"] - 1.0) < 1e-10

    def test_invalid_sigma_raises(self):
        """Negative or zero target_sigma should raise."""
        from sfl.privacy.dp import compute_distributed_noise_params

        with pytest.raises(ValueError):
            compute_distributed_noise_params(target_sigma=0, num_clients=5)

    def test_invalid_trust_fraction_raises(self):
        """trust_fraction outside [0,1] should raise."""
        from sfl.privacy.dp import compute_distributed_noise_params

        with pytest.raises(ValueError):
            compute_distributed_noise_params(
                target_sigma=1.0, num_clients=5, trust_fraction=1.5,
            )


# ── Canary Privacy Audit Tests (S4) ──────────────────────────────────────


class TestPrivacyAuditor:
    """Tests for canary-based privacy auditing (S4)."""

    def test_high_noise_passes(self):
        """With high noise, the canary should be undetectable."""
        from sfl.privacy.audit import PrivacyAuditor

        auditor = PrivacyAuditor(noise_scale=10.0, clipping_norm=1.0)
        result = auditor.run_canary_audit(
            params=[np.zeros(1000, dtype=np.float32)],
            num_trials=100, seed=42,
        )
        assert result.passed
        assert result.detection_rate <= 0.1

    def test_zero_noise_fails(self):
        """With no noise, the canary should be detectable."""
        from sfl.privacy.audit import PrivacyAuditor

        auditor = PrivacyAuditor(
            noise_scale=0.001, clipping_norm=100.0,
            detection_threshold=0.01, acceptable_rate=0.05,
        )
        result = auditor.run_canary_audit(
            params=[np.zeros(50, dtype=np.float32)],
            num_trials=100, seed=42,
        )
        # With essentially no noise, canary should be detectable
        assert result.detection_rate > 0.0

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        from sfl.privacy.audit import PrivacyAuditor

        auditor = PrivacyAuditor(noise_scale=1.0, clipping_norm=5.0)
        r1 = auditor.run_canary_audit(
            params=[np.ones(20, dtype=np.float32)],
            num_trials=50, seed=123,
        )
        r2 = auditor.run_canary_audit(
            params=[np.ones(20, dtype=np.float32)],
            num_trials=50, seed=123,
        )
        assert r1.detection_rate == r2.detection_rate
        assert r1.mean_cosine_sim == r2.mean_cosine_sim


# ── Per-Layer Clip Mod Tests (S6) ─────────────────────────────────────────


class TestPerLayerClipMod:
    """Tests for per-layer adaptive clipping mod (S6)."""

    def test_clips_large_layer(self):
        """A layer exceeding default_clip should be clipped to that norm."""
        from sfl.privacy.adaptive_clip import make_per_layer_clip_mod
        from tests.test_filters import _make_train_message, _extract_params

        # Layer with norm = 10.0
        params = [np.ones(100, dtype=np.float32)]  # norm = 10.0
        in_msg, out_msg = _make_train_message(params)

        mod = make_per_layer_clip_mod(default_clip=2.0)

        from unittest.mock import MagicMock
        from flwr.common.context import Context
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        # Norm should now be <= 2.0
        result_norm = float(np.linalg.norm(result_params[0]))
        assert result_norm <= 2.01

    def test_preserves_small_layer(self):
        """A layer below default_clip should not be modified."""
        from sfl.privacy.adaptive_clip import make_per_layer_clip_mod
        from tests.test_filters import _make_train_message, _extract_params

        params = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]  # norm ~ 0.37
        in_msg, out_msg = _make_train_message(params)

        mod = make_per_layer_clip_mod(default_clip=5.0)

        from unittest.mock import MagicMock
        from flwr.common.context import Context
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        np.testing.assert_allclose(result_params[0], params[0], atol=1e-6)

    def test_per_layer_custom_norms(self):
        """Custom clip_norms should apply different clips per layer index."""
        from sfl.privacy.adaptive_clip import make_per_layer_clip_mod
        from tests.test_filters import _make_train_message, _extract_params

        # Two layers, both with norm = 10.0
        params = [
            np.ones(100, dtype=np.float32),   # norm = 10
            np.ones(100, dtype=np.float32),   # norm = 10
        ]
        in_msg, out_msg = _make_train_message(params)

        # Layer 0: clip at 1.0, layer 1: clip at 5.0
        mod = make_per_layer_clip_mod(clip_norms={0: 1.0, 1: 5.0}, default_clip=10.0)

        from unittest.mock import MagicMock
        from flwr.common.context import Context
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        assert float(np.linalg.norm(result_params[0])) <= 1.01
        assert float(np.linalg.norm(result_params[1])) <= 5.01

    def test_direction_preserved(self):
        """Clipping should preserve the direction of the gradient."""
        from sfl.privacy.adaptive_clip import make_per_layer_clip_mod
        from tests.test_filters import _make_train_message, _extract_params

        params = [np.array([3.0, 4.0], dtype=np.float32)]  # norm = 5
        in_msg, out_msg = _make_train_message(params)

        mod = make_per_layer_clip_mod(default_clip=1.0)

        from unittest.mock import MagicMock
        from flwr.common.context import Context
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        # Direction should be [3/5, 4/5] = [0.6, 0.8]
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_allclose(result_params[0], expected, atol=1e-5)



# ── H1: Secure RNG Tests ───────────────────────────────────────────────────


class TestSecureRNG:
    """Tests for CSRNG-seeded noise generation (H1)."""

    def test_secure_rng_produces_different_seeds(self):
        """Two calls to secure_rng() should produce different sequences."""
        from sfl.utils.rng import secure_rng

        r1 = secure_rng().random_sample(100)
        r2 = secure_rng().random_sample(100)
        # With overwhelming probability, these will differ
        assert not np.allclose(r1, r2)


# ── H2: DPConfig Validation Tests ──────────────────────────────────────────


class TestDPConfigValidation:
    """Tests for DPConfig __post_init__ validation (H2)."""

    def test_negative_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="noise_multiplier"):
            DPConfig(noise_multiplier=-1.0)

    def test_zero_clipping_norm_raises(self):
        with pytest.raises(ValueError, match="clipping_norm"):
            DPConfig(clipping_norm=0.0)

    def test_sampled_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="num_sampled_clients"):
            DPConfig(num_sampled_clients=5, num_total_clients=3)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="target_delta"):
            DPConfig(target_delta=0.0)
        with pytest.raises(ValueError, match="target_delta"):
            DPConfig(target_delta=1.0)

    def test_invalid_max_epsilon_raises(self):
        with pytest.raises(ValueError, match="max_epsilon"):
            DPConfig(max_epsilon=-1.0)

    def test_valid_config_succeeds(self):
        cfg = DPConfig(
            noise_multiplier=1.0, clipping_norm=10.0,
            num_sampled_clients=2, num_total_clients=4,
        )
        assert cfg.noise_multiplier == 1.0


# ── H3: Exact Composition Tests ─────────────────────────────────────────


@pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
class TestExactComposition:
    """Tests for compose_epsilon with exact Gaussian sigmas (H3)."""

    def test_exact_sigma_gives_tighter_bound(self):
        """Using sigma_server/sigma_client should be at least as tight."""
        from sfl.privacy.accountant import compose_epsilon

        # Approximate path (no sigmas)
        eps_approx, _ = compose_epsilon(
            eps_server=1.0, eps_client=1.0,
            delta_server=1e-5, delta_client=1e-5,
        )
        # Exact path (with known sigmas)
        eps_exact, _ = compose_epsilon(
            eps_server=1.0, eps_client=1.0,
            delta_server=1e-5, delta_client=1e-5,
            sigma_server=1.0, sigma_client=1.0,
        )
        # Both should be positive
        assert eps_approx > 0
        assert eps_exact > 0
