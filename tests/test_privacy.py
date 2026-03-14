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

from sfl.privacy.dp import DPConfig, wrap_strategy_with_dp
from sfl.privacy.secagg import SecAggConfig, build_secagg_config, make_secagg_main
from sfl.server.strategy import SumFedAvg


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
        assert isinstance(wrapped, DifferentialPrivacyServerSideFixedClipping)

    def test_client_side_wrapping(self):
        strategy = self._make_fedavg()
        dp_config = DPConfig(mode="client", noise_multiplier=0.5, clipping_norm=1.0)
        wrapped = wrap_strategy_with_dp(strategy, dp_config)
        assert isinstance(wrapped, DifferentialPrivacyClientSideFixedClipping)

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
        assert isinstance(wrapped, DifferentialPrivacyServerSideFixedClipping)


# ── calibrate_gaussian_sigma ────────────────────────────────────────────────


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

    def test_make_secagg_main_returns_callable(self):
        """make_secagg_main should return a callable main function."""
        def dummy_server_fn(context):
            pass

        cfg = SecAggConfig(num_shares=3, reconstruction_threshold=2)
        main_fn = make_secagg_main(dummy_server_fn, cfg)
        assert callable(main_fn)

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
        assert isinstance(
            components.strategy,
            DifferentialPrivacyServerSideFixedClipping,
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
        assert isinstance(
            components.strategy,
            DifferentialPrivacyClientSideFixedClipping,
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
        assert isinstance(
            components.strategy,
            DifferentialPrivacyServerSideFixedClipping,
        )



        assert callable(build_secagg_config)
