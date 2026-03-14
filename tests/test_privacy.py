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
from sfl.privacy.secagg import SecAggConfig, build_secagg_config
from sfl.server.strategy import SumFedAvg


# ── DPConfig ────────────────────────────────────────────────────────────────


class TestDPConfig:
    """Tests for DPConfig dataclass."""

    def test_defaults(self):
        cfg = DPConfig()
        assert cfg.noise_multiplier == 0.1
        assert cfg.clipping_norm == 10.0
        assert cfg.num_sampled_clients == 2
        assert cfg.mode == "server"

    def test_custom_values(self):
        cfg = DPConfig(
            noise_multiplier=1.0,
            clipping_norm=5.0,
            num_sampled_clients=8,
            mode="client",
        )
        assert cfg.noise_multiplier == 1.0
        assert cfg.clipping_norm == 5.0
        assert cfg.num_sampled_clients == 8
        assert cfg.mode == "client"


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


# ── SecAggConfig ────────────────────────────────────────────────────────────


class TestSecAggConfig:
    """Tests for secure aggregation configuration."""

    def test_defaults(self):
        cfg = SecAggConfig()
        assert cfg.num_shares == 3
        assert cfg.reconstruction_threshold == 2
        assert cfg.clipping_range == 8.0
        assert cfg.quantization_range == 4194304

    def test_custom_values(self):
        cfg = SecAggConfig(
            num_shares=5,
            reconstruction_threshold=3,
            clipping_range=16.0,
            quantization_range=1048576,
        )
        assert cfg.num_shares == 5
        assert cfg.reconstruction_threshold == 3

    def test_build_secagg_config(self):
        cfg = SecAggConfig(num_shares=4, reconstruction_threshold=3)
        result = build_secagg_config(cfg)
        assert result["num_shares"] == 4
        assert result["reconstruction_threshold"] == 3
        assert result["clipping_range"] == 8.0
        assert result["quantization_range"] == 4194304


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


# ── Module exports ──────────────────────────────────────────────────────────


class TestPrivacyExports:
    """Test that the privacy __init__ exports are correct."""

    def test_dp_exports(self):
        from sfl.privacy import DPConfig, wrap_strategy_with_dp
        assert DPConfig is not None
        assert callable(wrap_strategy_with_dp)

    def test_secagg_exports(self):
        from sfl.privacy import SecAggConfig, build_secagg_config
        assert SecAggConfig is not None
        assert callable(build_secagg_config)
