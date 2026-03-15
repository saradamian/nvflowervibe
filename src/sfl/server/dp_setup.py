"""
Shared server-side DP configuration.

Reads DP settings from Flower run_config and SFL_* env vars,
builds a DPConfig, and wraps the strategy. Used by all server_fn
implementations (ESM2, LLM, base) to avoid duplicating the same
20-line config block.
"""

import os
from typing import Any, Dict

from flwr.server.strategy import Strategy

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


def apply_dp_if_enabled(
    strategy: Strategy,
    run_config: Dict[str, Any],
    num_clients: int,
) -> Strategy:
    """Wrap strategy with DP if enabled via run_config or env vars.

    Checks both ``run_config["dp-enabled"]`` and ``SFL_DP_ENABLED``
    env var. When enabled, builds a full DPConfig from all available
    sources (run_config takes priority over env vars) and wraps the
    strategy with accounting, adaptive clipping, and shuffle-model
    amplification as configured.

    Args:
        strategy: Base Flower strategy to wrap.
        run_config: Flower run_config dict from context.
        num_clients: Number of clients participating.

    Returns:
        DP-wrapped strategy if DP is enabled, otherwise the original.
    """
    dp_enabled = (
        str(run_config.get("dp-enabled", "false")).lower() == "true"
        or os.environ.get("SFL_DP_ENABLED", "").lower() == "true"
    )
    if not dp_enabled:
        return strategy

    from sfl.privacy.dp import DPConfig, wrap_strategy_with_dp

    def _get(key: str, env_key: str, default: str) -> str:
        return str(run_config.get(key, os.environ.get(env_key, default)))

    dp_config = DPConfig(
        noise_multiplier=float(_get("dp-noise-multiplier", "SFL_DP_NOISE", "1.0")),
        clipping_norm=float(_get("dp-clipping-norm", "SFL_DP_CLIP", "10.0")),
        num_sampled_clients=num_clients,
        mode=_get("dp-mode", "SFL_DP_MODE", "server"),
        target_delta=float(_get("dp-delta", "SFL_DP_DELTA", "1e-5")),
        max_epsilon=float(_get("dp-max-epsilon", "SFL_DP_MAX_EPSILON", "10.0")),
        num_total_clients=num_clients,
        adaptive_clipping=(
            _get("dp-adaptive-clip", "SFL_DP_ADAPTIVE_CLIP", "false").lower() == "true"
        ),
        target_quantile=float(
            _get("dp-target-quantile", "SFL_DP_TARGET_QUANTILE", "0.5")
        ),
        clip_learning_rate=float(
            _get("dp-clip-lr", "SFL_DP_CLIP_LR", "0.2")
        ),
        quantile_noise_multiplier=float(
            _get("dp-quantile-noise", "SFL_DP_QUANTILE_NOISE", "0.0")
        ),
        accounting_backend=_get(
            "dp-accounting-backend", "SFL_DP_ACCOUNTING_BACKEND", "pld"
        ),
        shuffle_model=(
            _get("dp-shuffle", "SFL_DP_SHUFFLE", "false").lower() == "true"
        ),
    )

    return wrap_strategy_with_dp(strategy, dp_config)
