"""
Shared server-side strategy setup.

Builds the aggregation strategy (FedAvg, Krum, TrimmedMean,
FoundationFL) from SFL_* env vars, then wraps with DP and
checkpoint/metrics if configured.

Used by all server_fn implementations (ESM2, LLM, base) to avoid
duplicating strategy selection, DP config, and operational wrappers.
"""

import os
from typing import Any, Dict, Optional

from flwr.common import Parameters
from flwr.server.strategy import FedAvg, Strategy

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


def build_strategy(
    *,
    initial_parameters: Parameters,
    num_clients: int,
    run_config: Dict[str, Any],
    min_fit_clients: Optional[int] = None,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    default_strategy_class: Optional[type] = None,
) -> Strategy:
    """Build a fully-configured strategy: aggregation + DP + checkpoint/metrics.

    Reads ``SFL_AGGREGATION`` env var (set by runners via
    ``build_privacy_mods``) to choose the aggregation strategy, then
    layers on DP wrapping and operational wrappers (checkpoint, metrics).

    Args:
        initial_parameters: Pre-trained weights for the strategy.
        num_clients: Total number of FL clients.
        run_config: Flower run_config dict from context.
        min_fit_clients: Minimum clients per round (defaults to num_clients).
        fraction_fit: Fraction of clients sampled for fit.
        fraction_evaluate: Fraction of clients sampled for evaluate.
        extra_kwargs: Additional kwargs passed to the strategy constructor.
        default_strategy_class: Class to use when aggregation is "fedavg".
            Defaults to ``FedAvg``. Pass a subclass (e.g. ``SumFedAvg``)
            to customize the default aggregation behavior.

    Returns:
        Fully wrapped strategy ready for ServerAppComponents.
    """
    if min_fit_clients is None:
        min_fit_clients = num_clients

    strategy_kwargs: Dict[str, Any] = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )
    if extra_kwargs:
        strategy_kwargs.update(extra_kwargs)

    aggregation = os.environ.get("SFL_AGGREGATION", "fedavg").lower()

    if aggregation == "krum":
        from sfl.server.robust import MultiKrumFedAvg
        strategy = MultiKrumFedAvg(
            num_byzantine=int(os.environ.get("SFL_KRUM_BYZANTINE", "1")),
            **strategy_kwargs,
        )
    elif aggregation == "trimmed-mean":
        from sfl.server.robust import TrimmedMeanFedAvg
        strategy = TrimmedMeanFedAvg(
            trim_ratio=float(os.environ.get("SFL_TRIM_RATIO", "0.1")),
            **strategy_kwargs,
        )
    elif aggregation == "foundation-fl":
        from sfl.server.robust import FoundationFLFedAvg
        strategy = FoundationFLFedAvg(
            trust_threshold=float(os.environ.get("SFL_FFL_THRESHOLD", "0.1")),
            weighted=os.environ.get("SFL_FFL_WEIGHTED", "true").lower() == "true",
            allow_untrusted_reference=os.environ.get(
                "SFL_FFL_ALLOW_UNTRUSTED", "false"
            ).lower() == "true",
            **strategy_kwargs,
        )
    else:
        cls = default_strategy_class or FedAvg
        strategy = cls(**strategy_kwargs)

    logger.info("Aggregation strategy: %s", aggregation)

    # Wrap with DP if configured
    strategy = apply_dp_if_enabled(strategy, run_config, num_clients)

    # Wrap with checkpoint/metrics if configured
    strategy = _apply_checkpoint_if_enabled(strategy)
    strategy = _apply_metrics_if_enabled(strategy)

    return strategy


def _apply_checkpoint_if_enabled(strategy: Strategy) -> Strategy:
    """Wrap strategy with checkpointing if SFL_CHECKPOINT_DIR is set."""
    checkpoint_dir = os.environ.get("SFL_CHECKPOINT_DIR")
    if not checkpoint_dir:
        return strategy

    from sfl.utils.checkpoint import CheckpointManager, make_checkpoint_strategy

    resume = os.environ.get("SFL_RESUME", "").lower() == "true"
    mgr = CheckpointManager(checkpoint_dir)

    if resume:
        latest = mgr.load_latest()
        if latest is not None:
            logger.info("Resuming from checkpoint round %d", latest["round"])

    logger.info("Checkpoint enabled: dir=%s, resume=%s", checkpoint_dir, resume)
    return make_checkpoint_strategy(strategy, mgr)


def _apply_metrics_if_enabled(strategy: Strategy) -> Strategy:
    """Wrap strategy with metrics collection if SFL_METRICS_DIR is set."""
    metrics_dir = os.environ.get("SFL_METRICS_DIR")
    if not metrics_dir:
        return strategy

    from sfl.utils.metrics import MetricsCollector, make_metrics_strategy

    fmt = os.environ.get("SFL_METRICS_FORMAT", "csv")
    collector = MetricsCollector(output_dir=metrics_dir, export_format=fmt)

    logger.info("Metrics collection enabled: dir=%s, format=%s", metrics_dir, fmt)
    return make_metrics_strategy(strategy, collector)



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
