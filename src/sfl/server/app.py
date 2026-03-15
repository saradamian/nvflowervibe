"""
Federated learning server application.

This module provides the server-side application for Flower,
configured with the SumFedAvg strategy for federated sum aggregation.
"""

import numpy as np
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from sfl.server.strategy import SumFedAvg
from sfl.utils.logging import get_logger
from sfl.utils.config import get_config

logger = get_logger(__name__)


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the federated learning server.
    
    This function is called by Flower's ServerApp to initialize
    the server with the appropriate strategy and configuration.
    
    Configuration sources (priority order):
    1. context.run_config (from Flower/NVFlare)
    2. SFL configuration (from YAML/env)
    3. Defaults
    
    Args:
        context: Flower context containing run_config.
    
    Returns:
        ServerAppComponents with strategy and config.
    
    Example:
        >>> from flwr.common import Context
        >>> context = Context(node_id=0, run_config={})
        >>> components = server_fn(context)
        >>> print(type(components.strategy))
        <class 'sfl.server.strategy.SumFedAvg'>
    """
    logger.info("Initializing federated server")
    
    # Get configuration from multiple sources
    run_config = context.run_config or {}
    
    try:
        sfl_config = get_config()
        default_rounds = sfl_config.federation.num_rounds
        default_clients = sfl_config.federation.num_clients
        default_min_clients = sfl_config.federation.min_fit_clients
        initial_param = sfl_config.server.initial_param
    except Exception:
        # Fall back to defaults if config not available
        default_rounds = 1
        default_clients = 2
        default_min_clients = 2
        initial_param = 0.0
    
    # Get values from run_config or use SFL config defaults
    num_rounds = int(run_config.get("num-server-rounds", default_rounds))
    num_clients = int(run_config.get("num-clients", default_clients))
    min_fit_clients = int(run_config.get("min-fit-clients", default_min_clients))
    
    logger.info(
        f"Server config: rounds={num_rounds}, "
        f"clients={num_clients}, min_fit={min_fit_clients}"
    )
    
    # Create initial parameters
    initial_params = ndarrays_to_parameters(
        [np.array([initial_param], dtype=np.float32)]
    )
    
    # Create strategy based on aggregation setting
    import os
    aggregation = os.environ.get("SFL_AGGREGATION", "fedavg").lower()
    strategy_kwargs = dict(
        min_fit_clients=min_fit_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        log_client_values=True,
    )

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
            allow_untrusted_reference=True,  # TODO: wire --ffl-root-data CLI flag
            **strategy_kwargs,
        )
    else:
        strategy = SumFedAvg(**strategy_kwargs)

    # Wrap with DP if configured (check run_config or env vars)
    dp_enabled = (
        str(run_config.get("dp-enabled", "false")).lower() == "true"
        or os.environ.get("SFL_DP_ENABLED", "").lower() == "true"
    )
    if dp_enabled:
        from sfl.privacy.dp import DPConfig, wrap_strategy_with_dp

        dp_config = DPConfig(
            noise_multiplier=float(
                run_config.get("dp-noise-multiplier",
                               os.environ.get("SFL_DP_NOISE", "1.0"))
            ),
            clipping_norm=float(
                run_config.get("dp-clipping-norm",
                               os.environ.get("SFL_DP_CLIP", "10.0"))
            ),
            num_sampled_clients=num_clients,
            mode=str(
                run_config.get("dp-mode",
                               os.environ.get("SFL_DP_MODE", "server"))
            ),
            target_delta=float(os.environ.get("SFL_DP_DELTA", "1e-5")),
            max_epsilon=float(os.environ.get("SFL_DP_MAX_EPSILON", "10.0")),
            num_total_clients=num_clients,
            adaptive_clipping=(
                os.environ.get("SFL_DP_ADAPTIVE_CLIP", "").lower() == "true"
            ),
            target_quantile=float(
                os.environ.get("SFL_DP_TARGET_QUANTILE", "0.5")
            ),
            clip_learning_rate=float(
                os.environ.get("SFL_DP_CLIP_LR", "0.2")
            ),
            quantile_noise_multiplier=float(
                os.environ.get("SFL_DP_QUANTILE_NOISE", "0.0")
            ),
            accounting_backend=os.environ.get("SFL_DP_ACCOUNTING_BACKEND", "pld"),
            shuffle_model=(
                os.environ.get("SFL_DP_SHUFFLE", "").lower() == "true"
            ),
        )
        strategy = wrap_strategy_with_dp(strategy, dp_config)
    
    # Create server config
    config = ServerConfig(num_rounds=num_rounds)
    
    logger.info("Server initialized successfully")
    
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


# Create the Flower ServerApp
app = ServerApp(server_fn=server_fn)
