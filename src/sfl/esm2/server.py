"""
ESM2 federated learning server.

Configures a Flower server with the appropriate aggregation strategy
for federated ESM2 model training, including robust aggregation,
DP, checkpointing, and metrics.
"""

import os

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig as FlowerServerConfig

from sfl.esm2.model import DEFAULT_MODEL_NAME, get_parameters, load_model
from sfl.utils.logging import get_logger

logger = get_logger(__name__)


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the ESM2 federated learning server.

    Initializes the aggregation strategy (FedAvg, Krum, TrimmedMean,
    or FoundationFL) seeded with pretrained ESM2 weights, then wraps
    with DP and operational layers as configured.

    Config resolution order: run_config (Flower context) → SFL_* env vars
    → module-level config defaults. This ensures HPC distributed mode
    (where run_config may be empty) still picks up SLURM-exported vars.

    Args:
        context: Flower context with run_config.

    Returns:
        ServerAppComponents with strategy and config.
    """
    from sfl.esm2.config import get_run_config
    from sfl.server.dp_setup import build_strategy

    cfg = get_run_config()
    run_config = context.run_config or {}

    def _cfg_val(rc_key: str, env_key: str, default):
        """Resolve config: run_config → env var → default."""
        val = run_config.get(rc_key)
        if val is not None:
            return val
        val = os.environ.get(env_key)
        if val is not None:
            return val
        return default

    num_rounds = int(_cfg_val("num-server-rounds", "SFL_NUM_ROUNDS", cfg.num_rounds))
    num_clients = int(_cfg_val("num-clients", "SFL_NUM_CLIENTS", cfg.num_clients))
    min_fit_clients = int(_cfg_val("min-fit-clients", "SFL_MIN_FIT_CLIENTS", num_clients))
    model_name = str(_cfg_val("esm2-model", "SFL_MODEL", cfg.model_name))
    fraction_fit = float(_cfg_val("fraction-fit", "SFL_FRACTION_FIT", cfg.fraction_fit))
    fraction_evaluate = float(_cfg_val("fraction-evaluate", "SFL_FRACTION_EVALUATE", cfg.fraction_evaluate))

    logger.info(
        f"ESM2 server: model={model_name}, rounds={num_rounds}, "
        f"clients={num_clients}, min_fit={min_fit_clients}"
    )

    # Load pretrained model to get initial parameters
    model = load_model(model_name)
    initial_params = ndarrays_to_parameters(get_parameters(model))
    del model  # free memory — clients will load their own copies

    strategy = build_strategy(
        initial_parameters=initial_params,
        num_clients=num_clients,
        run_config=run_config,
        min_fit_clients=min_fit_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
    )

    # Adjust rounds for resume — build_strategy sets SFL_RESUME_ROUND
    # when restoring from checkpoint
    resume_round = int(os.environ.get("SFL_RESUME_ROUND", "0"))
    remaining_rounds = max(1, num_rounds - resume_round)
    if resume_round > 0:
        logger.info(
            "Resuming: completed %d rounds, running %d remaining (of %d total)",
            resume_round, remaining_rounds, num_rounds,
        )
    config = FlowerServerConfig(num_rounds=remaining_rounds)

    logger.info("ESM2 server initialized")
    return ServerAppComponents(strategy=strategy, config=config)
