"""
LLM federated learning server.

Configures a Flower server with the appropriate aggregation strategy
for federated causal LM training, including robust aggregation,
DP, checkpointing, and metrics.

When LoRA is enabled, the server initializes with LoRA adapter weights
only, so the aggregation operates on the compact adapter parameter space.
"""

import os

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig as FlowerServerConfig

from sfl.llm.model import (
    DEFAULT_MODEL_NAME,
    apply_lora,
    get_lora_parameters,
    get_parameters,
    load_model,
)
from sfl.utils.logging import get_logger

logger = get_logger(__name__)


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the LLM federated learning server.

    Initializes the aggregation strategy (FedAvg, Krum, TrimmedMean,
    or FoundationFL) seeded with pretrained causal LM weights (or LoRA
    adapter weights), then wraps with DP and operational layers.

    Config resolution order: run_config (Flower context) → SFL_* env vars
    → module-level config defaults.

    Args:
        context: Flower context with run_config.

    Returns:
        ServerAppComponents with strategy and config.
    """
    from sfl.llm.config import get_run_config
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
    model_name = str(_cfg_val("llm-model", "SFL_MODEL", cfg.model_name))
    fraction_fit = float(_cfg_val("fraction-fit", "SFL_FRACTION_FIT", cfg.fraction_fit))
    fraction_evaluate = float(_cfg_val("fraction-evaluate", "SFL_FRACTION_EVALUATE", cfg.fraction_evaluate))
    use_lora = str(_cfg_val("use-lora", "SFL_USE_LORA", cfg.use_lora)).lower() in ("true", "1")
    lora_r = int(_cfg_val("lora-r", "SFL_LORA_R", cfg.lora_r))
    lora_alpha = int(_cfg_val("lora-alpha", "SFL_LORA_ALPHA", cfg.lora_alpha))

    logger.info(
        f"LLM server: model={model_name}, rounds={num_rounds}, "
        f"clients={num_clients}, min_fit={min_fit_clients}, lora={use_lora}"
    )

    # Load pretrained model to get initial parameters
    model = load_model(model_name)
    if use_lora:
        model = apply_lora(model, r=lora_r, alpha=lora_alpha)
        initial_params = ndarrays_to_parameters(get_lora_parameters(model))
    else:
        initial_params = ndarrays_to_parameters(get_parameters(model))
    del model  # free memory -- clients will load their own copies

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

    logger.info("LLM server initialized")
    return ServerAppComponents(strategy=strategy, config=config)
