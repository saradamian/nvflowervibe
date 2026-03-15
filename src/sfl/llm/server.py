"""
LLM federated learning server.

Configures a Flower server with FedAvg for aggregating causal LM
parameters across federated clients.

When LoRA is enabled, the server initializes with LoRA adapter weights
only, so the aggregation operates on the compact adapter parameter space.
"""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig as FlowerServerConfig
from flwr.server.strategy import FedAvg

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

    Initializes a FedAvg strategy seeded with the pretrained causal LM
    weights (or LoRA adapter weights) so all clients start from the
    same checkpoint.

    Args:
        context: Flower context with run_config.

    Returns:
        ServerAppComponents with strategy and config.
    """
    from sfl.llm.config import get_run_config

    cfg = get_run_config()
    run_config = context.run_config or {}

    num_rounds = int(run_config.get("num-server-rounds", cfg.num_rounds))
    num_clients = int(run_config.get("num-clients", cfg.num_clients))
    min_fit_clients = int(run_config.get("min-fit-clients", num_clients))
    model_name = str(run_config.get("llm-model", cfg.model_name))
    fraction_fit = float(run_config.get("fraction-fit", cfg.fraction_fit))
    fraction_evaluate = float(run_config.get("fraction-evaluate", cfg.fraction_evaluate))
    use_lora = str(run_config.get("use-lora", cfg.use_lora)).lower() in ("true", "1")
    lora_r = int(run_config.get("lora-r", cfg.lora_r))
    lora_alpha = int(run_config.get("lora-alpha", cfg.lora_alpha))

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

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
    )

    # Wrap with DP if configured (check run_config or env vars)
    from sfl.server.dp_setup import apply_dp_if_enabled
    strategy = apply_dp_if_enabled(strategy, run_config, num_clients)

    config = FlowerServerConfig(num_rounds=num_rounds)

    logger.info("LLM server initialized")
    return ServerAppComponents(strategy=strategy, config=config)
