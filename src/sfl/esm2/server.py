"""
ESM2 federated learning server.

Configures a Flower server with FedAvg for aggregating ESM2 model
parameters across federated clients.
"""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig as FlowerServerConfig
from flwr.server.strategy import FedAvg

from sfl.esm2.model import DEFAULT_MODEL_NAME, get_parameters, load_model
from sfl.utils.logging import get_logger

logger = get_logger(__name__)


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the ESM2 federated learning server.

    Initializes a FedAvg strategy seeded with the pretrained ESM2 weights
    so all clients start from the same checkpoint.

    Args:
        context: Flower context with run_config.

    Returns:
        ServerAppComponents with strategy and config.
    """
    from sfl.esm2.config import get_run_config

    cfg = get_run_config()
    run_config = context.run_config or {}

    num_rounds = int(run_config.get("num-server-rounds", cfg.num_rounds))
    num_clients = int(run_config.get("num-clients", cfg.num_clients))
    min_fit_clients = int(run_config.get("min-fit-clients", num_clients))
    model_name = str(run_config.get("esm2-model", cfg.model_name))
    fraction_fit = float(run_config.get("fraction-fit", cfg.fraction_fit))
    fraction_evaluate = float(run_config.get("fraction-evaluate", cfg.fraction_evaluate))

    logger.info(
        f"ESM2 server: model={model_name}, rounds={num_rounds}, "
        f"clients={num_clients}, min_fit={min_fit_clients}"
    )

    # Load pretrained model to get initial parameters
    model = load_model(model_name)
    initial_params = ndarrays_to_parameters(get_parameters(model))
    del model  # free memory — clients will load their own copies

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
    )

    config = FlowerServerConfig(num_rounds=num_rounds)

    logger.info("ESM2 server initialized")
    return ServerAppComponents(strategy=strategy, config=config)
