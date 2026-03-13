"""
ESM2 Federated Learning Runner.

Runs federated fine-tuning of ESM2 protein language models using Flower
simulation, optionally orchestrated by NVFlare.

Usage:
    python jobs/esm2_runner.py
    python jobs/esm2_runner.py --num-clients 4 --num-rounds 3
    python jobs/esm2_runner.py --model facebook/esm2_t12_35M_UR50D --local-epochs 2
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfl.utils.logging import setup_logging, get_logger
from sfl.types import LoggingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated ESM2 fine-tuning with Flower",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick demo (2 clients, 1 round, 8M-param model)
    python jobs/esm2_runner.py

    # More clients and rounds
    python jobs/esm2_runner.py --num-clients 4 --num-rounds 5

    # Larger model
    python jobs/esm2_runner.py --model facebook/esm2_t12_35M_UR50D

    # Adjust training hyperparams
    python jobs/esm2_runner.py --learning-rate 1e-4 --local-epochs 2 --batch-size 8
        """,
    )

    # Federation
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Number of federated clients (default: 2)")
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of FL rounds (default: 3)")

    # Model
    parser.add_argument("--model", type=str,
                        default="facebook/esm2_t6_8M_UR50D",
                        help="HuggingFace ESM2 model name (default: esm2_t6_8M)")

    # Training
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local training epochs per round (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token sequence length (default: 128)")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--log-format", type=str, default="rich",
                        choices=["rich", "simple", "json"],
                        help="Log format (default: rich)")

    # Backend
    parser.add_argument("--backend", type=str, default="flower",
                        choices=["flower", "nvflare"],
                        help="Simulation backend (default: flower)")

    return parser.parse_args()


def _set_esm2_config(args: argparse.Namespace) -> None:
    """Store CLI args in the shared ESM2 config module."""
    from sfl.esm2.config import ESM2RunConfig, set_run_config
    from sfl.types import FederationConfig

    set_run_config(ESM2RunConfig(
        federation=FederationConfig(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
        ),
        model_name=args.model,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ))


def run_flower(args: argparse.Namespace, logger) -> int:
    """Run ESM2 FL training using pure Flower simulation."""
    from flwr.client import ClientApp
    from flwr.server import ServerApp
    from flwr.simulation import run_simulation

    from sfl.esm2.client import client_fn
    from sfl.esm2.server import server_fn

    # Store config so client_fn/server_fn can read it
    _set_esm2_config(args)

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    logger.info("Starting Flower simulation for ESM2 FL...")

    # Detect GPU availability and allocate to clients
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 0

    gpus_per_client = num_gpus / args.num_clients if num_gpus > 0 else 0.0

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": gpus_per_client,
            }
        },
    )
    return 0


def _write_esm2_pyproject(staging_dir: Path, args) -> None:
    """Write a pyproject.toml that points Flower at the ESM2 apps."""
    content = f"""\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sfl-esm2"
version = "0.1.0"

[tool.hatch.build.targets.wheel]
packages = ["src/sfl"]

[tool.flwr.app]
publisher = "sfl-demo"

[tool.flwr.app.components]
serverapp = "sfl.esm2:server_app"
clientapp = "sfl.esm2:client_app"

[tool.flwr.app.config]
num-server-rounds = {args.num_rounds}
num-clients = {args.num_clients}
esm2-model = "{args.model}"
learning-rate = {args.learning_rate}
local-epochs = {args.local_epochs}
batch-size = {args.batch_size}
max-length = {args.max_length}

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = {args.num_clients}
address = "127.0.0.1:9093"
insecure = true
"""
    (staging_dir / "pyproject.toml").write_text(content)


def run_nvflare(args: argparse.Namespace, logger) -> int:
    """Run ESM2 FL training using NVFlare FlowerRecipe."""
    import shutil
    import tempfile

    try:
        from nvflare.app_opt.flower.recipe import FlowerRecipe
        from nvflare.recipe.sim_env import SimEnv
    except ImportError as e:
        logger.error(f"NVFlare not available: {e}")
        logger.error("Install with: pip install nvflare==2.7.1")
        logger.info("Falling back to pure Flower simulation...")
        return run_flower(args, logger)

    content_dir = Path(__file__).parent.parent
    if not (content_dir / "pyproject.toml").exists():
        logger.error(f"pyproject.toml not found in {content_dir}")
        return 1

    # Stage only needed files (avoids copying .git/, .venv/, etc.)
    staging_dir = Path(tempfile.mkdtemp(prefix="sfl_esm2_nvflare_"))
    try:
        shutil.copytree(content_dir / "src", staging_dir / "src")
        if (content_dir / "config").exists():
            shutil.copytree(content_dir / "config", staging_dir / "config")

        # Write a pyproject.toml that points Flower at ESM2 apps
        _write_esm2_pyproject(staging_dir, args)

        logger.info("Starting NVFlare SimEnv for ESM2 FL...")

        recipe = FlowerRecipe(
            flower_content=str(staging_dir),
            name="esm2-federated-mlm",
            min_clients=args.num_clients,
        )

        env = SimEnv(
            num_clients=args.num_clients,
            num_threads=args.num_clients,
        )

        recipe.execute(env=env)
        return 0
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(LoggingConfig(level=log_level, format=args.log_format))
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ESM2 Federated Learning")
    logger.info("=" * 60)
    logger.info(f"Model:         {args.model}")
    logger.info(f"Clients:       {args.num_clients}")
    logger.info(f"Rounds:        {args.num_rounds}")
    logger.info(f"Local epochs:  {args.local_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size:    {args.batch_size}")
    logger.info(f"Backend:       {args.backend}")
    logger.info("-" * 60)

    try:
        if args.backend == "nvflare":
            return run_nvflare(args, logger)
        else:
            return run_flower(args, logger)
    except Exception as e:
        logger.error(f"ESM2 FL training failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
