"""
Standalone Flower runner (without NVFlare).

This provides a pure Flower simulation for testing the federated
learning code when NVFlare integration has compatibility issues.

Usage:
    python jobs/flower_runner.py --num-clients 2 --num-rounds 1
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flwr.server import ServerConfig
from flwr.simulation import run_simulation

from sfl.utils.config import load_config
from sfl.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SFL with pure Flower simulation (no NVFlare)",
    )
    
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of federated clients",
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of training rounds",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Build CLI overrides
    cli_overrides = {
        "federation": {
            "num_clients": args.num_clients,
            "num_rounds": args.num_rounds,
        }
    }
    
    if args.verbose:
        cli_overrides["logging"] = {"level": "DEBUG"}
    
    # Load configuration
    config_path = args.config
    if config_path is None:
        default_config = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_config.exists():
            config_path = str(default_config)
    
    config = load_config(config_path=config_path, cli_overrides=cli_overrides)
    
    # Setup logging
    setup_logging(config.logging)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("SFL - Flower Simulation (Standalone)")
    logger.info("=" * 60)
    logger.info(f"Clients: {config.federation.num_clients}")
    logger.info(f"Rounds: {config.federation.num_rounds}")
    logger.info("-" * 60)
    
    # Import apps
    from sfl.client import client_fn
    from sfl.server import server_fn
    
    # Create apps
    from flwr.client import ClientApp
    from flwr.server import ServerApp
    
    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
    
    # Run simulation
    try:
        logger.info("Starting Flower simulation...")
        
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=config.federation.num_clients,
            backend_config={
                "client_resources": {
                    "num_cpus": 1,
                    "num_gpus": 0.0,
                }
            },
        )
        
        logger.info("-" * 60)
        logger.info("Simulation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
