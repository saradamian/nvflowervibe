"""
NVFlare Job Runner for SFL.

This module provides the main entry point for running federated learning
simulations using NVFlare's Flower integration. It supports configuration
via CLI arguments, YAML files, and environment variables.

Usage:
    python jobs/runner.py --num-clients 4 --num-rounds 3
    python jobs/runner.py --config config/default.yaml
    python jobs/runner.py --help
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfl.utils.config import load_config
from sfl.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run SFL federated learning simulation with NVFlare + Flower",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults (2 clients, 1 round)
    python jobs/runner.py
    
    # Run with custom settings
    python jobs/runner.py --num-clients 4 --num-rounds 3
    
    # Run with config file
    python jobs/runner.py --config config/default.yaml
    
    # Enable metrics streaming to TensorBoard
    python jobs/runner.py --stream-metrics
        """,
    )
    
    # Job configuration
    parser.add_argument(
        "--job-name",
        type=str,
        default="sfl-federated-sum",
        help="Name of the NVFlare job (default: sfl-federated-sum)",
    )
    
    parser.add_argument(
        "--content-dir",
        type=str,
        default=None,
        help="Directory containing Flower app (default: auto-detected)",
    )
    
    # Federation settings
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Number of federated clients (default: 2)",
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of training rounds (default: 1)",
    )
    
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of simulation threads (default: num_clients)",
    )
    
    # Features
    parser.add_argument(
        "--stream-metrics",
        action="store_true",
        help="Enable TensorBoard metrics streaming",
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (default: INFO)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    
    return parser.parse_args()


def get_content_dir() -> Path:
    """Get the content directory for Flower app.
    
    Returns:
        Path to the directory containing pyproject.toml.
    """
    # The content dir should be the project root (where pyproject.toml is)
    project_root = Path(__file__).parent.parent
    
    if (project_root / "pyproject.toml").exists():
        return project_root
    
    raise FileNotFoundError(
        f"Could not find pyproject.toml in {project_root}. "
        "Please run from the project root or specify --content-dir."
    )


def main() -> int:
    """Main entry point for the NVFlare job runner.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()
    
    # Build CLI overrides
    cli_overrides = {}
    
    if args.num_clients is not None:
        cli_overrides.setdefault("federation", {})["num_clients"] = args.num_clients
    
    if args.num_rounds is not None:
        cli_overrides.setdefault("federation", {})["num_rounds"] = args.num_rounds
    
    if args.job_name:
        cli_overrides.setdefault("nvflare", {})["job_name"] = args.job_name
    
    if args.stream_metrics:
        cli_overrides.setdefault("nvflare", {})["stream_metrics"] = True
    
    if args.num_threads is not None:
        cli_overrides.setdefault("nvflare", {})["num_threads"] = args.num_threads
    
    if args.verbose:
        cli_overrides.setdefault("logging", {})["level"] = "DEBUG"
    elif args.log_level:
        cli_overrides.setdefault("logging", {})["level"] = args.log_level
    
    # Load configuration
    config_path = args.config
    if config_path is None:
        # Try default config path
        default_config = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_config.exists():
            config_path = str(default_config)
    
    config = load_config(
        config_path=config_path,
        cli_overrides=cli_overrides if cli_overrides else None,
    )
    
    # Setup logging
    setup_logging(config.logging)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("SFL - Simple Federated Learning Demo")
    logger.info("=" * 60)
    logger.info(f"Configuration loaded from: {config_path or 'defaults'}")
    logger.info(f"Federation: {config.federation.num_clients} clients, "
                f"{config.federation.num_rounds} rounds")
    
    # Import NVFlare components (after logging is set up)
    try:
        from nvflare.app_opt.flower.recipe import FlowerRecipe
        from nvflare.recipe.sim_env import SimEnv
    except ImportError as e:
        logger.error(f"Failed to import NVFlare: {e}")
        logger.error("Please install nvflare: pip install nvflare==2.7.1")
        return 1
    
    # Get content directory
    try:
        content_dir = Path(args.content_dir) if args.content_dir else get_content_dir()
        logger.info(f"Content directory: {content_dir}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Create NVFlare recipe
    num_threads = config.nvflare.num_threads or config.federation.num_clients
    
    # Stage only the files NVFlare needs into a clean temp directory
    # (avoids copying .git/, .venv/, __pycache__, etc.)
    staging_dir = Path(tempfile.mkdtemp(prefix="sfl_nvflare_"))
    try:
        shutil.copy(content_dir / "pyproject.toml", staging_dir)
        shutil.copytree(content_dir / "src", staging_dir / "src")
        if (content_dir / "config").exists():
            shutil.copytree(content_dir / "config", staging_dir / "config")

        logger.info("Creating NVFlare FlowerRecipe...")
        recipe = FlowerRecipe(
            flower_content=str(staging_dir),
            name=config.nvflare.job_name,
            min_clients=config.federation.num_clients,
        )
    
        # Create simulation environment
        logger.info(f"Creating SimEnv with {config.federation.num_clients} clients, "
                    f"{num_threads} threads...")
        env = SimEnv(
            num_clients=config.federation.num_clients,
            num_threads=num_threads,
        )
    
        # Execute the job
        logger.info("Starting federated learning simulation...")
        logger.info("-" * 60)
    
        recipe.execute(env=env)
        logger.info("-" * 60)
        logger.info("Simulation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        logger.exception("Full traceback:")
        return 1
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
