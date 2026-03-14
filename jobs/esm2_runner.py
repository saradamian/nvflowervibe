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

    # Dataset
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (default: built-in demo)")
    parser.add_argument("--sequence-column", type=str, default="sequence",
                        help="Column name for protein sequences (default: sequence)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to load from dataset (default: all)")

    # Output
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save final model (default: no save)")

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

    # Privacy
    parser.add_argument("--dp", action="store_true",
                        help="Enable differential privacy")
    parser.add_argument("--dp-noise", type=float, default=0.1,
                        help="DP noise multiplier (default: 0.1)")
    parser.add_argument("--dp-clip", type=float, default=10.0,
                        help="DP clipping norm (default: 10.0)")
    parser.add_argument("--dp-mode", type=str, default="server",
                        choices=["server", "client"],
                        help="DP mode: server-side or client-side (default: server)")
    parser.add_argument("--dp-delta", type=float, default=1e-5,
                        help="DP target delta for privacy accounting (default: 1e-5)")
    parser.add_argument("--dp-max-epsilon", type=float, default=10.0,
                        help="DP budget cap — stop training when epsilon exceeds this (default: 10.0)")

    # Privacy filters
    parser.add_argument("--percentile-privacy", type=int, default=None, metavar="PCT",
                        help="Enable percentile privacy: only share top PCT%% of diffs")
    parser.add_argument("--percentile-gamma", type=float, default=0.01,
                        help="Clipping bound for percentile privacy (default: 0.01)")
    parser.add_argument("--percentile-noise", type=float, default=0.0,
                        help="Gaussian noise scale for percentile privacy (default: 0, no noise)")
    parser.add_argument("--percentile-epsilon", type=float, default=0.0,
                        help="Calibrate percentile noise to this ε (overrides --percentile-noise)")
    parser.add_argument("--percentile-delta", type=float, default=1e-5,
                        help="δ for percentile noise calibration (default: 1e-5)")
    parser.add_argument("--svt-privacy", action="store_true",
                        help="Enable SVT (Sparse Vector Technique) differential privacy")
    parser.add_argument("--svt-epsilon", type=float, default=0.1,
                        help="SVT privacy budget epsilon (default: 0.1)")
    parser.add_argument("--svt-fraction", type=float, default=0.1,
                        help="SVT fraction of params to upload (default: 0.1)")
    parser.add_argument("--exclude-layers", type=str, default=None,
                        help="Comma-separated parameter indices to exclude (e.g. '0,1')")

    # Secure Aggregation
    parser.add_argument("--secagg", action="store_true",
                        help="Enable SecAgg+ (secure aggregation)")
    parser.add_argument("--secagg-shares", type=int, default=3,
                        help="SecAgg+ number of secret shares per client (default: 3)")
    parser.add_argument("--secagg-threshold", type=int, default=2,
                        help="SecAgg+ reconstruction threshold (default: 2)")
    parser.add_argument("--secagg-clip", type=float, default=8.0,
                        help="SecAgg+ clipping range (default: 8.0)")

    return parser.parse_args()


def _save_final_model(args: argparse.Namespace, logger) -> None:
    """Save the trained model after FL completes.

    Loads the model, which in simulation picks up the last state,
    and saves weights + tokenizer to save_dir.
    """
    from sfl.esm2.model import load_model, load_tokenizer

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)  # type: ignore[union-attr]

    logger.info(f"Model and tokenizer saved to {save_path}")


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
        dataset_name=args.dataset,
        sequence_column=args.sequence_column,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
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

    # Build client mods for privacy
    client_mods = []
    if args.dp:
        import os
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_NOISE"] = str(args.dp_noise)
        os.environ["SFL_DP_CLIP"] = str(args.dp_clip)
        os.environ["SFL_DP_MODE"] = args.dp_mode
        os.environ["SFL_DP_DELTA"] = str(args.dp_delta)
        os.environ["SFL_DP_MAX_EPSILON"] = str(args.dp_max_epsilon)

        if args.dp_mode == "client":
            from flwr.client.mod import fixedclipping_mod
            client_mods.append(fixedclipping_mod)

    if args.percentile_privacy is not None:
        from sfl.privacy.filters import make_percentile_privacy_mod
        client_mods.append(
            make_percentile_privacy_mod(
                args.percentile_privacy, args.percentile_gamma, args.percentile_noise,
                epsilon=args.percentile_epsilon, delta=args.percentile_delta,
            )
        )
    if args.svt_privacy:
        from sfl.privacy.filters import make_svt_privacy_mod
        client_mods.append(
            make_svt_privacy_mod(fraction=args.svt_fraction, epsilon=args.svt_epsilon)
        )
    if args.exclude_layers:
        from sfl.privacy.filters import make_exclude_vars_mod
        indices = [int(x.strip()) for x in args.exclude_layers.split(",")]
        client_mods.append(make_exclude_vars_mod(exclude_indices=indices))
    if args.secagg:
        from flwr.client.mod import secaggplus_mod
        client_mods.append(secaggplus_mod)

    if client_mods:
        client_app = ClientApp(client_fn=client_fn, mods=client_mods)

    if args.secagg:
        from sfl.privacy.secagg import SecAggConfig, make_secagg_main
        secagg_cfg = SecAggConfig(
            num_shares=args.secagg_shares,
            reconstruction_threshold=args.secagg_threshold,
            clipping_range=args.secagg_clip,
        )
        server_app = ServerApp()
        server_app.main()(make_secagg_main(server_fn, secagg_cfg))
    else:
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

    # Save final model if requested
    if args.save_dir:
        _save_final_model(args, logger)

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
dataset-name = "{args.dataset or ''}"
sequence-column = "{args.sequence_column}"
max-samples = {args.max_samples if args.max_samples else 0}

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
    logger.info(f"Dataset:       {args.dataset or 'built-in demo (32 seqs)'}")
    logger.info(f"Save dir:      {args.save_dir or 'none'}")
    logger.info(f"Backend:       {args.backend}")
    logger.info(f"DP:            {'ON ('+args.dp_mode+', noise='+str(args.dp_noise)+', clip='+str(args.dp_clip)+')' if args.dp else 'OFF'}")
    if args.percentile_privacy is not None:
        logger.info(f"Filter:        PercentilePrivacy (top {args.percentile_privacy}%, gamma={args.percentile_gamma})")
    if args.svt_privacy:
        logger.info(f"Filter:        SVTPrivacy (eps={args.svt_epsilon}, frac={args.svt_fraction})")
    if args.exclude_layers:
        logger.info(f"Excluded:      layers {args.exclude_layers}")
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
