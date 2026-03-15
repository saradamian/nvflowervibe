"""
ESM2 Federated Learning Runner.

Runs federated fine-tuning of ESM2 protein language models using Flower
simulation or NVFlare distributed execution.

Usage:
    python jobs/esm2_runner.py
    python jobs/esm2_runner.py --num-clients 4 --num-rounds 3
    python jobs/esm2_runner.py --backend nvflare-sim
    python jobs/esm2_runner.py --backend nvflare --startup-kit ~/.nvflare/admin
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfl.utils.logging import setup_logging, get_logger
from sfl.types import LoggingConfig
from sfl.privacy.runner_utils import add_privacy_args, build_privacy_mods, validate_env_vars


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

    # NVFlare local simulation
    python jobs/esm2_runner.py --backend nvflare-sim

    # NVFlare distributed (requires provisioned startup kits)
    python jobs/esm2_runner.py --backend nvflare --startup-kit ~/.nvflare/admin
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
                        choices=["flower", "nvflare-sim", "nvflare-poc", "nvflare"],
                        help="Execution backend (default: flower)")
    parser.add_argument("--startup-kit", type=str, default=None,
                        help="Path to NVFlare admin startup kit (required for --backend nvflare)")

    # All privacy/security flags (DP, filters, SecAgg, aggregation, DP-SGD)
    add_privacy_args(parser)

    return parser.parse_args()


def _save_final_model(args: argparse.Namespace, logger) -> None:
    """Save the trained model after FL completes."""
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


def _build_esm2_run_config(args: argparse.Namespace) -> dict:
    """Build Flower run_config dict from CLI args for NVFlare staging."""
    return {
        "num-server-rounds": args.num_rounds,
        "num-clients": args.num_clients,
        "esm2-model": args.model,
        "learning-rate": args.learning_rate,
        "local-epochs": args.local_epochs,
        "batch-size": args.batch_size,
        "max-length": args.max_length,
        "dataset-name": args.dataset or "",
        "sequence-column": args.sequence_column,
        "max-samples": args.max_samples or 0,
    }


def run_flower(args: argparse.Namespace, logger) -> int:
    """Run ESM2 FL training using pure Flower simulation."""
    from flwr.client import ClientApp
    from flwr.server import ServerApp
    from flwr.simulation import run_simulation

    from sfl.esm2.client import client_fn
    from sfl.esm2.server import server_fn

    _set_esm2_config(args)

    client_mods = build_privacy_mods(args)
    validate_env_vars()

    if client_mods:
        client_app = ClientApp(client_fn=client_fn, mods=client_mods)
    else:
        client_app = ClientApp(client_fn=client_fn)

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
                "num_cpus": getattr(args, "client_cpus", 1),
                "num_gpus": gpus_per_client if not getattr(args, "no_auto_detect_gpu", False) else getattr(args, "client_gpus", 0),
            }
        },
    )

    if args.save_dir:
        _save_final_model(args, logger)

    return 0


def run_nvflare(args: argparse.Namespace, logger) -> int:
    """Run ESM2 FL training using NVFlare backend."""
    from sfl.nvflare.backend import (
        NVFlareBackendConfig, NVFlareMode, build_extra_env, run_nvflare as nvflare_run,
    )
    from sfl.nvflare.staging import stage_flower_content

    _set_esm2_config(args)

    # Build privacy mods (sets SFL_* env vars) — for Flower simulation
    # the mods list is used directly; for NVFlare the env vars are
    # propagated via extra_env and auto_build_client_mods() rebuilds
    # the mods on each site.
    build_privacy_mods(args)
    validate_env_vars()

    extra_env = build_extra_env(include_non_sfl=True)

    # Add DP config to run_config so server_fn can read it
    run_config = _build_esm2_run_config(args)
    if args.dp:
        run_config["dp-enabled"] = True
        run_config["dp-noise-multiplier"] = args.dp_noise
        run_config["dp-clipping-norm"] = args.dp_clip
        run_config["dp-mode"] = args.dp_mode

    project_root = Path(__file__).parent.parent
    staging_dir = stage_flower_content(project_root, "esm2", run_config)

    mode_map = {
        "nvflare-sim": NVFlareMode.SIM,
        "nvflare-poc": NVFlareMode.POC,
        "nvflare": NVFlareMode.PROD,
    }

    try:
        config = NVFlareBackendConfig(
            mode=mode_map[args.backend],
            num_clients=args.num_clients,
            flower_content=str(staging_dir),
            extra_env=extra_env,
            job_name=f"sfl-esm2-{args.model.split('/')[-1]}",
            startup_kit=args.startup_kit,
        )
        return nvflare_run(config)
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
        if args.backend == "flower":
            return run_flower(args, logger)
        else:
            return run_nvflare(args, logger)
    except Exception as e:
        logger.error(f"ESM2 FL training failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
