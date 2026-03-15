"""
LLM Federated Fine-Tuning Runner.

Runs federated fine-tuning of causal language models (e.g., GPT-2)
using Flower simulation.

Usage:
    python jobs/llm_runner.py
    python jobs/llm_runner.py --num-clients 4 --num-rounds 3
    python jobs/llm_runner.py --model gpt2-medium --use-lora
    python jobs/llm_runner.py --dp --dp-noise 0.5 --dp-clip 5.0
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfl.utils.logging import setup_logging, get_logger
from sfl.types import LoggingConfig
from sfl.privacy.runner_utils import add_privacy_args, build_privacy_mods, validate_env_vars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated causal LM fine-tuning with Flower",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick demo (2 clients, 1 round, GPT-2 on CPU)
    python jobs/llm_runner.py

    # More clients and rounds
    python jobs/llm_runner.py --num-clients 4 --num-rounds 5

    # LoRA fine-tuning (parameter-efficient)
    python jobs/llm_runner.py --use-lora --lora-r 8

    # With differential privacy
    python jobs/llm_runner.py --dp --dp-noise 0.5

    # Custom dataset from HuggingFace
    python jobs/llm_runner.py --dataset wikitext --text-column text
        """,
    )

    # Federation
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Number of federated clients (default: 2)")
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of FL rounds (default: 3)")

    # Model
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace causal LM name (default: gpt2)")

    # Training
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local training epochs per round (default: 1)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size (default: 2)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token sequence length (default: 128)")

    # LoRA
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoRA parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha scaling factor (default: 16)")

    # Dataset
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (default: built-in demo)")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column name for text data (default: text)")
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

    # All privacy/security flags (DP, filters, SecAgg, aggregation, DP-SGD)
    add_privacy_args(parser)

    return parser.parse_args()


def _set_llm_config(args: argparse.Namespace) -> None:
    """Store CLI args in the shared LLM config module."""
    from sfl.llm.config import LLMRunConfig, set_run_config
    from sfl.types import FederationConfig

    set_run_config(LLMRunConfig(
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
        text_column=args.text_column,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ))


def _save_final_model(args: argparse.Namespace, logger) -> None:
    """Save the trained model after FL completes.

    Loads the model, which in simulation picks up the last state,
    and saves weights + tokenizer to save_dir.
    """
    from sfl.llm.model import load_model, load_tokenizer

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)  # type: ignore[union-attr]

    logger.info(f"Model and tokenizer saved to {save_path}")


def run_flower(args: argparse.Namespace, logger) -> int:
    """Run LLM FL training using pure Flower simulation."""
    from flwr.client import ClientApp
    from flwr.server import ServerApp
    from flwr.simulation import run_simulation

    from sfl.llm.client import client_fn
    from sfl.llm.server import server_fn

    # Store config so client_fn/server_fn can read it
    _set_llm_config(args)

    # Build client mods for privacy (sets SFL_* env vars + returns mod list)
    client_mods = build_privacy_mods(args)
    validate_env_vars()

    if client_mods:
        client_app = ClientApp(client_fn=client_fn, mods=client_mods)
    else:
        client_app = ClientApp(client_fn=client_fn)

    # SecAgg server wrapping
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

    logger.info("Starting Flower simulation for LLM FL...")

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


def main() -> int:
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(LoggingConfig(level=log_level, format=args.log_format))
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("LLM Federated Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Model:         {args.model}")
    logger.info(f"Clients:       {args.num_clients}")
    logger.info(f"Rounds:        {args.num_rounds}")
    logger.info(f"Local epochs:  {args.local_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size:    {args.batch_size}")
    logger.info(f"Max length:    {args.max_length}")
    logger.info(f"LoRA:          {'ON (r=' + str(args.lora_r) + ', alpha=' + str(args.lora_alpha) + ')' if args.use_lora else 'OFF'}")
    logger.info(f"Dataset:       {args.dataset or 'built-in demo (20 texts)'}")
    logger.info(f"Save dir:      {args.save_dir or 'none'}")
    logger.info(f"DP:            {'ON (' + args.dp_mode + ', noise=' + str(args.dp_noise) + ', clip=' + str(args.dp_clip) + ')' if args.dp else 'OFF'}")
    if args.percentile_privacy is not None:
        logger.info(f"Filter:        PercentilePrivacy (top {args.percentile_privacy}%, gamma={args.percentile_gamma})")
    if args.svt_privacy:
        logger.info(f"Filter:        SVTPrivacy (eps={args.svt_epsilon}, frac={args.svt_fraction})")
    if args.exclude_layers:
        logger.info(f"Excluded:      layers {args.exclude_layers}")
    logger.info("-" * 60)

    try:
        return run_flower(args, logger)
    except Exception as e:
        logger.error(f"LLM FL training failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
