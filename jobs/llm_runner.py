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
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfl.utils.logging import setup_logging, get_logger
from sfl.types import LoggingConfig


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

    # Privacy -- DP
    parser.add_argument("--dp", action="store_true",
                        help="Enable differential privacy")
    parser.add_argument("--dp-noise", type=float, default=1.0,
                        help="DP noise multiplier (default: 1.0)")
    parser.add_argument("--dp-clip", type=float, default=10.0,
                        help="DP clipping norm (default: 10.0)")
    parser.add_argument("--dp-mode", type=str, default="server",
                        choices=["server", "client"],
                        help="DP mode: server-side or client-side (default: server)")
    parser.add_argument("--dp-delta", type=float, default=1e-5,
                        help="DP target delta (default: 1e-5)")

    # Privacy filters
    parser.add_argument("--percentile-privacy", type=int, default=None, metavar="PCT",
                        help="Enable percentile privacy: only share top PCT%% of diffs")
    parser.add_argument("--percentile-gamma", type=float, default=0.01,
                        help="Clipping bound for percentile privacy (default: 0.01)")
    parser.add_argument("--percentile-noise", type=float, default=0.0,
                        help="Gaussian noise scale for percentile privacy (default: 0)")
    parser.add_argument("--percentile-epsilon", type=float, default=0.0,
                        help="Calibrate percentile noise to this epsilon")
    parser.add_argument("--percentile-delta", type=float, default=1e-5,
                        help="Delta for percentile noise calibration (default: 1e-5)")
    parser.add_argument("--svt-privacy", action="store_true",
                        help="Enable SVT differential privacy")
    parser.add_argument("--svt-epsilon", type=float, default=0.1,
                        help="SVT privacy budget epsilon (default: 0.1)")
    parser.add_argument("--svt-fraction", type=float, default=0.1,
                        help="SVT fraction of params to upload (default: 0.1)")
    parser.add_argument("--svt-no-optimal", action="store_true",
                        help="Use standard budget split instead of optimal")
    parser.add_argument("--svt-prescreen", type=float, default=1.0,
                        help="SVT pre-screen ratio (default: 1.0)")
    parser.add_argument("--exclude-layers", type=str, default=None,
                        help="Comma-separated parameter indices to exclude")

    # Gradient compression
    parser.add_argument("--compress", type=float, default=None, metavar="RATIO",
                        help="Gradient compression ratio (e.g. 0.1)")
    parser.add_argument("--compress-noise", type=float, default=0.01,
                        help="Noise scale for compressed gradients (default: 0.01)")
    parser.add_argument("--compress-topk", action="store_true",
                        help="Use deterministic TopK instead of random masking")
    parser.add_argument("--compress-error-feedback", action="store_true",
                        help="Accumulate compression residuals across rounds")

    # Per-example DP-SGD
    parser.add_argument("--dpsgd", action="store_true",
                        help="Enable per-example DP-SGD via Opacus")
    parser.add_argument("--dpsgd-clip", type=float, default=1.0,
                        help="Per-example gradient clip norm (default: 1.0)")
    parser.add_argument("--dpsgd-noise", type=float, default=1.0,
                        help="Noise multiplier for DP-SGD (default: 1.0)")
    parser.add_argument("--dpsgd-delta", type=float, default=1e-5,
                        help="Delta for per-example DP (default: 1e-5)")
    parser.add_argument("--dpsgd-autoclip", action="store_true",
                        help="Enable AutoClip for DP-SGD")
    parser.add_argument("--dpsgd-ghost", action="store_true",
                        help="Enable Ghost Clipping for memory-efficient DP-SGD")

    # Secure Aggregation
    parser.add_argument("--secagg", action="store_true",
                        help="Enable SecAgg+ (secure aggregation)")
    parser.add_argument("--secagg-shares", type=int, default=3,
                        help="SecAgg+ number of secret shares (default: 3)")
    parser.add_argument("--secagg-threshold", type=int, default=2,
                        help="SecAgg+ reconstruction threshold (default: 2)")
    parser.add_argument("--secagg-clip", type=float, default=8.0,
                        help="SecAgg+ clipping range (default: 8.0)")

    return parser.parse_args()


def _build_privacy_mods(args: argparse.Namespace) -> list:
    """Build the list of Flower client mods for privacy features.

    Reads privacy-related CLI args and constructs the appropriate
    Flower client mods. This keeps the main runner function clean
    and the privacy setup reusable.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of Flower client mod callables.
    """
    client_mods = []

    # Server/client-side DP
    if args.dp:
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_NOISE"] = str(args.dp_noise)
        os.environ["SFL_DP_CLIP"] = str(args.dp_clip)
        os.environ["SFL_DP_MODE"] = args.dp_mode
        os.environ["SFL_DP_DELTA"] = str(args.dp_delta)

        if args.dp_mode == "client":
            from flwr.client.mod import fixedclipping_mod
            client_mods.append(fixedclipping_mod)

    # Percentile privacy filter
    if args.percentile_privacy is not None:
        from sfl.privacy.filters import make_percentile_privacy_mod
        client_mods.append(
            make_percentile_privacy_mod(
                args.percentile_privacy, args.percentile_gamma,
                args.percentile_noise,
                epsilon=args.percentile_epsilon,
                delta=args.percentile_delta,
            )
        )

    # SVT privacy filter
    if args.svt_privacy:
        from sfl.privacy.filters import make_svt_privacy_mod
        client_mods.append(
            make_svt_privacy_mod(
                fraction=args.svt_fraction, epsilon=args.svt_epsilon,
                optimal_budget=not args.svt_no_optimal,
                pre_screen_ratio=args.svt_prescreen,
            )
        )

    # Layer exclusion
    if args.exclude_layers:
        from sfl.privacy.filters import make_exclude_vars_mod
        indices = [int(x.strip()) for x in args.exclude_layers.split(",")]
        client_mods.append(make_exclude_vars_mod(exclude_indices=indices))

    # Gradient compression
    if args.compress is not None:
        from sfl.privacy.filters import make_gradient_compression_mod
        client_mods.append(
            make_gradient_compression_mod(
                compression_ratio=args.compress,
                noise_scale=args.compress_noise,
                use_random_mask=not args.compress_topk,
                error_feedback=args.compress_error_feedback,
            )
        )

    # Per-example DP-SGD (Opacus) -- configured via env vars, read by client_fn
    if args.dpsgd:
        os.environ["SFL_DPSGD_ENABLED"] = "true"
        os.environ["SFL_DPSGD_CLIP"] = str(args.dpsgd_clip)
        os.environ["SFL_DPSGD_NOISE"] = str(args.dpsgd_noise)
        os.environ["SFL_DPSGD_DELTA"] = str(args.dpsgd_delta)
        if args.dpsgd_autoclip:
            os.environ["SFL_DPSGD_AUTOCLIP"] = "true"
        if args.dpsgd_ghost:
            os.environ["SFL_DPSGD_GHOST"] = "true"

    # SecAgg
    if args.secagg:
        from flwr.client.mod import secaggplus_mod
        client_mods.append(secaggplus_mod)

    return client_mods


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

    # Build client mods for privacy
    client_mods = _build_privacy_mods(args)

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
