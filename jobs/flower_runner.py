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

    # Privacy
    parser.add_argument(
        "--dp",
        action="store_true",
        help="Enable differential privacy",
    )
    parser.add_argument(
        "--dp-noise",
        type=float,
        default=0.1,
        help="DP noise multiplier (default: 0.1)",
    )
    parser.add_argument(
        "--dp-clip",
        type=float,
        default=10.0,
        help="DP clipping norm (default: 10.0)",
    )
    parser.add_argument(
        "--dp-mode",
        type=str,
        default="server",
        choices=["server", "client"],
        help="DP mode: server-side or client-side (default: server)",
    )
    parser.add_argument(
        "--dp-delta",
        type=float,
        default=1e-5,
        help="DP target delta for privacy accounting (default: 1e-5)",
    )
    parser.add_argument(
        "--dp-max-epsilon",
        type=float,
        default=10.0,
        help="DP budget cap — stop training when epsilon exceeds this (default: 10.0)",
    )

    # Privacy filters
    parser.add_argument(
        "--percentile-privacy",
        type=int,
        default=None,
        metavar="PCT",
        help="Enable percentile privacy: only share top PCT%% of diffs (e.g. 10)",
    )
    parser.add_argument(
        "--percentile-gamma",
        type=float,
        default=0.01,
        help="Clipping bound for percentile privacy (default: 0.01)",
    )
    parser.add_argument(
        "--percentile-noise",
        type=float,
        default=0.0,
        help="Gaussian noise scale for percentile privacy (default: 0, no noise)",
    )
    parser.add_argument(
        "--svt-privacy",
        action="store_true",
        help="Enable SVT (Sparse Vector Technique) differential privacy",
    )
    parser.add_argument(
        "--svt-epsilon",
        type=float,
        default=0.1,
        help="SVT privacy budget epsilon (default: 0.1)",
    )
    parser.add_argument(
        "--svt-fraction",
        type=float,
        default=0.1,
        help="SVT fraction of params to upload (default: 0.1)",
    )

    # Secure Aggregation
    parser.add_argument(
        "--secagg",
        action="store_true",
        help="Enable SecAgg+ (secure aggregation)",
    )
    parser.add_argument(
        "--secagg-shares",
        type=int,
        default=3,
        help="SecAgg+ number of secret shares per client (default: 3)",
    )
    parser.add_argument(
        "--secagg-threshold",
        type=int,
        default=2,
        help="SecAgg+ reconstruction threshold (default: 2)",
    )
    parser.add_argument(
        "--secagg-clip",
        type=float,
        default=8.0,
        help="SecAgg+ clipping range (default: 8.0)",
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
        },
        "dp": {
            "enabled": args.dp,
            "noise_multiplier": args.dp_noise,
            "clipping_norm": args.dp_clip,
            "mode": args.dp_mode,
        },
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
    logger.info(f"DP:      {'ON ('+args.dp_mode+', noise='+str(args.dp_noise)+', clip='+str(args.dp_clip)+')' if args.dp else 'OFF'}")
    if args.percentile_privacy is not None:
        logger.info(f"Filter:  PercentilePrivacy (top {args.percentile_privacy}%, gamma={args.percentile_gamma})")
    if args.svt_privacy:
        logger.info(f"Filter:  SVTPrivacy (eps={args.svt_epsilon}, frac={args.svt_fraction})")
    if args.secagg:
        logger.info(f"SecAgg+: ON (shares={args.secagg_shares}, threshold={args.secagg_threshold}, clip={args.secagg_clip})")
    logger.info("-" * 60)
    
    # Import apps
    from sfl.client import client_fn
    from sfl.server import server_fn
    
    # Create apps
    from flwr.client import ClientApp
    from flwr.server import ServerApp
    
    # Build client mods list
    client_mods = []
    if args.dp and args.dp_mode == "client":
        from flwr.client.mod import fixedclipping_mod
        client_mods.append(fixedclipping_mod)
    if args.percentile_privacy is not None:
        from sfl.privacy.filters import make_percentile_privacy_mod
        client_mods.append(
            make_percentile_privacy_mod(args.percentile_privacy, args.percentile_gamma, args.percentile_noise)
        )
    if args.svt_privacy:
        from sfl.privacy.filters import make_svt_privacy_mod
        client_mods.append(
            make_svt_privacy_mod(fraction=args.svt_fraction, epsilon=args.svt_epsilon)
        )
    if args.secagg:
        from flwr.client.mod import secaggplus_mod
        client_mods.append(secaggplus_mod)

    client_app_kwargs = {"client_fn": client_fn}
    if client_mods:
        client_app_kwargs["mods"] = client_mods

    client_app = ClientApp(**client_app_kwargs)

    # Pass DP config via env vars (run_simulation doesn't support run_config)
    import os
    if args.dp:
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_NOISE"] = str(args.dp_noise)
        os.environ["SFL_DP_CLIP"] = str(args.dp_clip)
        os.environ["SFL_DP_MODE"] = args.dp_mode
        os.environ["SFL_DP_DELTA"] = str(args.dp_delta)
        os.environ["SFL_DP_MAX_EPSILON"] = str(args.dp_max_epsilon)

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
