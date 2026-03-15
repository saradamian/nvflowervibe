"""
Shared utilities for building privacy mods from CLI arguments.

Centralizes the privacy mod assembly logic that all runners need,
eliminating ~100 lines of duplication per new runner.

Usage in a runner::

    from sfl.privacy.runner_utils import add_privacy_args, build_privacy_mods, validate_env_vars

    def parse_args():
        parser = argparse.ArgumentParser(...)
        # ... runner-specific args ...
        add_privacy_args(parser)
        return parser.parse_args()

    def run_flower(args, logger):
        validate_env_vars()
        client_mods = build_privacy_mods(args)
        client_app = ClientApp(client_fn=client_fn, mods=client_mods or None)
        ...
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, List


def add_privacy_args(parser: argparse.ArgumentParser) -> None:
    """Add all privacy-related CLI arguments to an argparse parser.

    Call this from any runner to get the full set of privacy flags
    (--dp, --percentile-privacy, --svt-privacy, --compress, etc.)
    without copy-pasting the argparse definitions.
    """
    # ── Checkpointing ────────────────────────────────────────────────
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for round-level checkpoints (enables auto-save)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint in --checkpoint-dir")

    # ── Differential Privacy ─────────────────────────────────────────
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
                        help="DP target delta for privacy accounting (default: 1e-5)")
    parser.add_argument("--dp-max-epsilon", type=float, default=10.0,
                        help="DP budget cap -- stop training when epsilon exceeds this (default: 10.0)")
    parser.add_argument("--dp-adaptive-clip", action="store_true",
                        help="Enable adaptive clipping norm (Andrew et al. 2021)")
    parser.add_argument("--dp-target-quantile", type=float, default=0.5,
                        help="Target unclipped fraction for adaptive clipping (default: 0.5)")
    parser.add_argument("--dp-clip-lr", type=float, default=0.2,
                        help="Learning rate for adaptive clip norm update (default: 0.2)")
    parser.add_argument("--dp-quantile-noise", type=float, default=None,
                        help="Noise multiplier for private quantile tracking in adaptive "
                             "clipping (default: 0.1 when adaptive clipping enabled, "
                             "0 otherwise). Set to 0 for non-private (testing only).")
    parser.add_argument("--dp-accounting-backend", type=str, default="pld",
                        choices=["pld", "prv"],
                        help="Privacy accounting backend: pld (Google, default) or "
                             "prv (Microsoft, with error bounds)")
    parser.add_argument("--dp-shuffle", action="store_true",
                        help="Enable shuffle-model DP amplification (assumes anonymous channel)")

    # ── Privacy Filters ──────────────────────────────────────────────
    parser.add_argument("--percentile-privacy", type=int, default=None, metavar="PCT",
                        help="Enable percentile privacy: only share top PCT%%%% of diffs")
    parser.add_argument("--percentile-gamma", type=float, default=0.01,
                        help="Clipping bound for percentile privacy (default: 0.01)")
    parser.add_argument("--percentile-noise", type=float, default=0.0,
                        help="Gaussian noise scale for percentile privacy (default: 0, no noise)")
    parser.add_argument("--percentile-epsilon", type=float, default=0.0,
                        help="Calibrate percentile noise to this epsilon (overrides --percentile-noise)")
    parser.add_argument("--percentile-delta", type=float, default=1e-5,
                        help="delta for percentile noise calibration (default: 1e-5)")
    parser.add_argument("--svt-privacy", action="store_true",
                        help="Enable SVT (Sparse Vector Technique) differential privacy")
    parser.add_argument("--svt-epsilon", type=float, default=0.1,
                        help="SVT privacy budget epsilon (default: 0.1)")
    parser.add_argument("--svt-fraction", type=float, default=0.1,
                        help="SVT fraction of params to upload (default: 0.1)")
    parser.add_argument("--svt-no-optimal", action="store_true",
                        help="Use standard eps/2 + eps/(2c) budget split instead of optimal")
    parser.add_argument("--svt-prescreen", type=float, default=1.0,
                        help="Pre-screen ratio: run SVT on top X%%%% by magnitude (default: 1.0)")
    parser.add_argument("--exclude-layers", type=str, default=None,
                        help="Comma-separated parameter indices to exclude (e.g. '0,1')")

    # ── Gradient Compression ─────────────────────────────────────────
    parser.add_argument("--compress", type=float, default=None, metavar="RATIO",
                        help="Gradient compression: keep this fraction of values (e.g. 0.1)")
    parser.add_argument("--compress-noise", type=float, default=0.01,
                        help="Noise scale for compressed gradients (default: 0.01)")
    parser.add_argument("--compress-topk", action="store_true",
                        help="Use deterministic TopK instead of random masking")
    parser.add_argument("--compress-error-feedback", action="store_true",
                        help="Enable error feedback: accumulate compression residuals across rounds")

    # ── Partial Freezing (Lambda-SecAgg) ─────────────────────────────
    parser.add_argument("--freeze-layers", type=str, default=None,
                        help="Comma-separated trainable layer indices (e.g. '4,5,6'). "
                             "Frozen layers stripped from updates before SecAgg.")

    # ── Per-layer Clipping ───────────────────────────────────────────
    parser.add_argument("--per-layer-clip", type=float, default=None, metavar="NORM",
                        help="Enable per-layer clipping with this default L2 norm per layer "
                             "(Yu et al., ICLR 2022)")
    parser.add_argument("--per-layer-clip-map", type=str, default=None,
                        help="JSON mapping of layer index to clip norm, e.g. '{\"0\": 5.0}'")

    # ── Secure Aggregation ───────────────────────────────────────────
    parser.add_argument("--secagg", action="store_true",
                        help="Enable SecAgg+ (secure aggregation)")
    parser.add_argument("--secagg-shares", type=int, default=3,
                        help="SecAgg+ number of secret shares per client (default: 3)")
    parser.add_argument("--secagg-threshold", type=int, default=2,
                        help="SecAgg+ reconstruction threshold (default: 2)")
    parser.add_argument("--secagg-clip", type=float, default=8.0,
                        help="SecAgg+ clipping range (default: 8.0)")

    # ── Aggregation Strategy ─────────────────────────────────────────
    parser.add_argument("--aggregation", type=str, default="fedavg",
                        choices=["fedavg", "krum", "trimmed-mean", "foundation-fl"],
                        help="Aggregation strategy (default: fedavg)")
    parser.add_argument("--krum-byzantine", type=int, default=1,
                        help="Expected number of Byzantine clients for Multi-Krum (default: 1)")
    parser.add_argument("--trim-ratio", type=float, default=0.1,
                        help="Fraction to trim per side for trimmed-mean (default: 0.1)")
    parser.add_argument("--ffl-threshold", type=float, default=0.1,
                        help="FoundationFL trust threshold -- min cosine similarity (default: 0.1)")
    parser.add_argument("--ffl-no-weighted", action="store_true",
                        help="Disable trust-weighted averaging in FoundationFL")

    # ── Per-example DP-SGD (Opacus) ──────────────────────────────────
    parser.add_argument("--dpsgd", action="store_true",
                        help="Enable per-example DP-SGD via Opacus")
    parser.add_argument("--dpsgd-clip", type=float, default=1.0,
                        help="Per-example gradient clip norm (default: 1.0)")
    parser.add_argument("--dpsgd-noise", type=float, default=1.0,
                        help="Noise multiplier for DP-SGD (default: 1.0)")
    parser.add_argument("--dpsgd-delta", type=float, default=1e-5,
                        help="Delta for per-example DP accounting (default: 1e-5)")
    parser.add_argument("--dpsgd-autoclip", action="store_true",
                        help="Enable AutoClip (Li et al. 2023): normalize per-example "
                             "gradients to unit norm, eliminating clipping norm tuning")
    parser.add_argument("--dpsgd-ghost", action="store_true",
                        help="Enable Ghost Clipping: memory-efficient two-pass DP-SGD "
                             "(reduces memory from O(B*P) to O(B+P))")

    # ── Metrics ─────────────────────────────────────────────────────────
    parser.add_argument("--metrics-dir", type=str, default=None,
                        help="Directory for metrics output files (default: None, no export)")
    parser.add_argument("--metrics-format", type=str, default="csv",
                        choices=["csv", "json", "tensorboard", "all"],
                        help="Metrics export format (default: csv)")

    # ── Resource Allocation ──────────────────────────────────────────
    parser.add_argument("--client-cpus", type=int, default=1,
                        help="CPU cores per simulation client (default: 1)")
    parser.add_argument("--client-gpus", type=float, default=0,
                        help="GPUs per client (default: 0 = auto-detect and split evenly)")
    parser.add_argument("--client-memory", type=int, default=None, metavar="MB",
                        help="Memory hint per client in MB (for documentation/scheduling)")
    parser.add_argument("--no-auto-detect-gpu", action="store_true",
                        help="Disable automatic GPU detection; use --client-gpus literally")


def build_privacy_mods(args: argparse.Namespace) -> List[Any]:
    """Build a list of Flower client mods from parsed CLI args.

    Returns a list of callables suitable for passing to
    ``ClientApp(client_fn=..., mods=mods)``.

    Also sets the appropriate ``SFL_*`` environment variables for
    server-side DP, aggregation strategy, and DP-SGD so that the
    server and client code can read them at runtime.
    """
    client_mods: List[Any] = []

    # ── Checkpointing env vars ──────────────────────────────────────
    if getattr(args, "checkpoint_dir", None) is not None:
        os.environ["SFL_CHECKPOINT_DIR"] = args.checkpoint_dir
        if getattr(args, "resume", False):
            os.environ["SFL_RESUME"] = "true"

    # ── Differential Privacy env vars ────────────────────────────────
    if args.dp:
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_NOISE"] = str(args.dp_noise)
        os.environ["SFL_DP_CLIP"] = str(args.dp_clip)
        os.environ["SFL_DP_MODE"] = args.dp_mode
        os.environ["SFL_DP_DELTA"] = str(args.dp_delta)
        os.environ["SFL_DP_MAX_EPSILON"] = str(args.dp_max_epsilon)
        if args.dp_adaptive_clip:
            os.environ["SFL_DP_ADAPTIVE_CLIP"] = "true"
            os.environ["SFL_DP_TARGET_QUANTILE"] = str(args.dp_target_quantile)
            os.environ["SFL_DP_CLIP_LR"] = str(args.dp_clip_lr)
            # Default to private quantile tracking (0.1) unless explicitly set
            quantile_noise = (
                args.dp_quantile_noise if args.dp_quantile_noise is not None else 0.1
            )
            os.environ["SFL_DP_QUANTILE_NOISE"] = str(quantile_noise)

        os.environ["SFL_DP_ACCOUNTING_BACKEND"] = args.dp_accounting_backend
        if args.dp_shuffle:
            os.environ["SFL_DP_SHUFFLE"] = "true"

        if args.dp_mode == "client":
            from flwr.client.mod import fixedclipping_mod
            client_mods.append(fixedclipping_mod)

    # ── Percentile Privacy ───────────────────────────────────────────
    if args.percentile_privacy is not None:
        from sfl.privacy.filters import make_percentile_privacy_mod
        client_mods.append(
            make_percentile_privacy_mod(
                args.percentile_privacy, args.percentile_gamma, args.percentile_noise,
                epsilon=args.percentile_epsilon, delta=args.percentile_delta,
            )
        )

    # ── SVT Privacy ──────────────────────────────────────────────────
    if args.svt_privacy:
        from sfl.privacy.filters import make_svt_privacy_mod
        client_mods.append(
            make_svt_privacy_mod(
                fraction=args.svt_fraction, epsilon=args.svt_epsilon,
                optimal_budget=not args.svt_no_optimal,
                pre_screen_ratio=args.svt_prescreen,
            )
        )

    # ── Exclude Layers ───────────────────────────────────────────────
    if args.exclude_layers:
        from sfl.privacy.filters import make_exclude_vars_mod
        indices = [int(x.strip()) for x in args.exclude_layers.split(",")]
        client_mods.append(make_exclude_vars_mod(exclude_indices=indices))

    # ── Gradient Compression ─────────────────────────────────────────
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

    # ── Per-layer Clipping ───────────────────────────────────────────
    if args.per_layer_clip is not None:
        from sfl.privacy.adaptive_clip import make_per_layer_clip_mod
        clip_map = None
        if args.per_layer_clip_map:
            clip_map = {int(k): v for k, v in json.loads(args.per_layer_clip_map).items()}
        client_mods.append(
            make_per_layer_clip_mod(
                clip_norms=clip_map,
                default_clip=args.per_layer_clip,
            )
        )

    # ── Aggregation Strategy env vars ────────────────────────────────
    os.environ["SFL_AGGREGATION"] = args.aggregation
    if args.aggregation == "krum":
        os.environ["SFL_KRUM_BYZANTINE"] = str(args.krum_byzantine)
    elif args.aggregation == "trimmed-mean":
        os.environ["SFL_TRIM_RATIO"] = str(args.trim_ratio)
        os.environ["SFL_FFL_THRESHOLD"] = str(args.ffl_threshold)
        os.environ["SFL_FFL_WEIGHTED"] = str(not args.ffl_no_weighted).lower()

    # ── Per-example DP-SGD env vars ──────────────────────────────────
    if args.dpsgd:
        os.environ["SFL_DPSGD_ENABLED"] = "true"
        os.environ["SFL_DPSGD_CLIP"] = str(args.dpsgd_clip)
        os.environ["SFL_DPSGD_NOISE"] = str(args.dpsgd_noise)
        os.environ["SFL_DPSGD_DELTA"] = str(args.dpsgd_delta)
        if args.dpsgd_autoclip:
            os.environ["SFL_DPSGD_AUTOCLIP"] = "true"
        if args.dpsgd_ghost:
            os.environ["SFL_DPSGD_GHOST"] = "true"

    # ── Metrics ───────────────────────────────────────────────────────
    if getattr(args, "metrics_dir", None) is not None:
        os.environ["SFL_METRICS_DIR"] = args.metrics_dir
        os.environ["SFL_METRICS_FORMAT"] = getattr(args, "metrics_format", "csv")

    # ── Partial Freezing ─────────────────────────────────────────────
    if args.freeze_layers is not None:
        from sfl.privacy.filters import make_partial_freeze_mod
        trainable = [int(x.strip()) for x in args.freeze_layers.split(",")]
        client_mods.append(make_partial_freeze_mod(trainable_indices=trainable))

    # ── SecAgg (must be last mod) ────────────────────────────────────
    if args.secagg:
        from flwr.client.mod import secaggplus_mod
        client_mods.append(secaggplus_mod)

    return client_mods


# ── Valid values for SFL_* env vars ──────────────────────────────────

_VALID_AGGREGATIONS = {"fedavg", "krum", "trimmed-mean", "foundation-fl"}
_VALID_DP_MODES = {"server", "client"}
_VALID_ACCOUNTING_BACKENDS = {"pld", "prv"}


def _check_float(name: str, *, positive: bool = False, strict_positive: bool = False) -> None:
    """Validate that an env var is a parseable float with optional constraints."""
    value = os.environ.get(name)
    if value is None:
        return
    try:
        f = float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"{name}={value!r} is not a valid float. "
            f"Set it to a numeric value (e.g. {name}=1.0)."
        )
    if positive and f < 0:
        raise ValueError(
            f"{name}={value} must be non-negative (>= 0)."
        )
    if strict_positive and f <= 0:
        raise ValueError(
            f"{name}={value} must be strictly positive (> 0)."
        )


def _check_int(name: str, *, positive: bool = False) -> None:
    """Validate that an env var is a parseable int with optional constraints."""
    value = os.environ.get(name)
    if value is None:
        return
    try:
        i = int(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"{name}={value!r} is not a valid integer. "
            f"Set it to a whole number (e.g. {name}=1)."
        )
    if positive and i < 1:
        raise ValueError(
            f"{name}={value} must be a positive integer (>= 1)."
        )


def _check_bool(name: str) -> None:
    """Validate that an env var looks like a boolean."""
    value = os.environ.get(name)
    if value is None:
        return
    if value.lower() not in {"true", "false", "1", "0", ""}:
        raise ValueError(
            f"{name}={value!r} is not a valid boolean. "
            f"Use 'true' or 'false'."
        )


def _check_choice(name: str, choices: set) -> None:
    """Validate that an env var is one of the allowed values."""
    value = os.environ.get(name)
    if value is None:
        return
    if value.lower() not in choices:
        sorted_choices = sorted(choices)
        raise ValueError(
            f"{name}={value!r} is not valid. "
            f"Must be one of: {', '.join(sorted_choices)}."
        )


def _check_probability(name: str, *, exclusive: bool = True) -> None:
    """Validate that an env var is a float in (0, 1) or [0, 1]."""
    value = os.environ.get(name)
    if value is None:
        return
    try:
        f = float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"{name}={value!r} is not a valid float. "
            f"Set it to a value between 0 and 1."
        )
    if exclusive:
        if f <= 0 or f >= 1:
            raise ValueError(
                f"{name}={value} must be in the open interval (0, 1)."
            )
    else:
        if f < 0 or f > 1:
            raise ValueError(
                f"{name}={value} must be in the interval [0, 1]."
            )


def validate_env_vars() -> None:
    """Validate all SFL_* environment variables at startup.

    Catches misconfiguration early with clear error messages instead
    of failing deep in strategy initialization.

    Call this at the beginning of every runner's ``run_flower()``
    function, after ``build_privacy_mods()`` has set the env vars.

    Raises:
        ValueError: If any ``SFL_*`` env var has an invalid value.
    """
    errors: List[str] = []

    validators = [
        # ── DP core ──────────────────────────────────────────────
        lambda: _check_bool("SFL_DP_ENABLED"),
        lambda: _check_float("SFL_DP_NOISE", positive=True),
        lambda: _check_float("SFL_DP_CLIP", strict_positive=True),
        lambda: _check_choice("SFL_DP_MODE", _VALID_DP_MODES),
        lambda: _check_probability("SFL_DP_DELTA"),
        lambda: _check_float("SFL_DP_MAX_EPSILON", strict_positive=True),

        # ── DP adaptive clipping ─────────────────────────────────
        lambda: _check_bool("SFL_DP_ADAPTIVE_CLIP"),
        lambda: _check_probability("SFL_DP_TARGET_QUANTILE"),
        lambda: _check_float("SFL_DP_CLIP_LR", strict_positive=True),
        lambda: _check_float("SFL_DP_QUANTILE_NOISE", positive=True),

        # ── DP accounting ────────────────────────────────────────
        lambda: _check_choice("SFL_DP_ACCOUNTING_BACKEND", _VALID_ACCOUNTING_BACKENDS),
        lambda: _check_bool("SFL_DP_SHUFFLE"),

        # ── Aggregation ──────────────────────────────────────────
        lambda: _check_choice("SFL_AGGREGATION", _VALID_AGGREGATIONS),
        lambda: _check_int("SFL_KRUM_BYZANTINE", positive=True),
        lambda: _check_probability("SFL_TRIM_RATIO"),
        lambda: _check_float("SFL_FFL_THRESHOLD", positive=True),
        lambda: _check_bool("SFL_FFL_WEIGHTED"),

        # ── DP-SGD (Opacus) ─────────────────────────────────────
        lambda: _check_bool("SFL_DPSGD_ENABLED"),
        lambda: _check_float("SFL_DPSGD_CLIP", strict_positive=True),
        lambda: _check_float("SFL_DPSGD_NOISE", positive=True),
        lambda: _check_probability("SFL_DPSGD_DELTA"),
        lambda: _check_bool("SFL_DPSGD_AUTOCLIP"),
        lambda: _check_bool("SFL_DPSGD_GHOST"),
    ]

    for validator in validators:
        try:
            validator()
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        msg = "Invalid SFL_* environment variable(s):\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)
