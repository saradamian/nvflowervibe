"""
Auto-build client mods from SFL_* environment variables.

In distributed NVFlare mode, the runner process that calls
build_privacy_mods() is NOT the same process that runs the Flower
ClientApp. NVFlare spawns separate flower-supernode processes on
each site.

This module reconstructs the same mods list from the SFL_* env vars
that build_privacy_mods() originally set. It is called at ClientApp
construction time in sfl/esm2/__init__.py and sfl/llm/__init__.py.

The env vars are propagated to NVFlare sites via FlowerJob's extra_env
parameter.
"""

from __future__ import annotations

import json
import os
from typing import Any, List

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


def auto_build_client_mods() -> List[Any]:
    """Build Flower client mods from SFL_* environment variables.

    Mirrors the logic in ``build_privacy_mods()`` but reads from env
    vars instead of argparse. Used when the Flower app is started by
    NVFlare (or any external launcher) rather than a runner script.

    Returns:
        List of Flower client mod callables, ready for
        ``ClientApp(client_fn=..., mods=mods)``.
    """
    mods: List[Any] = []

    # ── Client-side DP clipping ──────────────────────────────────────
    if (
        os.environ.get("SFL_DP_ENABLED", "").lower() == "true"
        and os.environ.get("SFL_DP_MODE", "server").lower() == "client"
    ):
        from flwr.client.mod import fixedclipping_mod
        mods.append(fixedclipping_mod)
        logger.info("auto_mods: added fixedclipping_mod (client-side DP)")

    # ── Percentile Privacy ───────────────────────────────────────────
    pct = os.environ.get("SFL_PERCENTILE_PRIVACY")
    if pct is not None:
        from sfl.privacy.filters import make_percentile_privacy_mod
        mods.append(make_percentile_privacy_mod(
            top_percentile=int(pct),
            gamma=float(os.environ.get("SFL_PERCENTILE_GAMMA", "0.01")),
            noise_scale=float(os.environ.get("SFL_PERCENTILE_NOISE", "0.0")),
            epsilon=float(os.environ.get("SFL_PERCENTILE_EPSILON", "0.0")),
            delta=float(os.environ.get("SFL_PERCENTILE_DELTA", "1e-5")),
        ))
        logger.info("auto_mods: added percentile_privacy (top %s%%)", pct)

    # ── SVT Privacy ──────────────────────────────────────────────────
    if os.environ.get("SFL_SVT_PRIVACY", "").lower() == "true":
        from sfl.privacy.filters import make_svt_privacy_mod
        mods.append(make_svt_privacy_mod(
            fraction=float(os.environ.get("SFL_SVT_FRACTION", "0.1")),
            epsilon=float(os.environ.get("SFL_SVT_EPSILON", "0.1")),
            optimal_budget=os.environ.get("SFL_SVT_OPTIMAL", "true").lower() == "true",
            pre_screen_ratio=float(os.environ.get("SFL_SVT_PRESCREEN", "1.0")),
        ))
        logger.info("auto_mods: added svt_privacy")

    # ── Exclude Layers ───────────────────────────────────────────────
    exclude = os.environ.get("SFL_EXCLUDE_LAYERS")
    if exclude:
        from sfl.privacy.filters import make_exclude_vars_mod
        indices = [int(x.strip()) for x in exclude.split(",")]
        mods.append(make_exclude_vars_mod(exclude_indices=indices))
        logger.info("auto_mods: added exclude_vars (indices=%s)", indices)

    # ── Gradient Compression ─────────────────────────────────────────
    compress = os.environ.get("SFL_COMPRESS_RATIO")
    if compress is not None:
        from sfl.privacy.filters import make_gradient_compression_mod
        mods.append(make_gradient_compression_mod(
            compression_ratio=float(compress),
            noise_scale=float(os.environ.get("SFL_COMPRESS_NOISE", "0.01")),
            use_random_mask=os.environ.get("SFL_COMPRESS_TOPK", "").lower() != "true",
            error_feedback=os.environ.get("SFL_COMPRESS_ERROR_FEEDBACK", "").lower() == "true",
        ))
        logger.info("auto_mods: added gradient_compression (ratio=%s)", compress)

    # ── Per-layer Clipping ───────────────────────────────────────────
    plc = os.environ.get("SFL_PER_LAYER_CLIP")
    if plc is not None:
        from sfl.privacy.adaptive_clip import make_per_layer_clip_mod
        clip_map = None
        clip_map_str = os.environ.get("SFL_PER_LAYER_CLIP_MAP")
        if clip_map_str:
            clip_map = {int(k): v for k, v in json.loads(clip_map_str).items()}
        mods.append(make_per_layer_clip_mod(
            clip_norms=clip_map,
            default_clip=float(plc),
        ))
        logger.info("auto_mods: added per_layer_clip (default=%s)", plc)

    # ── Partial Freezing ─────────────────────────────────────────────
    freeze = os.environ.get("SFL_FREEZE_LAYERS")
    if freeze:
        from sfl.privacy.filters import make_partial_freeze_mod
        trainable = [int(x.strip()) for x in freeze.split(",")]
        mods.append(make_partial_freeze_mod(trainable_indices=trainable))
        logger.info("auto_mods: added partial_freeze (trainable=%s)", trainable)

    # ── SecAgg (must be last) ────────────────────────────────────────
    if os.environ.get("SFL_SECAGG_ENABLED", "").lower() == "true":
        from flwr.client.mod import secaggplus_mod
        mods.append(secaggplus_mod)
        logger.info("auto_mods: added secaggplus_mod")

    if mods:
        logger.info("auto_mods: %d client mod(s) built from env vars", len(mods))

    return mods
