"""
Privacy-preserving Flower client mods.

Ports NVFlare's server-enforced DXO privacy filters to work as Flower
client mods. Each mod intercepts the client's FitRes after local
training and transforms the parameters before they reach the server.

Three filters are provided:

1. **PercentilePrivacy** — only shares weight diffs in the top
   percentile by absolute value, zeros the rest. Based on
   Shokri & Shmatikov, CCS '15.

2. **SVTPrivacy** — Sparse Vector Technique differential privacy.
   Uses a noisy threshold to select which parameters to share,
   adds calibrated Laplace noise. Provides formal ε-DP guarantees.

3. **ExcludeVars** — zeros out parameter arrays at specified indices,
   preventing those layers from leaving the client. Useful for
   keeping embedding layers private.

Usage:
    from sfl.privacy.filters import (
        make_percentile_privacy_mod,
        make_svt_privacy_mod,
        make_exclude_vars_mod,
    )

    mods = [make_percentile_privacy_mod(percentile=10, gamma=0.01)]
    client_app = ClientApp(client_fn=client_fn, mods=mods)
"""

from dataclasses import dataclass
from logging import INFO, WARNING
from typing import Callable, List, Optional

import numpy as np
from flwr.client.typing import ClientAppCallable
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import recorddict_compat as compat
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message

from sfl.utils.rng import secure_rng


# ── PercentilePrivacy ───────────────────────────────────────────────────────


@dataclass
class PercentilePrivacyConfig:
    """Configuration for percentile privacy filter.

    Args:
        percentile: Only abs diffs above this percentile are kept (0–100).
        gamma: Upper limit to clip abs values of weight diffs.
        noise_scale: Gaussian noise std as a multiple of gamma.
            0 = no noise (legacy, NOT private). >0 adds noise
            after clipping. Overridden by epsilon if set.
        epsilon: If set, auto-calibrate noise to this (ε,δ)-DP
            guarantee via the analytic Gaussian mechanism
            (Balle & Wang 2018). Overrides noise_scale.
        delta: Target δ for (ε,δ)-DP (used with epsilon).
    """
    percentile: int = 10
    gamma: float = 0.01
    noise_scale: float = 0.0
    epsilon: float = 0.0
    delta: float = 1e-5


def make_percentile_privacy_mod(
    percentile: int = 10,
    gamma: float = 0.01,
    noise_scale: float = 0.0,
    epsilon: float = 0.0,
    delta: float = 1e-5,
) -> Callable[[Message, Context, ClientAppCallable], Message]:
    """Create a Flower client mod that applies percentile privacy.

    Only weight diffs in the top ``percentile`` by absolute magnitude
    are shared; smaller diffs are zeroed. Remaining values are clipped
    to [-gamma, gamma].

    When ``epsilon > 0``, noise is auto-calibrated to the target
    (ε,δ)-DP guarantee via the Gaussian mechanism. When only
    ``noise_scale > 0`` is provided, uncalibrated noise is added
    with a warning. With neither, this is bandwidth reduction only.
    """
    cfg = PercentilePrivacyConfig(
        percentile=percentile, gamma=gamma,
        noise_scale=noise_scale, epsilon=epsilon, delta=delta,
    )

    # Pre-import calibration function if epsilon is set.
    # Actual calibration is deferred to per-round application because
    # the L2 sensitivity depends on the number of surviving elements K:
    # Δ₂ = gamma * √K, not just gamma.
    _calibrate_fn = None
    if cfg.epsilon > 0:
        from sfl.privacy.dp import calibrate_gaussian_sigma
        _calibrate_fn = calibrate_gaussian_sigma
    elif cfg.noise_scale > 0:
        log(WARNING,
            "PercentilePrivacy: noise_scale=%.4f is uncalibrated (no formal "
            "(ε,δ) guarantee). Use --percentile-epsilon for calibrated noise.",
            cfg.noise_scale)

    def percentile_privacy_mod(
        msg: Message, ctxt: Context, call_next: ClientAppCallable,
    ) -> Message:
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        out_msg = call_next(msg, ctxt)
        if out_msg.has_error():
            return out_msg

        fit_res = compat.recorddict_to_fitres(out_msg.content, keep_input=True)
        params = parameters_to_ndarrays(fit_res.parameters)

        # Compute percentile cutoff across all parameters
        all_abs = np.concatenate([np.abs(p.ravel()) for p in params])
        cutoff = np.percentile(all_abs, cfg.percentile)

        # Count surviving elements for correct L2 sensitivity
        k_surviving = int(np.sum(all_abs >= cutoff))

        # Calibrate noise per-round using correct L2 sensitivity = γ√K
        effective_noise_scale = cfg.noise_scale
        if _calibrate_fn is not None and k_surviving > 0:
            import math
            sensitivity = cfg.gamma * math.sqrt(k_surviving)
            sigma = _calibrate_fn(cfg.epsilon, cfg.delta, sensitivity)
            effective_noise_scale = sigma / cfg.gamma
            log(INFO,
                "PercentilePrivacy: calibrated σ=%.4f for (ε=%.2f, δ=%.1e), "
                "K=%d, Δ₂=%.4f",
                sigma, cfg.epsilon, cfg.delta, k_surviving, sensitivity)

        filtered = []
        for p in params:
            arr = p.copy()
            if arr.ndim == 0:
                filtered.append(arr)
                continue
            # Zero out values below cutoff
            mask = (arr > -cutoff) & (arr < cutoff)
            arr[mask] = 0.0
            # Clip remaining to gamma
            arr = np.clip(arr, -cfg.gamma, cfg.gamma)
            # Add Gaussian noise if configured
            if effective_noise_scale > 0:
                _rng = secure_rng()
                noise = _rng.normal(0, effective_noise_scale * cfg.gamma, size=arr.shape)
                arr = np.clip(arr + noise, -cfg.gamma, cfg.gamma)
            filtered.append(arr)

        if effective_noise_scale == 0:
            log(WARNING,
                "PercentilePrivacy with noise_scale=0 provides NO formal privacy "
                "guarantee. It reduces bandwidth but does NOT prevent gradient "
                "inversion attacks. Use --percentile-noise > 0 or --svt-privacy.")

        log(INFO, "percentile_privacy_mod: cutoff=%.6f, percentile=%d, noise=%.4f",
            cutoff, cfg.percentile, effective_noise_scale)

        fit_res.parameters = ndarrays_to_parameters(filtered)
        out_msg.content = compat.fitres_to_recorddict(fit_res, keep_input=True)
        return out_msg

    return percentile_privacy_mod


# ── SVTPrivacy ──────────────────────────────────────────────────────────────


@dataclass
class SVTPrivacyConfig:
    """Configuration for SVT differential privacy filter.

    Args:
        fraction: Fraction of parameters to upload (0–1).
        epsilon: Privacy budget. Lower = more private, noisier.
        gamma: Clipping bound (L1 sensitivity) for parameter values.
        tau: Base threshold for SVT selection.
        optimal_budget: Use numerically-optimal budget split
            (Lyu et al. 2017) instead of the standard ε/2 + ε/(2c).
        pre_screen_ratio: Run SVT only on this fraction of
            parameters (top by magnitude). 1.0 = no pre-screening.
    """
    fraction: float = 0.1
    epsilon: float = 0.1
    gamma: float = 1e-5
    tau: float = 1e-6
    optimal_budget: bool = True
    pre_screen_ratio: float = 1.0


_SVT_MAX_ITERATIONS = 100


def make_svt_privacy_mod(
    fraction: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1e-5,
    tau: float = 1e-6,
    optimal_budget: bool = True,
    pre_screen_ratio: float = 1.0,
    **_kwargs,
) -> Callable[[Message, Context, ClientAppCallable], Message]:
    """Create a Flower client mod that applies SVT differential privacy.

    Sparse Vector Technique (Dwork & Roth 2014, Theorem 3.25):
    selects a fraction of parameter values using a noisy threshold,
    adds calibrated Laplace noise to accepted values, and zeros the rest.

    When ``optimal_budget=True`` (default), uses the numerically-optimal
    budget split from Lyu et al. (2017, "Understanding the Sparse Vector
    Technique for Differential Privacy"):
        α* = 1 / (1 + √c)
        ε_threshold = α* · ε
        ε_per_query = (1 - α*) · ε / c

    When ``pre_screen_ratio < 1.0``, only the top parameters by magnitude
    are candidates for SVT, reducing c and thus the per-query noise.
    """
    cfg = SVTPrivacyConfig(
        fraction=fraction, epsilon=epsilon,
        gamma=gamma, tau=tau,
        optimal_budget=optimal_budget,
        pre_screen_ratio=pre_screen_ratio,
    )

    def svt_privacy_mod(
        msg: Message, ctxt: Context, call_next: ClientAppCallable,
    ) -> Message:
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        out_msg = call_next(msg, ctxt)
        if out_msg.has_error():
            return out_msg

        fit_res = compat.recorddict_to_fitres(out_msg.content, keep_input=True)
        params = parameters_to_ndarrays(fit_res.parameters)

        # Flatten all parameters
        delta_w = np.concatenate([p.ravel().astype(np.float64) for p in params])
        n_total = delta_w.size
        n_upload = int(min(np.ceil(n_total * cfg.fraction), n_total))

        # Pre-screen: only consider top params by magnitude
        if cfg.pre_screen_ratio < 1.0:
            n_screen = max(n_upload, int(np.ceil(n_total * cfg.pre_screen_ratio)))
            candidate_idx = np.argpartition(
                np.abs(delta_w), -n_screen
            )[-n_screen:]
        else:
            candidate_idx = np.arange(n_total)

        c = candidate_idx.size  # number of SVT queries

        # Budget split
        if cfg.optimal_budget and c > 1:
            # Numerically-optimal split (Lyu et al. 2017)
            import math
            alpha = 1.0 / (1.0 + math.sqrt(c))
            eps_1 = alpha * cfg.epsilon
            eps_2 = (1.0 - alpha) * cfg.epsilon / c
        else:
            # Standard split (Dwork & Roth, Theorem 3.25)
            eps_1 = cfg.epsilon / 2.0
            eps_2 = cfg.epsilon / (2.0 * c) if c > 0 else cfg.epsilon

        # Noisy threshold
        _rng = secure_rng()
        threshold = cfg.tau + _rng.laplace(scale=cfg.gamma / eps_1)

        # Per-query noise scale (same for selection and output)
        query_scale = cfg.gamma / eps_2

        # Single-pass SVT: each candidate is queried exactly once.
        # Re-querying rejected candidates with fresh noise would violate
        # the one-shot SVT ε-DP proof (Dwork & Roth 2014, Theorem 3.25).
        clipped_w = np.abs(np.clip(delta_w, -cfg.gamma, cfg.gamma))

        nu_i = _rng.laplace(scale=query_scale, size=candidate_idx.shape)
        above = (clipped_w[candidate_idx] + nu_i) >= threshold
        accepted = candidate_idx[above].tolist()

        if len(accepted) < n_upload:
            log(WARNING, "SVT: accepted only %d/%d params in single pass "
                "(consider increasing epsilon or pre_screen_ratio)",
                len(accepted), n_upload)

        # Sample exactly n_upload if we got more
        if len(accepted) > n_upload:
            accepted = list(_rng.choice(accepted, size=n_upload, replace=False))

        # Add output noise calibrated to the same per-query budget
        output_noise = _rng.laplace(scale=query_scale, size=len(accepted))
        delta_w_out = np.zeros_like(delta_w)
        accepted_arr = np.array(accepted, dtype=np.intp)
        delta_w_out[accepted_arr] = np.clip(
            delta_w[accepted_arr] + output_noise, -cfg.gamma, cfg.gamma,
        )

        log(
            INFO,
            "svt_privacy_mod: selected %d/%d params (fraction=%.2f, eps=%.3f)",
            len(accepted), n_total, cfg.fraction, cfg.epsilon,
        )

        # Reshape back to original parameter shapes
        filtered = []
        offset = 0
        for p in params:
            size = p.size
            filtered.append(
                delta_w_out[offset:offset + size].reshape(p.shape).astype(p.dtype)
            )
            offset += size

        fit_res.parameters = ndarrays_to_parameters(filtered)
        out_msg.content = compat.fitres_to_recorddict(fit_res, keep_input=True)
        return out_msg

    return svt_privacy_mod


# ── ExcludeVars ─────────────────────────────────────────────────────────────


def make_exclude_vars_mod(
    exclude_indices: Optional[List[int]] = None,
) -> Callable[[Message, Context, ClientAppCallable], Message]:
    """Create a Flower client mod that zeros out specified parameter arrays.

    In Flower, parameters are a flat list of NDArrays (no names). This
    mod zeros out arrays at the given indices, preventing those layers
    from leaving the client.

    For ESM2, typical indices to exclude:
    - 0, 1: token embeddings (word_embeddings.weight, position_embeddings)
    - Last 2: LM head (decoder.weight, decoder.bias)

    Args:
        exclude_indices: List of parameter array indices to zero out.
            If None or empty, no parameters are excluded.
    """
    indices = set(exclude_indices or [])

    def exclude_vars_mod(
        msg: Message, ctxt: Context, call_next: ClientAppCallable,
    ) -> Message:
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        out_msg = call_next(msg, ctxt)
        if out_msg.has_error():
            return out_msg

        if not indices:
            return out_msg

        fit_res = compat.recorddict_to_fitres(out_msg.content, keep_input=True)
        params = parameters_to_ndarrays(fit_res.parameters)

        filtered = []
        n_excluded = 0
        for i, p in enumerate(params):
            if i in indices:
                filtered.append(np.zeros_like(p))
                n_excluded += 1
            else:
                filtered.append(p)

        log(WARNING,
            "ExcludeVars zeroed %d/%d layers but does NOT guarantee those "
            "layers' information won't leak through other shared layers. "
            "Combine with DP for formal guarantees.",
            n_excluded, len(params))

        fit_res.parameters = ndarrays_to_parameters(filtered)
        out_msg.content = compat.fitres_to_recorddict(fit_res, keep_input=True)
        return out_msg

    return exclude_vars_mod


# ── GradientCompression ────────────────────────────────────────────────────


@dataclass
class GradientCompressionConfig:
    """Configuration for gradient compression defense.

    Args:
        compression_ratio: Fraction of gradient values to keep (0–1).
        noise_scale: Gaussian noise std relative to the L2 norm
            of surviving values, divided by sqrt(K).
            Ignored when ``epsilon`` is set.
        use_random_mask: Use magnitude-weighted random masking
            instead of deterministic TopK.
        epsilon: Target ε for (ε,δ)-DP calibrated noise.
            When set, ``delta`` and ``clipping_norm`` are required,
            and noise is computed via PLD calibration instead of
            the heuristic ``noise_scale``.
        delta: Target δ for (ε,δ)-DP. Required when ``epsilon``
            is set.
        clipping_norm: L2 sensitivity bound. Required when
            ``epsilon`` is set.
        error_feedback: Enable error feedback (FedSparQ-style).
            Compression residuals are accumulated across rounds and
            added back to the next round's update before sparsification.
            This recovers convergence lost to aggressive compression
            without additional communication cost.
    """
    compression_ratio: float = 0.1
    noise_scale: float = 0.01
    use_random_mask: bool = True
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    clipping_norm: Optional[float] = None
    error_feedback: bool = False


def make_gradient_compression_mod(
    compression_ratio: float = 0.1,
    noise_scale: float = 0.01,
    use_random_mask: bool = True,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    clipping_norm: Optional[float] = None,
    error_feedback: bool = False,
) -> Callable[[Message, Context, ClientAppCallable], Message]:
    """Create a Flower client mod that compresses gradient updates.

    Keeps only ``compression_ratio`` fraction of gradient values (TopK or
    random masking), zeros the rest, and adds noise to surviving values.

    When ``epsilon`` is set, noise is calibrated via PLD (Balle & Wang
    2018) to satisfy (ε,δ)-DP with sensitivity = ``clipping_norm``.
    Otherwise falls back to the heuristic ``noise_scale``.

    Random masking (default) selects values with probability proportional
    to magnitude, preventing deterministic TopK from leaking which
    parameters are consistently large.

    **Important**: When ``epsilon`` is set, ``use_random_mask`` is forced
    to ``False`` because data-dependent masking is an adaptive mechanism
    whose DP cost is not accounted for in the noise calibration. Only
    deterministic TopK gives a data-independent mask compatible with the
    calibrated (ε,δ)-DP guarantee.
    """
    # Force deterministic TopK in DP mode — data-dependent random masking
    # is an adaptive mechanism whose privacy cost is not composed into
    # the calibrated noise. Using it would invalidate the ε claim.
    if epsilon is not None and use_random_mask:
        log(WARNING,
            "GradientCompression: forcing use_random_mask=False because "
            "epsilon is set. Data-dependent masking would invalidate the "
            "(ε,δ)-DP guarantee.")
        use_random_mask = False

    cfg = GradientCompressionConfig(
        compression_ratio=compression_ratio,
        noise_scale=noise_scale,
        use_random_mask=use_random_mask,
        epsilon=epsilon,
        delta=delta,
        clipping_norm=clipping_norm,
        error_feedback=error_feedback,
    )

    # Error feedback state: accumulated residual from previous rounds
    _error_state: dict = {"residual": None}

    # Calibrate noise if (ε,δ) are provided
    if cfg.epsilon is not None:
        if cfg.delta is None or cfg.clipping_norm is None:
            raise ValueError(
                "delta and clipping_norm are required when epsilon is set"
            )
        from sfl.privacy.dp import calibrate_gaussian_sigma
        sigma = calibrate_gaussian_sigma(
            epsilon=cfg.epsilon,
            delta=cfg.delta,
            sensitivity=cfg.clipping_norm,
        )
        cfg._calibrated_sigma = sigma
        log(INFO,
            "GradientCompression: calibrated σ=%.4f for ε=%.2f, δ=%.1e, "
            "C=%.2f", sigma, cfg.epsilon, cfg.delta, cfg.clipping_norm)
    else:
        cfg._calibrated_sigma = None

    def gradient_compression_mod(
        msg: Message, ctxt: Context, call_next: ClientAppCallable,
    ) -> Message:
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        out_msg = call_next(msg, ctxt)
        if out_msg.has_error():
            return out_msg

        fit_res = compat.recorddict_to_fitres(out_msg.content, keep_input=True)
        params = parameters_to_ndarrays(fit_res.parameters)

        flat = np.concatenate([p.ravel().astype(np.float64) for p in params])

        # Error feedback: add accumulated residual from previous rounds
        if cfg.error_feedback and _error_state["residual"] is not None:
            if _error_state["residual"].shape == flat.shape:
                flat = flat + _error_state["residual"]

        n = flat.size
        k = max(1, int(np.ceil(n * cfg.compression_ratio)))

        if cfg.use_random_mask:
            # Magnitude-weighted random selection
            abs_flat = np.abs(flat)
            total = abs_flat.sum()
            if total > 0:
                probs = abs_flat / total
            else:
                probs = np.ones(n) / n
            selected = secure_rng().choice(n, size=k, replace=False, p=probs)
        else:
            # Deterministic TopK
            selected = np.argpartition(np.abs(flat), -k)[-k:]

        # Build sparse output
        out = np.zeros_like(flat)
        out[selected] = flat[selected]

        # Error feedback: save residual (what was zeroed) for next round
        if cfg.error_feedback:
            _error_state["residual"] = flat - out

        # Add noise to surviving values
        if k > 0:
            if cfg._calibrated_sigma is not None:
                # Calibrated (ε,δ)-DP noise
                out[selected] += secure_rng().normal(
                    scale=cfg._calibrated_sigma, size=k,
                )
            elif cfg.noise_scale > 0:
                l2 = np.linalg.norm(out[selected])
                sigma = cfg.noise_scale * l2 / np.sqrt(k)
                out[selected] += secure_rng().normal(scale=sigma, size=k)

        log(INFO,
            "gradient_compression: kept %d/%d values (%.1f%%), random=%s",
            k, n, 100.0 * k / n, cfg.use_random_mask)

        # Reshape back
        compressed = []
        offset = 0
        for p in params:
            size = p.size
            compressed.append(
                out[offset:offset + size].reshape(p.shape).astype(p.dtype)
            )
            offset += size

        fit_res.parameters = ndarrays_to_parameters(compressed)
        out_msg.content = compat.fitres_to_recorddict(fit_res, keep_input=True)
        return out_msg

    return gradient_compression_mod


# ── Partial Freezing (Lambda-SecAgg) ──────────────────────────────────────


def make_partial_freeze_mod(
    trainable_indices: Optional[List[int]] = None,
) -> Callable[[Message, Context, ClientAppCallable], Message]:
    """Create a Flower client mod that strips frozen layers from updates.

    When fine-tuning large models (e.g., ESM2), most layers are frozen and
    only a subset are trained. Sending full-sized updates through SecAgg
    wastes computation on zero-valued frozen layers. This mod removes
    frozen layers from the outgoing update, reducing SecAgg encryption
    and communication overhead proportionally.

    The server must use the corresponding ``make_partial_freeze_strategy``
    wrapper to restore full-sized parameter arrays before aggregation.

    This implements "Lambda-SecAgg" (Bonawitz et al., 2019 extension):
    only the trainable subset (λ) passes through the secure aggregation
    protocol, cutting cost from O(P) to O(λ·P).

    Args:
        trainable_indices: List of parameter array indices that are
            trainable (not frozen). Only these are kept in the update.
            If None, all parameters are sent (no-op).
    """
    _indices = set(trainable_indices) if trainable_indices is not None else None

    def partial_freeze_mod(
        msg: Message, ctxt: Context, call_next: ClientAppCallable,
    ) -> Message:
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        out_msg = call_next(msg, ctxt)
        if out_msg.has_error():
            return out_msg

        if _indices is None:
            return out_msg

        fit_res = compat.recorddict_to_fitres(out_msg.content, keep_input=True)
        params = parameters_to_ndarrays(fit_res.parameters)

        # Keep only trainable layers
        filtered = [p for i, p in enumerate(params) if i in _indices]

        n_original = len(params)
        n_kept = len(filtered)
        original_size = sum(p.size for p in params)
        kept_size = sum(p.size for p in filtered)
        reduction = 1.0 - (kept_size / max(original_size, 1))

        log(INFO,
            "partial_freeze: sending %d/%d layers (%d/%d params, %.1f%% reduction)",
            n_kept, n_original, kept_size, original_size, 100.0 * reduction)

        fit_res.parameters = ndarrays_to_parameters(filtered)
        # Store the trainable indices and frozen layer shapes so the server
        # can reconstruct the full parameter array with correct shapes.
        fit_res.metrics["_trainable_indices"] = ",".join(str(i) for i in sorted(_indices))
        frozen_shapes = {
            str(i): ",".join(str(s) for s in p.shape)
            for i, p in enumerate(params)
            if i not in _indices
        }
        fit_res.metrics["_frozen_shapes"] = ";".join(
            f"{k}:{v}" for k, v in frozen_shapes.items()
        )
        out_msg.content = compat.fitres_to_recorddict(fit_res, keep_input=True)
        return out_msg

    return partial_freeze_mod


def make_partial_freeze_strategy(
    strategy,
    trainable_indices: List[int],
):
    """Wrap a Flower strategy to restore full parameter arrays from partial updates.

    Clients using ``make_partial_freeze_mod`` send only trainable layers.
    This wrapper intercepts ``aggregate_fit`` results and inserts zeros for
    frozen layers, restoring the full parameter structure expected by Flower.

    Args:
        strategy: The inner Flower strategy to wrap.
        trainable_indices: The same indices passed to ``make_partial_freeze_mod``.

    Returns:
        The wrapped strategy (mutated in-place).
    """
    _indices = sorted(trainable_indices)
    _original_aggregate_fit = strategy.aggregate_fit

    def _restoring_aggregate_fit(server_round, results, failures):
        # Expand partial updates back to full-size before aggregation
        from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays

        expanded_results = []
        for proxy, res in results:
            partial_params = parameters_to_ndarrays(res.parameters)

            # Determine how many total layers there should be
            # Use _trainable_indices from metrics if available
            idx_str = res.metrics.get("_trainable_indices", "")
            if idx_str:
                indices = [int(x) for x in idx_str.split(",")]
            else:
                indices = _indices

            max_idx = max(indices) if indices else 0
            total_layers = max_idx + 1

            # Parse frozen layer shapes from client metadata
            frozen_shapes: dict = {}
            shapes_str = res.metrics.get("_frozen_shapes", "")
            if shapes_str:
                for entry in shapes_str.split(";"):
                    if ":" not in entry:
                        continue
                    idx_s, shape_s = entry.split(":", 1)
                    frozen_shapes[int(idx_s)] = tuple(
                        int(d) for d in shape_s.split(",")
                    )

            # Build full parameter array with correctly-shaped zeros
            # for frozen layers
            full_params = []
            partial_iter = iter(partial_params)
            for i in range(total_layers):
                if i in set(indices):
                    full_params.append(next(partial_iter))
                elif i in frozen_shapes:
                    full_params.append(
                        np.zeros(frozen_shapes[i], dtype=np.float32)
                    )
                else:
                    raise ValueError(
                        f"Frozen layer {i} has no shape metadata — "
                        f"cannot reconstruct. Ensure client uses "
                        f"make_partial_freeze_mod from this version."
                    )

            # Clean up internal metrics before passing to inner strategy
            metrics = dict(res.metrics)
            metrics.pop("_trainable_indices", None)
            metrics.pop("_frozen_shapes", None)

            expanded_res = FitRes(
                status=res.status,
                parameters=ndarrays_to_parameters(full_params),
                num_examples=res.num_examples,
                metrics=metrics,
            )
            expanded_results.append((proxy, expanded_res))

        return _original_aggregate_fit(server_round, expanded_results, failures)

    strategy.aggregate_fit = _restoring_aggregate_fit
    return strategy
