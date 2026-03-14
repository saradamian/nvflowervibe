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


# ── PercentilePrivacy ───────────────────────────────────────────────────────


@dataclass
class PercentilePrivacyConfig:
    """Configuration for percentile privacy filter.

    Args:
        percentile: Only abs diffs above this percentile are kept (0–100).
        gamma: Upper limit to clip abs values of weight diffs.
        noise_scale: Gaussian noise std as a multiple of gamma.
            0 = no noise (legacy, NOT private). >0 adds calibrated
            noise after clipping for actual privacy.
    """
    percentile: int = 10
    gamma: float = 0.01
    noise_scale: float = 0.0


def make_percentile_privacy_mod(
    percentile: int = 10,
    gamma: float = 0.01,
    noise_scale: float = 0.0,
) -> Callable[[Message, Context, ClientAppCallable], Message]:
    """Create a Flower client mod that applies percentile privacy.

    Only weight diffs in the top ``percentile`` by absolute magnitude
    are shared; smaller diffs are zeroed. Remaining values are clipped
    to [-gamma, gamma].

    With ``noise_scale > 0``, Gaussian noise (std = noise_scale * gamma)
    is added after clipping, providing actual privacy. Without noise,
    this is a bandwidth optimization only — NOT a privacy mechanism.

    Algorithm (Shokri & Shmatikov, CCS '15):
    1. Flatten all parameter diffs into a single vector
    2. Compute the ``percentile``-th percentile of |diffs|
    3. Zero out all diffs below the cutoff
    4. Clip remaining diffs to [-gamma, gamma]
    5. (Optional) Add Gaussian noise, re-clip
    """
    cfg = PercentilePrivacyConfig(percentile=percentile, gamma=gamma, noise_scale=noise_scale)

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
            if cfg.noise_scale > 0:
                noise = np.random.normal(0, cfg.noise_scale * cfg.gamma, size=arr.shape)
                arr = np.clip(arr + noise, -cfg.gamma, cfg.gamma)
            filtered.append(arr)

        if cfg.noise_scale == 0:
            log(WARNING,
                "PercentilePrivacy with noise_scale=0 provides NO formal privacy "
                "guarantee. It reduces bandwidth but does NOT prevent gradient "
                "inversion attacks. Use --percentile-noise > 0 or --svt-privacy.")

        log(INFO, "percentile_privacy_mod: cutoff=%.6f, percentile=%d, noise=%.4f",
            cutoff, cfg.percentile, cfg.noise_scale)

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
    """
    fraction: float = 0.1
    epsilon: float = 0.1
    gamma: float = 1e-5
    tau: float = 1e-6


_SVT_MAX_ITERATIONS = 100


def make_svt_privacy_mod(
    fraction: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1e-5,
    tau: float = 1e-6,
    **_kwargs,
) -> Callable[[Message, Context, ClientAppCallable], Message]:
    """Create a Flower client mod that applies SVT differential privacy.

    Canonical Sparse Vector Technique (Dwork & Roth 2014, Theorem 3.25):
    selects a fraction of parameter values using a noisy threshold,
    adds calibrated Laplace noise to accepted values, and zeros the rest.

    Total privacy cost: ε (split as ε/2 for threshold + ε/(2c) per
    query × c queries, where c = n_upload).

    Reference:
        Dwork & Roth, "The Algorithmic Foundations of Differential
        Privacy" (2014), Section 3.6 (Sparse Vector Technique).

    Algorithm:
    1. Flatten all params, compute upload budget c = fraction * total_params
    2. Split budget: ε₁ = ε/2 for threshold, ε₂ = ε/(2c) per query
    3. Noisy threshold = tau + Laplace(scale = gamma / ε₁)
    4. For each candidate: accept if |value| + Laplace(scale = gamma / ε₂) >= threshold
    5. Add output noise Laplace(scale = gamma / ε₂), clip to [-gamma, gamma]
    6. Zero all non-accepted values
    """
    cfg = SVTPrivacyConfig(
        fraction=fraction, epsilon=epsilon,
        gamma=gamma, tau=tau,
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

        # Canonical SVT budget split (Dwork & Roth, Theorem 3.25)
        eps_1 = cfg.epsilon / 2.0               # threshold noise budget
        eps_2 = cfg.epsilon / (2.0 * n_upload)   # per-query noise budget

        # Noisy threshold
        threshold = cfg.tau + np.random.laplace(scale=cfg.gamma / eps_1)

        # Per-query noise scale (same for selection and output)
        query_scale = cfg.gamma / eps_2

        # Select values above noisy threshold
        clipped_w = np.abs(np.clip(delta_w, -cfg.gamma, cfg.gamma))
        candidate_idx = np.arange(n_total)
        accepted: list = []

        iteration = 0
        while len(accepted) < n_upload and candidate_idx.size > 0 and iteration < _SVT_MAX_ITERATIONS:
            iteration += 1
            nu_i = np.random.laplace(scale=query_scale, size=candidate_idx.shape)
            above = (clipped_w[candidate_idx] + nu_i) >= threshold
            accepted.extend(candidate_idx[above].tolist())
            candidate_idx = candidate_idx[~above]

        if iteration >= _SVT_MAX_ITERATIONS:
            log(WARNING, "SVT: hit iteration cap (%d), selected only %d/%d",
                _SVT_MAX_ITERATIONS, len(accepted), n_upload)

        # Sample exactly n_upload if we got more
        if len(accepted) > n_upload:
            accepted = list(np.random.choice(accepted, size=n_upload, replace=False))

        # Add output noise calibrated to the same per-query budget
        output_noise = np.random.laplace(scale=query_scale, size=len(accepted))
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
