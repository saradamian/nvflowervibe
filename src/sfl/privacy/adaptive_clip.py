"""
Adaptive clipping for differential privacy.

Implements the geometric adaptive clipping algorithm from
Andrew et al. (2021, "Differentially Private Learning with
Adaptive Clipping"). The clipping norm is adjusted each round
toward a target quantile of client update norms.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from sfl.utils.logging import get_logger
from sfl.utils.rng import secure_rng

logger = get_logger(__name__)


@dataclass
class AdaptiveClipConfig:
    """Configuration for adaptive clipping.

    Args:
        target_quantile: Fraction of updates that should be unclipped.
        learning_rate: Step size for geometric clip norm update.
        clip_min: Floor for the clipping norm.
        clip_max: Ceiling for the clipping norm.
        quantile_noise_multiplier: Gaussian noise multiplier for the
            clipped-fraction estimate. Required for DP-private quantile
            tracking (Andrew et al. 2021 §3.2). Sensitivity is 1/n
            (one client flips at most one binary indicator), so noise
            std = quantile_noise_multiplier / n. Set to 0.0 to disable
            (non-private, for testing only).
    """
    target_quantile: float = 0.5
    learning_rate: float = 0.2
    clip_min: float = 0.1
    clip_max: float = 100.0
    quantile_noise_multiplier: float = 0.0


class AdaptiveClipWrapper(Strategy):
    """Wraps a DP strategy to adaptively update the clipping norm.

    After each round, estimates what fraction of client updates exceeded
    the current clip norm. When ``quantile_noise_multiplier > 0``, adds
    calibrated Gaussian noise to the binary clipped/not-clipped indicators
    before computing the fraction, ensuring the quantile estimate itself
    satisfies DP (Andrew et al. 2021 §3.2).

    Adjusts clip norm toward target_quantile using the geometric update
    rule from Andrew et al. (2021).

    The wrapped strategy must have ``clipping_norm`` and
    ``current_round_params`` attributes (as Flower's DP wrappers do).
    """

    def __init__(self, strategy: Strategy, config: AdaptiveClipConfig) -> None:
        super().__init__()
        self.strategy = strategy
        self.config = config
        # Stores the DP event from the most recent quantile query,
        # so the accountant can compose it into the total budget.
        self._last_quantile_dp_event = None

    def __repr__(self) -> str:
        return f"AdaptiveClipWrapper({self.strategy!r})"

    # ── Delegate everything except aggregate_fit ─────────────────────────

    def initialize_parameters(self, client_manager):
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)

    def evaluate(self, server_round, parameters):
        return self.strategy.evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    # ── Core: adaptive clipping in aggregate_fit ─────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Compute update norms, adapt clip, then delegate to inner strategy."""
        clip_norm = getattr(self.strategy, "clipping_norm", None)
        current_params = getattr(self.strategy, "current_round_params", None)

        new_clip = None
        self._last_quantile_dp_event = None

        if clip_norm is not None and current_params and results:
            # Compute L2 norms of un-clipped client updates
            norms = []
            for _, res in results:
                update = parameters_to_ndarrays(res.parameters)
                flat = np.concatenate(
                    [
                        (u - c).ravel()
                        for u, c in zip(update, current_params)
                    ]
                )
                norms.append(float(np.linalg.norm(flat)))

            # Fraction exceeding current clip norm (with DP noise)
            n = len(norms)
            raw_sum = sum(1 for norm in norms if norm > clip_norm)

            cfg = self.config
            if cfg.quantile_noise_multiplier > 0:
                # Per Andrew et al. 2021 §3.2: sensitivity of the sum
                # of binary indicators is 1 (one client changes the sum
                # by at most 1).  Noise std = quantile_noise_multiplier.
                noise = secure_rng().normal(0, cfg.quantile_noise_multiplier)
                noisy_sum = max(0.0, min(float(n), raw_sum + noise))
                fraction_clipped = noisy_sum / n

                # Record the DP cost of this quantile query so the
                # accountant can compose it into the total budget.
                # GaussianDpEvent with noise_multiplier = σ/sensitivity
                # = quantile_noise_multiplier / 1 = quantile_noise_multiplier.
                try:
                    from dp_accounting import dp_event as _dp_event
                    self._last_quantile_dp_event = _dp_event.GaussianDpEvent(
                        noise_multiplier=cfg.quantile_noise_multiplier
                    )
                except ImportError:
                    pass
            else:
                fraction_clipped = raw_sum / n

            # Geometric update (Andrew et al., 2021)
            new_clip = clip_norm * math.exp(
                cfg.learning_rate * (fraction_clipped - cfg.target_quantile)
            )
            new_clip = max(cfg.clip_min, min(cfg.clip_max, new_clip))

            logger.info(
                f"[adaptive-clip] round={server_round} "
                f"clip={clip_norm:.4f}->{new_clip:.4f} "
                f"clipped={fraction_clipped:.0%} "
                f"median_norm={float(np.median(norms)):.4f}"
            )

        # Delegate clipping + noise + aggregation using the CURRENT clip norm.
        # Per Andrew et al. 2021: quantile estimate from round t updates
        # the clip for round t+1, not round t.
        result = self.strategy.aggregate_fit(server_round, results, failures)

        if clip_norm is not None and current_params and results:
            # Apply the new clip norm AFTER this round's aggregation
            self.strategy.clipping_norm = new_clip

        return result


# ── Per-Layer Adaptive Clipping ──────────────────────────────────────────────


def make_per_layer_clip_mod(
    clip_norms: Optional[Dict[int, float]] = None,
    default_clip: float = 1.0,
) -> "Callable":
    """Create a Flower client mod that clips each parameter layer independently.

    For transformers like ESM2, embedding/head layers have much larger
    norms than attention layers. A single global clip wastes budget on
    small layers and under-clips large layers. Per-layer clipping
    (Yu et al., ICLR 2022; De et al., 2022) significantly improves
    utility by applying separate L2 clips to each parameter tensor.

    The total update is the concatenation of per-layer clipped tensors,
    so the overall L2 sensitivity is sqrt(sum of clip_i^2).

    Args:
        clip_norms: Dict mapping parameter index → L2 clip norm.
            If None, all layers use ``default_clip``.
        default_clip: Default per-layer clip for layers not in ``clip_norms``.

    Returns:
        A Flower client mod (callable).
    """
    from flwr.client.typing import ClientAppCallable
    from flwr.common import (
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.common import recorddict_compat as compat
    from flwr.common.constant import MessageType
    from flwr.common.context import Context
    from flwr.common.message import Message

    _clip_norms = clip_norms or {}

    def per_layer_clip_mod(
        msg: Message, ctxt: Context, call_next: ClientAppCallable,
    ) -> Message:
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        out_msg = call_next(msg, ctxt)
        if out_msg.has_error():
            return out_msg

        fit_res = compat.recorddict_to_fitres(out_msg.content, keep_input=True)
        params = parameters_to_ndarrays(fit_res.parameters)

        clipped = []
        for i, p in enumerate(params):
            clip = _clip_norms.get(i, default_clip)
            norm = float(np.linalg.norm(p))
            if norm > clip:
                p = p * (clip / norm)
            clipped.append(p)

        fit_res.parameters = ndarrays_to_parameters(clipped)
        out_msg.content = compat.fitres_to_recorddict(fit_res, keep_input=True)
        return out_msg

    return per_layer_clip_mod
