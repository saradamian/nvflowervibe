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

logger = get_logger(__name__)


@dataclass
class AdaptiveClipConfig:
    """Configuration for adaptive clipping.

    Args:
        target_quantile: Fraction of updates that should be unclipped.
        learning_rate: Step size for geometric clip norm update.
        clip_min: Floor for the clipping norm.
        clip_max: Ceiling for the clipping norm.
    """
    target_quantile: float = 0.5
    learning_rate: float = 0.2
    clip_min: float = 0.1
    clip_max: float = 100.0


class AdaptiveClipWrapper(Strategy):
    """Wraps a DP strategy to adaptively update the clipping norm.

    After each round, estimates what fraction of client updates exceeded
    the current clip norm. Adjusts clip norm toward target_quantile using
    the geometric update rule from Andrew et al. (2021).

    The wrapped strategy must have ``clipping_norm`` and
    ``current_round_params`` attributes (as Flower's DP wrappers do).
    """

    def __init__(self, strategy: Strategy, config: AdaptiveClipConfig) -> None:
        super().__init__()
        self.strategy = strategy
        self.config = config

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

            # Fraction exceeding current clip norm
            fraction_clipped = sum(1 for n in norms if n > clip_norm) / len(norms)

            # Geometric update (Andrew et al., 2021)
            cfg = self.config
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

            # Update the inner DP strategy's clipping norm for THIS round
            self.strategy.clipping_norm = new_clip

        # Delegate clipping + noise + aggregation to the DP wrapper
        return self.strategy.aggregate_fit(server_round, results, failures)
