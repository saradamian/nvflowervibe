"""
Byzantine-robust aggregation strategies for federated learning.

Multi-Krum (Blanchard et al., NeurIPS 2017):
    Selects the K most "typical" client updates by pairwise distance,
    discarding outliers that may be adversarial.

Trimmed Mean (Yin et al., ICML 2018):
    Coordinate-wise trimmed mean — removes the top and bottom β fraction
    of values at each coordinate before averaging.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class MultiKrumFedAvg(FedAvg):
    """FedAvg with Multi-Krum selection to exclude outlier updates.

    For each client update, computes the sum of distances to the
    nearest (n - f - 2) other updates. Selects the ``num_to_select``
    clients with the smallest scores, then averages only those.

    Args:
        num_byzantine: Expected number of Byzantine (adversarial) clients.
        num_to_select: Number of clients to keep after filtering.
            Defaults to ``n - num_byzantine`` (majority).
        **kwargs: Passed to FedAvg.
    """

    def __init__(
        self,
        num_byzantine: int = 0,
        num_to_select: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine
        self._num_to_select = num_to_select

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        n = len(results)
        f = self.num_byzantine
        num_to_select = self._num_to_select or max(1, n - f)

        if n <= 2 * f + 2 or num_to_select >= n:
            # Not enough clients for Krum filtering, fall back to FedAvg
            logger.warning(
                "Multi-Krum: n=%d too small for f=%d, falling back to FedAvg",
                n, f,
            )
            return super().aggregate_fit(server_round, results, failures)

        # Flatten each client's update into a single vector
        updates = []
        for _, res in results:
            flat = np.concatenate(
                [p.ravel() for p in parameters_to_ndarrays(res.parameters)]
            )
            updates.append(flat)

        # Pairwise L2 distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(updates[i] - updates[j]))
                dists[i, j] = d
                dists[j, i] = d

        # Krum score: sum of distances to (n - f - 2) nearest neighbors
        k_nearest = n - f - 2
        scores = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(dists[i])  # includes self (0)
            scores[i] = np.sum(sorted_dists[1 : k_nearest + 1])

        # Select the num_to_select clients with smallest scores
        selected = np.argsort(scores)[:num_to_select]

        logger.info(
            "[multi-krum] round=%d selected %d/%d clients (f=%d), "
            "scores: min=%.2f max=%.2f",
            server_round, num_to_select, n, f,
            scores[selected[0]], scores[selected[-1]],
        )

        # Average only selected clients
        filtered_results = [results[i] for i in selected]
        return super().aggregate_fit(server_round, filtered_results, [])


class TrimmedMeanFedAvg(FedAvg):
    """FedAvg with coordinate-wise trimmed mean aggregation.

    For each parameter coordinate, sorts the client values, removes
    the top and bottom ``trim_ratio`` fraction, and averages the rest.

    Args:
        trim_ratio: Fraction to trim from each side (0–0.5).
            E.g. 0.1 removes 10% lowest and 10% highest at each coordinate.
        **kwargs: Passed to FedAvg.
    """

    def __init__(self, trim_ratio: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")
        self.trim_ratio = trim_ratio

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        n = len(results)
        trim_count = int(n * self.trim_ratio)

        if trim_count == 0 or n < 3:
            # No trimming possible, fall back to FedAvg
            return super().aggregate_fit(server_round, results, failures)

        # Stack all client updates coordinate-wise
        all_params = [
            parameters_to_ndarrays(res.parameters) for _, res in results
        ]

        # For each parameter array, do coordinate-wise trimming
        aggregated = []
        for layer_idx in range(len(all_params[0])):
            stacked = np.stack(
                [params[layer_idx] for params in all_params], axis=0
            )
            # Sort along client axis (axis=0), trim both ends
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[trim_count : n - trim_count]
            aggregated.append(trimmed.mean(axis=0))

        logger.info(
            "[trimmed-mean] round=%d trimmed %d/%d clients per coordinate "
            "(ratio=%.2f)",
            server_round, 2 * trim_count, n, self.trim_ratio,
        )

        parameters = ndarrays_to_parameters(aggregated)

        # Aggregate custom metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters, metrics_aggregated
