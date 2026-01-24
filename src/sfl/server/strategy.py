"""
Custom aggregation strategies for federated learning.

This module provides custom FedAvg-based strategies that extend
Flower's built-in aggregation with additional functionality like
logging, custom metrics, and sum computation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

# Type aliases for clarity
ClientFitResults = List[Tuple[ClientProxy, FitRes]]
AggregateResult = Optional[Tuple[Parameters, Dict[str, Scalar]]]


class SumFedAvg(FedAvg):
    """FedAvg strategy that computes and logs the sum of client values.
    
    This strategy extends FedAvg to compute the actual sum of values
    contributed by clients, rather than the average. It's designed
    for the federated sum demonstration but can be extended for
    other aggregation patterns.
    
    The strategy:
    1. Calls parent FedAvg.aggregate_fit for standard aggregation
    2. Extracts individual client values from fit results
    3. Computes and logs the sum
    4. Adds sum to metrics for tracking
    
    Attributes:
        All attributes inherited from FedAvg.
    
    Example:
        >>> strategy = SumFedAvg(
        ...     min_fit_clients=2,
        ...     min_available_clients=2,
        ... )
    """
    
    def __init__(
        self,
        *args,
        log_client_values: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the SumFedAvg strategy.
        
        Args:
            log_client_values: Whether to log individual client values.
            *args: Passed to parent FedAvg.
            **kwargs: Passed to parent FedAvg.
        """
        super().__init__(*args, **kwargs)
        self.log_client_values = log_client_values
        logger.info("SumFedAvg strategy initialized")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: ClientFitResults,
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> AggregateResult:
        """Aggregate client fit results and compute the sum.
        
        This method extends FedAvg's aggregation by also computing
        the sum of all client values and logging the results.
        
        Args:
            server_round: Current round number.
            results: List of (client, fit_result) tuples from clients.
            failures: List of failed client results or exceptions.
        
        Returns:
            Tuple of aggregated parameters and metrics, or None if failed.
        """
        # Log any failures
        if failures:
            logger.warning(
                f"Round {server_round}: {len(failures)} client(s) failed"
            )
        
        # Call parent aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is None:
            logger.error(f"Round {server_round}: Aggregation failed")
            return None
        
        params, metrics = aggregated
        
        # Extract values from each client
        client_values = self._extract_client_values(results)
        
        # Compute the sum
        federated_sum = sum(client_values)
        
        # Log results
        if self.log_client_values:
            logger.info(
                f"[server] round={server_round} "
                f"client_vals={client_values} "
                f"federated_sum={federated_sum}"
            )
        
        # Add to metrics
        metrics = dict(metrics) if metrics else {}
        metrics["federated_sum"] = federated_sum
        metrics["num_clients"] = len(client_values)
        metrics["client_values"] = str(client_values)  # For tracking
        
        return params, metrics
    
    def _extract_client_values(
        self,
        results: ClientFitResults,
    ) -> List[float]:
        """Extract scalar values from client fit results.
        
        Args:
            results: List of client fit results.
        
        Returns:
            List of scalar values from each client.
        """
        values = []
        
        for client_proxy, fit_res in results:
            try:
                # Convert parameters to numpy arrays
                arrays = parameters_to_ndarrays(fit_res.parameters)
                
                if arrays and len(arrays) > 0:
                    # Extract the first (and only) scalar value
                    value = float(arrays[0].item())
                    values.append(value)
                else:
                    logger.warning(
                        f"Client {client_proxy.cid}: Empty parameters received"
                    )
            except Exception as e:
                logger.error(
                    f"Client {client_proxy.cid}: Failed to extract value: {e}"
                )
        
        return values
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluation results.
        
        Currently a pass-through to parent implementation.
        Override to add custom evaluation aggregation logic.
        """
        return super().aggregate_evaluate(server_round, results, failures)
