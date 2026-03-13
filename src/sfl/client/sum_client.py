"""
Sum Client implementation for federated sum demonstration.

This client demonstrates basic federated learning concepts by contributing
a "secret" value that gets aggregated across all clients. It's designed
to be simple, educational, and easy to extend.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from flwr.common import Context

from sfl.client.base import BaseFederatedClient
from sfl.utils.logging import get_logger
from sfl.utils.config import get_config

logger = get_logger(__name__)


class SumClient(BaseFederatedClient):
    """Federated client that contributes a secret value for aggregation.
    
    Each client has a "secret" number based on its ID. During federated
    learning, clients send their secrets to the server, which aggregates
    them to compute the sum. This demonstrates:
    
    - Client-to-server parameter transmission
    - Server-side aggregation
    - Metrics collection and reporting
    
    Attributes:
        secret: The client's secret value to contribute.
    
    Example:
        >>> client = SumClient(client_id=0, secret=7.0)
        >>> params, n, metrics = client.fit([], {})
        >>> print(metrics["client_secret"])
        7.0
    """
    
    def __init__(
        self,
        client_id: int = 0,
        secret: float = 0.0,
        config: Dict[str, Any] = None,
    ) -> None:
        """Initialize the Sum client.
        
        Args:
            client_id: Unique identifier for this client.
            secret: The secret value this client will contribute.
            config: Optional additional configuration.
        """
        super().__init__(client_id=client_id, config=config)
        self.secret = float(secret)
        logger.info(f"SumClient {client_id} initialized with secret={self.secret}")
    
    def compute_update(
        self,
        parameters: List[NDArray[np.float32]],
        config: Dict[str, Any],
    ) -> Tuple[List[NDArray[np.float32]], int, Dict[str, Any]]:
        """Contribute this client's secret value.
        
        The client packages its secret as a single-element numpy array
        and reports it as a metric for tracking.
        
        Args:
            parameters: Current global parameters (ignored for sum).
            config: Server configuration (ignored for sum).
        
        Returns:
            Tuple of:
                - Parameters containing the secret value
                - Number of examples (always 1 for sum)
                - Metrics including the client's secret
        """
        # Package secret as model parameters
        updated_params = [np.array([self.secret], dtype=np.float32)]
        
        # Report metrics for tracking/debugging
        metrics: Dict[str, Any] = {
            "client_id": self.client_id,
            "client_secret": self.secret,
        }
        
        logger.debug(
            f"Client {self.client_id}: Contributing secret={self.secret}"
        )
        
        # Return: parameters, num_examples, metrics
        return updated_params, 1, metrics


def client_fn(context: Context) -> SumClient:
    """Factory function to create SumClient instances.
    
    This function is called by Flower's ClientApp to create client
    instances. Each client gets a unique node_id from the context,
    which is used to generate a deterministic secret value.
    
    Args:
        context: Flower context containing node_id and run_config.
    
    Returns:
        A configured SumClient instance wrapped for Flower.
    
    Note:
        The secret is computed as: base_secret + partition_id
        For partition_ids 0 and 1 with base_secret=7.0:
            - Client 0: secret = 7.0
            - Client 1: secret = 8.0
    """
    # Get configuration
    try:
        config = get_config()
        base_secret = config.client.base_secret
    except (RuntimeError, FileNotFoundError, KeyError, AttributeError) as e:
        # Fall back to default if config not loaded or malformed
        logger.warning(f"Config unavailable ({type(e).__name__}: {e}), using default base_secret=7.0")
        base_secret = 7.0
    
    # Get partition ID (simpler than node_id for demos)
    # partition_id is 0-indexed and sequential in simulation
    partition_id = context.node_config.get("partition-id", 0)
    if isinstance(partition_id, str):
        partition_id = int(partition_id)
    
    # Use node_id as client_id (for tracking) but partition_id for secret
    node_id = int(context.node_id)
    secret = base_secret + float(partition_id)
    
    logger.info(
        f"Creating SumClient: node_id={node_id}, partition_id={partition_id}, "
        f"base_secret={base_secret}, secret={secret}"
    )
    
    # Create client and convert to Flower client interface
    client = SumClient(
        client_id=node_id,
        secret=secret,
    )
    
    return client.to_client()
