"""
Base client for federated learning.

This module provides an abstract base class for all federated clients.
Inherit from BaseFederatedClient to create custom client implementations
with different training logic, data handling, or update strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from flwr.client import NumPyClient

from sfl.types import Parameters, Metrics, Config, ClientUpdate
from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class BaseFederatedClient(NumPyClient, ABC):
    """Abstract base class for federated learning clients.
    
    This class provides the scaffolding for creating federated clients.
    Subclasses must implement the `compute_update` method to define
    how the client processes parameters and produces updates.
    
    Attributes:
        client_id: Unique identifier for this client.
        config: Configuration dictionary for the client.
    
    Example:
        >>> class MyClient(BaseFederatedClient):
        ...     def compute_update(self, parameters, config):
        ...         # Custom training logic here
        ...         return updated_params, num_examples, metrics
    """
    
    def __init__(
        self,
        client_id: int = 0,
        config: Dict[str, Any] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the federated client.
        
        Args:
            client_id: Unique identifier for this client.
            config: Optional configuration dictionary.
            device: Compute device (e.g. 'cpu', 'cuda'). None = auto-detect.
        """
        super().__init__()
        self.client_id = client_id
        self.config = config or {}
        self.device = device or "cpu"
        logger.debug(f"Initialized client {client_id} on device={self.device}")
    
    @abstractmethod
    def compute_update(
        self,
        parameters: Parameters,
        config: Config,
    ) -> ClientUpdate:
        """Compute the client's update based on received parameters.
        
        This is the main method that subclasses must implement.
        It defines how the client processes the global model parameters
        and produces its local update.
        
        Args:
            parameters: Current global model parameters.
            config: Configuration from the server for this round.
        
        Returns:
            Tuple of:
                - Updated parameters (list of numpy arrays)
                - Number of examples used
                - Dictionary of metrics
        """
        raise NotImplementedError
    
    def get_initial_parameters(self) -> Parameters:
        """Get initial parameters for the model.
        
        Override this method to provide custom initial parameters.
        
        Returns:
            List of numpy arrays representing initial parameters.
        """
        return [np.array([0.0], dtype=np.float32)]
    
    def get_parameters(self, config: Config) -> Parameters:
        """Get the current model parameters.
        
        Args:
            config: Configuration from the server.
        
        Returns:
            Current model parameters as a list of numpy arrays.
        """
        return self.get_initial_parameters()
    
    def fit(
        self,
        parameters: Parameters,
        config: Config,
    ) -> ClientUpdate:
        """Perform a training round with the given parameters.
        
        This method is called by the Flower framework during federated
        learning. It delegates to `compute_update` for the actual logic.
        
        Args:
            parameters: Parameters from the server.
            config: Training configuration for this round.
        
        Returns:
            Tuple of updated parameters, example count, and metrics.
        """
        logger.info(f"Client {self.client_id}: Starting fit round")
        
        updated_params, num_examples, metrics = self.compute_update(parameters, config)
        
        logger.info(f"Client {self.client_id}: Completed fit with metrics={metrics}")
        
        return updated_params, num_examples, metrics
    
    def evaluate(
        self,
        parameters: Parameters,
        config: Config,
    ) -> tuple[float, int, Metrics]:
        """Evaluate the model with the given parameters.
        
        Override this method to implement model evaluation logic.
        Subclasses doing real ML should override this.
        
        Args:
            parameters: Parameters from the server.
            config: Evaluation configuration.
        
        Returns:
            Tuple of loss, example count, and metrics.
        """
        logger.info(f"Client {self.client_id}: Evaluate not implemented, returning defaults")
        return 0.0, 1, {}
