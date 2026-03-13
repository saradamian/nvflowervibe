"""
Type definitions for SFL.

This module contains all shared type definitions, dataclasses, and type aliases
used throughout the SFL project. Centralizing types here improves maintainability
and ensures consistency across the codebase.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
Parameters = List[NDArray[np.float32]]
Metrics = Dict[str, Any]
Config = Dict[str, Any]
ClientUpdate = Tuple[Parameters, int, Metrics]


@dataclass
class FederationConfig:
    """Configuration for the federated learning setup.
    
    Attributes:
        num_clients: Number of federated clients participating.
        num_rounds: Number of federated learning rounds.
        min_available_clients: Minimum clients required to start.
        min_fit_clients: Minimum clients required for fit aggregation.
        fraction_fit: Fraction of clients sampled for training each round.
        fraction_evaluate: Fraction of clients sampled for evaluation.
    """
    num_clients: int = 2
    num_rounds: int = 1
    min_available_clients: int = 2
    min_fit_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_clients < 1:
            raise ValueError("num_clients must be at least 1")
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be at least 1")
        if self.min_available_clients > self.num_clients:
            raise ValueError("min_available_clients cannot exceed num_clients")
        if not 0.0 < self.fraction_fit <= 1.0:
            raise ValueError("fraction_fit must be in (0.0, 1.0]")
        if not 0.0 < self.fraction_evaluate <= 1.0:
            raise ValueError("fraction_evaluate must be in (0.0, 1.0]")


@dataclass
class ClientConfig:
    """Configuration for federated clients.
    
    Attributes:
        base_secret: Base value for client secrets (node_id is added).
    """
    base_secret: float = 7.0


@dataclass
class ServerConfig:
    """Configuration for the federated server.
    
    Attributes:
        initial_param: Initial model parameter value.
    """
    initial_param: float = 0.0


@dataclass
class NVFlareConfig:
    """Configuration for NVFlare simulation.
    
    Attributes:
        job_name: Name of the NVFlare job.
        stream_metrics: Whether to stream metrics to TensorBoard.
        num_threads: Number of simulation threads (defaults to num_clients).
    """
    job_name: str = "sfl-federated-sum"
    stream_metrics: bool = False
    num_threads: Optional[int] = None


@dataclass
class LoggingConfig:
    """Configuration for logging.
    
    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log format (rich, simple, json).
        file_enabled: Whether to log to file.
        file_path: Path to log file.
    """
    level: str = "INFO"
    format: str = "rich"
    file_enabled: bool = False
    file_path: str = "./logs/sfl.log"


@dataclass
class SFLConfig:
    """Complete SFL configuration.
    
    This is the top-level configuration object that contains all
    sub-configurations for the federated learning system.
    """
    federation: FederationConfig = field(default_factory=FederationConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    nvflare: NVFlareConfig = field(default_factory=NVFlareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
