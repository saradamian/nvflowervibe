"""
Base client for federated inference.

Provides an abstract base class for inference-only federated clients.
Unlike BaseFederatedClient (training), inference clients receive model
parameters and return predictions or evaluation metrics — they do NOT
send updated parameters back to the server.

Use cases:
- Private prediction serving across institutions
- Federated evaluation (aggregate test metrics without sharing data)
- Split inference (client runs first N layers, server runs the rest)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from flwr.client import NumPyClient
from sfl.types import Parameters, Metrics, Config
from sfl.utils.logging import get_logger

logger = get_logger(__name__)

# Return type: predictions (any serializable), num_examples, metrics
InferenceResult = Tuple[Any, int, Metrics]


class BaseInferenceClient(NumPyClient, ABC):
    """Abstract base class for federated inference clients.

    Unlike BaseFederatedClient which returns updated model parameters,
    inference clients receive parameters and return evaluation results.
    The server aggregates metrics (accuracy, loss, etc.) across clients
    without ever seeing raw predictions or input data.

    Subclasses must implement:
        - compute_predictions: Run inference and return metrics

    Optionally override:
        - get_parameters: Return empty list (inference clients don't send params)
        - evaluate: Delegates to compute_predictions by default

    Example:
        class DiagnosticClient(BaseInferenceClient):
            def __init__(self, client_id, model, test_data, device=None):
                super().__init__(client_id=client_id, device=device or "cpu")
                self.model = model
                self.test_data = test_data

            def compute_predictions(self, parameters, config):
                set_parameters(self.model, parameters)
                accuracy = evaluate_model(self.model, self.test_data)
                return None, len(self.test_data), {"accuracy": accuracy}
    """

    def __init__(self, client_id=0, device=None):
        super().__init__()
        self.client_id = client_id
        self.device = device or "cpu"

    @abstractmethod
    def compute_predictions(self, parameters: Parameters, config: Config) -> InferenceResult:
        """Run inference with the given model parameters.

        Args:
            parameters: Model parameters from the server.
            config: Configuration for this inference round.

        Returns:
            Tuple of (predictions, num_examples, metrics).
            predictions can be None if only metrics are needed.
        """
        raise NotImplementedError

    def get_parameters(self, config: Config) -> Parameters:
        """Inference clients don't send parameters back."""
        return []

    def fit(self, parameters: Parameters, config: Config):
        """Inference clients don't train — delegates to evaluate."""
        logger.warning(
            "InferenceClient %d received fit() call — delegating to evaluate(). "
            "If you need training, use BaseFederatedClient instead.",
            self.client_id,
        )
        loss, num, metrics = self.evaluate(parameters, config)
        return self.get_parameters(config), num, metrics

    def evaluate(self, parameters: Parameters, config: Config):
        """Run inference and return metrics."""
        _, num_examples, metrics = self.compute_predictions(parameters, config)
        loss = float(metrics.get("loss", 0.0))
        return loss, num_examples, metrics
