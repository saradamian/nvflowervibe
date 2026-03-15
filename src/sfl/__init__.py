"""
SFL - Simple Federated Learning Demo

A clean, maintainable federated learning framework using NVFlare + Flower.
"""

__version__ = "0.1.0"
__author__ = "SFL Team"

from sfl.types import FederationConfig, ClientConfig, ServerConfig
from sfl.client.inference import BaseInferenceClient, InferenceResult

__all__ = [
    "__version__",
    "FederationConfig",
    "ClientConfig",
    "ServerConfig",
    "BaseInferenceClient",
    "InferenceResult",
]
