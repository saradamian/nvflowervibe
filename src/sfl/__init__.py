"""
SFL - Simple Federated Learning Demo

A clean, maintainable federated learning framework using NVFlare + Flower.
"""

__version__ = "0.1.0"
__author__ = "SFL Team"

from sfl.types import FederationConfig, ClientConfig, ServerConfig

__all__ = [
    "__version__",
    "FederationConfig",
    "ClientConfig", 
    "ServerConfig",
]
