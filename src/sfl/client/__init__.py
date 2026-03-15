"""
SFL Client Module

Provides federated learning client implementations using the Flower framework.
The client module is designed with extensibility in mind - new client types
can be easily added by inheriting from BaseFederatedClient.
"""

from sfl.client.base import BaseFederatedClient
from sfl.client.inference import BaseInferenceClient, InferenceResult
from sfl.client.sum_client import SumClient, client_fn

# Flower ClientApp for NVFlare integration
from flwr.client import ClientApp

# Create the Flower ClientApp
app = ClientApp(client_fn=client_fn)

__all__ = [
    "BaseFederatedClient",
    "BaseInferenceClient",
    "InferenceResult",
    "SumClient",
    "client_fn",
    "app",
]
