"""
SFL Client Module

Provides federated learning client implementations using the Flower framework.
The client module is designed with extensibility in mind - new client types
can be easily added by inheriting from BaseFederatedClient.
"""

import os

from sfl.client.base import BaseFederatedClient
from sfl.client.inference import BaseInferenceClient, InferenceResult
from sfl.client.sum_client import SumClient, client_fn

from flwr.client import ClientApp

# Build client mods from env vars (populated by runner or NVFlare extra_env)
from sfl.privacy.auto_mods import auto_build_client_mods

_mods = auto_build_client_mods()
app = ClientApp(client_fn=client_fn, mods=_mods if _mods else None)

__all__ = [
    "BaseFederatedClient",
    "BaseInferenceClient",
    "InferenceResult",
    "SumClient",
    "client_fn",
    "app",
]
