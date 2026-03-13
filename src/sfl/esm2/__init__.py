"""
ESM2 Federated Learning Module.

Provides federated fine-tuning of ESM2 protein language models
using the Flower framework, orchestrated by NVFlare.
"""

from sfl.esm2.client import ESM2Client, client_fn
from sfl.esm2.server import server_fn

# Flower apps for NVFlare integration
from flwr.client import ClientApp
from flwr.server import ServerApp

client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp(server_fn=server_fn)

__all__ = [
    "ESM2Client",
    "client_fn",
    "server_fn",
    "client_app",
    "server_app",
]
