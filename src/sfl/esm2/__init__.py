"""
ESM2 Federated Learning Module.

Provides federated fine-tuning of ESM2 protein language models
using the Flower framework, orchestrated by NVFlare.

In distributed mode (NVFlare PocEnv/ProdEnv), privacy mods are
rebuilt from SFL_* env vars via auto_build_client_mods().
"""

import os

from sfl.esm2.client import ESM2Client, client_fn
from sfl.esm2.server import server_fn

from flwr.client import ClientApp
from flwr.server import ServerApp

# Build client mods from env vars (populated by runner or NVFlare extra_env)
from sfl.privacy.auto_mods import auto_build_client_mods

_mods = auto_build_client_mods()
client_app = ClientApp(client_fn=client_fn, mods=_mods if _mods else None)

# Wrap server with SecAgg if configured via env var
if os.environ.get("SFL_SECAGG_ENABLED", "").lower() == "true":
    from sfl.privacy.secagg import SecAggConfig, make_secagg_main
    _secagg_cfg = SecAggConfig(
        num_shares=int(os.environ.get("SFL_SECAGG_SHARES", "3")),
        reconstruction_threshold=int(os.environ.get("SFL_SECAGG_THRESHOLD", "2")),
        clipping_range=float(os.environ.get("SFL_SECAGG_CLIP", "8.0")),
    )
    server_app = ServerApp()
    server_app.main()(make_secagg_main(server_fn, _secagg_cfg))
else:
    server_app = ServerApp(server_fn=server_fn)

__all__ = [
    "ESM2Client",
    "client_fn",
    "server_fn",
    "client_app",
    "server_app",
]
