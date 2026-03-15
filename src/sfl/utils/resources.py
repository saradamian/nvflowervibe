"""
Resource-aware client scheduling for heterogeneous HPC clusters.

Provides configurable per-client resource allocation for Flower
simulations, replacing the naive ``num_gpus / num_clients`` split
with explicit resource specs that account for mixed GPU types
(A100, V100, CPU-only nodes, etc.).

Usage::

    from sfl.utils.resources import (
        ClientResources, ResourceConfig,
        detect_resources, build_backend_config, parse_resource_config,
    )

    resource_cfg = parse_resource_config(args)
    backend_cfg = build_backend_config(resource_cfg, num_clients=8)
    run_simulation(..., backend_config=backend_cfg)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ClientResources:
    """Resource specification for a single simulation client.

    Attributes:
        num_cpus: Number of CPU cores allocated to this client.
        num_gpus: Fractional GPU count (e.g. 0.5 = share a GPU with
            one other client).
        memory_mb: Optional memory hint in MB for documentation and
            external schedulers (not enforced by Flower).
        label: Human-readable tag for the resource tier
            (e.g. ``"a100"``, ``"v100"``, ``"cpu-only"``).
    """

    num_cpus: int = 1
    num_gpus: float = 0.0
    memory_mb: Optional[int] = None
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return the Flower-compatible ``client_resources`` dict."""
        return {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
        }


@dataclass
class ResourceConfig:
    """Cluster-level resource configuration for a simulation run.

    Attributes:
        default: Default resource spec applied to every client.
        overrides: Per-client overrides keyed by client index (0-based).
        auto_detect: When True, ``detect_resources`` is called to fill
            in GPU count and CPU count if the default leaves them at
            their zero/one sentinel values.
    """

    default: ClientResources = field(default_factory=ClientResources)
    overrides: Dict[int, ClientResources] = field(default_factory=dict)
    auto_detect: bool = True


def detect_resources() -> ClientResources:
    """Auto-detect available compute resources on the current node.

    GPU detection uses ``torch.cuda`` when available; CPU count comes
    from ``os.cpu_count()``.

    Returns:
        A ``ClientResources`` populated with detected values.
    """
    cpus = os.cpu_count() or 1

    num_gpus = 0
    label = "cpu-only"
    try:
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                label = torch.cuda.get_device_name(0).lower().replace(" ", "-")
    except ImportError:
        pass

    return ClientResources(
        num_cpus=cpus,
        num_gpus=float(num_gpus),
        label=label,
    )


def build_backend_config(
    resource_config: ResourceConfig,
    num_clients: int,
) -> Dict[str, Any]:
    """Build a Flower ``backend_config`` dict for ``run_simulation()``.

    When ``resource_config.auto_detect`` is True and the default
    ``num_gpus`` is 0, the function detects hardware and distributes
    GPUs evenly across clients.  Per-client overrides are stored under
    the ``"client_resources_overrides"`` key so callers can apply them
    if the Flower backend supports heterogeneous allocation.

    Args:
        resource_config: The cluster resource configuration.
        num_clients: Number of simulation super-nodes.

    Returns:
        A dict suitable for the ``backend_config`` parameter of
        ``flwr.simulation.run_simulation()``.
    """
    default = resource_config.default

    if resource_config.auto_detect:
        detected = detect_resources()
        # Only override GPU count when the user left the default at 0
        if default.num_gpus == 0.0:
            gpus_per_client = (
                detected.num_gpus / num_clients
                if detected.num_gpus > 0 and num_clients > 0
                else 0.0
            )
            default = ClientResources(
                num_cpus=default.num_cpus,
                num_gpus=gpus_per_client,
                memory_mb=default.memory_mb,
                label=default.label or detected.label,
            )

    config: Dict[str, Any] = {
        "client_resources": default.to_dict(),
    }

    if resource_config.overrides:
        config["client_resources_overrides"] = {
            idx: res.to_dict() for idx, res in resource_config.overrides.items()
        }

    return config


def parse_resource_config(args: Any) -> ResourceConfig:
    """Build a ``ResourceConfig`` from parsed CLI arguments.

    Expects the attributes added by
    :func:`sfl.privacy.runner_utils.add_privacy_args`:
    ``client_cpus``, ``client_gpus``, ``client_memory``, and
    ``no_auto_detect_gpu``.

    Args:
        args: An ``argparse.Namespace`` (or any object with the same
            attributes).

    Returns:
        A fully populated ``ResourceConfig``.
    """
    auto_detect = not getattr(args, "no_auto_detect_gpu", False)

    default = ClientResources(
        num_cpus=getattr(args, "client_cpus", 1),
        num_gpus=float(getattr(args, "client_gpus", 0)),
        memory_mb=getattr(args, "client_memory", None),
    )

    return ResourceConfig(
        default=default,
        overrides={},
        auto_detect=auto_detect,
    )
