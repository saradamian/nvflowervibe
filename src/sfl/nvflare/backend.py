"""
NVFlare execution backend for SFL.

Provides a clean abstraction over NVFlare's three execution environments:
- SimEnv: local simulation (all clients in one process)
- PocEnv: proof-of-concept (separate processes, single machine)
- ProdEnv: production (distributed across machines via startup kits)

All three use the same FlowerRecipe — the Flower apps (strategies,
privacy mods, client training) are identical across environments.
Only the execution infrastructure changes.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class NVFlareMode(Enum):
    """NVFlare execution mode."""
    SIM = "sim"       # SimEnv: local simulation
    POC = "poc"       # PocEnv: multi-process on one machine
    PROD = "prod"     # ProdEnv: real distributed via startup kits


@dataclass
class NVFlareBackendConfig:
    """Configuration for NVFlare execution.

    Args:
        mode: Execution environment (sim, poc, prod).
        num_clients: Number of federated clients.
        flower_content: Path to staged Flower content directory.
        job_name: NVFlare job name.
        extra_env: SFL_* env vars to propagate to NVFlare sites.
        num_threads: Thread count for SimEnv (default: num_clients).
        startup_kit: Path to admin startup kit (required for prod mode).
        progress_timeout: Seconds to wait for job progress (prod mode).
    """
    mode: NVFlareMode = NVFlareMode.SIM
    num_clients: int = 2
    flower_content: str = ""
    job_name: str = "sfl-federated"
    extra_env: Dict[str, str] = field(default_factory=dict)
    num_threads: Optional[int] = None
    startup_kit: Optional[str] = None
    progress_timeout: float = 3600.0

    def __post_init__(self):
        if self.mode == NVFlareMode.PROD and not self.startup_kit:
            raise ValueError(
                "startup_kit is required for production mode. "
                "Run 'nvflare provision' first to generate startup kits."
            )


def build_extra_env(include_non_sfl: bool = False) -> Dict[str, str]:
    """Capture SFL_* env vars for propagation to NVFlare sites.

    After ``build_privacy_mods()`` sets SFL_* env vars in the runner
    process, this function captures them into a dict suitable for
    ``FlowerRecipe(extra_env=...)``. NVFlare passes these to each
    site's flower-supernode subprocess.

    Args:
        include_non_sfl: Also capture NCCL_*, CUDA_* vars (for HPC).

    Returns:
        Dict of env var name -> value.
    """
    env = {}
    for key, value in os.environ.items():
        if key.startswith("SFL_"):
            env[key] = value
        elif include_non_sfl and key.startswith(("NCCL_", "CUDA_VISIBLE")):
            env[key] = value
    return env


def run_nvflare(config: NVFlareBackendConfig) -> int:
    """Execute a Flower app via NVFlare.

    Creates the appropriate NVFlare execution environment and runs
    the FlowerRecipe. Privacy mods, strategies, and all Flower logic
    run unchanged — NVFlare handles distribution and lifecycle.

    Args:
        config: Backend configuration.

    Returns:
        0 on success, non-zero on failure.
    """
    try:
        from nvflare.app_opt.flower.recipe import FlowerRecipe
    except ImportError:
        logger.error(
            "NVFlare not available. Install with: pip install nvflare>=2.5"
        )
        return 1

    if not config.flower_content or not Path(config.flower_content).exists():
        logger.error("flower_content directory not found: %s", config.flower_content)
        return 1

    recipe = FlowerRecipe(
        flower_content=config.flower_content,
        name=config.job_name,
        min_clients=config.num_clients,
        extra_env=config.extra_env or None,
    )

    env = _create_exec_env(config)

    logger.info(
        "Starting NVFlare %s: job=%s, clients=%d, env_vars=%d",
        config.mode.value, config.job_name, config.num_clients,
        len(config.extra_env),
    )

    try:
        recipe.execute(env=env)
        return 0
    except Exception as e:
        logger.error("NVFlare execution failed: %s", e)
        logger.exception("Full traceback:")
        return 1


def _create_exec_env(config: NVFlareBackendConfig):
    """Create the appropriate NVFlare execution environment."""
    if config.mode == NVFlareMode.SIM:
        from nvflare.recipe.sim_env import SimEnv
        num_threads = config.num_threads or config.num_clients
        return SimEnv(
            num_clients=config.num_clients,
            num_threads=num_threads,
        )

    elif config.mode == NVFlareMode.POC:
        from nvflare.recipe.poc_env import PocEnv
        return PocEnv(num_clients=config.num_clients)

    elif config.mode == NVFlareMode.PROD:
        from nvflare.recipe.prod_env import ProdEnv
        return ProdEnv(startup_kit_location=config.startup_kit)

    raise ValueError(f"Unknown NVFlare mode: {config.mode}")
