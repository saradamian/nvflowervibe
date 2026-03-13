"""
Runtime configuration for ESM2 federated learning.

This module stores ESM2 training configuration at module level so it
can be set by the runner before launching Flower simulation, and read
by client_fn/server_fn which don't receive CLI args directly.
"""

from dataclasses import dataclass, field
from typing import Optional

from sfl.types import FederationConfig


@dataclass
class ESM2RunConfig:
    """Runtime configuration for ESM2 FL training.

    Composes with FederationConfig for federation-level settings,
    adding ESM2-specific model and training hyperparameters.
    """
    federation: FederationConfig = field(
        default_factory=lambda: FederationConfig(num_rounds=3)
    )
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    learning_rate: float = 5e-5
    local_epochs: int = 1
    batch_size: int = 4
    max_length: int = 128

    # Dataset — empty means use built-in demo sequences
    dataset_name: Optional[str] = None
    sequence_column: str = "sequence"
    max_samples: Optional[int] = None

    # Output
    save_dir: Optional[str] = None

    # Convenience properties — delegate to FederationConfig
    @property
    def num_clients(self) -> int:
        return self.federation.num_clients

    @property
    def num_rounds(self) -> int:
        return self.federation.num_rounds

    @property
    def fraction_fit(self) -> float:
        return self.federation.fraction_fit

    @property
    def fraction_evaluate(self) -> float:
        return self.federation.fraction_evaluate


# Module-level config — set by runner, read by client_fn/server_fn
_run_config: Optional[ESM2RunConfig] = None


def set_run_config(config: ESM2RunConfig) -> None:
    """Set the ESM2 run configuration."""
    global _run_config
    _run_config = config


def get_run_config() -> ESM2RunConfig:
    """Get the ESM2 run configuration, or defaults if not set."""
    if _run_config is None:
        return ESM2RunConfig()
    return _run_config
