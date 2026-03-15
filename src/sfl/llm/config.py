"""
Runtime configuration for LLM federated fine-tuning.

This module stores LLM training configuration at module level so it
can be set by the runner before launching Flower simulation, and read
by client_fn/server_fn which don't receive CLI args directly.

Follows the same singleton-config pattern as sfl.esm2.config.
"""

from dataclasses import dataclass, field
from typing import Optional

from sfl.types import FederationConfig


@dataclass
class LLMRunConfig:
    """Runtime configuration for LLM FL fine-tuning.

    Composes with FederationConfig for federation-level settings,
    adding LLM-specific model and training hyperparameters.

    Attributes:
        federation: Federation-level settings (clients, rounds, fractions).
        model_name: HuggingFace model identifier for a causal LM.
        learning_rate: Optimizer learning rate.
        local_epochs: Number of local training epochs per FL round.
        batch_size: Training batch size.
        max_length: Maximum tokenized sequence length.
        dataset_name: HuggingFace dataset name (None = built-in demo).
        text_column: Name of the text column in HuggingFace datasets.
        max_samples: Maximum samples to load (None = all).
        save_dir: Directory to save the final model (None = no save).
        use_lora: Whether to use LoRA adapters for parameter-efficient fine-tuning.
        lora_r: LoRA rank (number of low-rank dimensions).
        lora_alpha: LoRA scaling factor.
    """
    federation: FederationConfig = field(
        default_factory=lambda: FederationConfig(num_rounds=3)
    )
    model_name: str = "gpt2"  # Small enough to run anywhere
    learning_rate: float = 5e-5
    local_epochs: int = 1
    batch_size: int = 2
    max_length: int = 128
    dataset_name: Optional[str] = None  # HuggingFace dataset
    text_column: str = "text"
    max_samples: Optional[int] = None
    save_dir: Optional[str] = None
    use_lora: bool = False  # LoRA fine-tuning for efficiency
    lora_r: int = 8
    lora_alpha: int = 16

    # Convenience properties -- delegate to FederationConfig
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


# Module-level config -- set by runner, read by client_fn/server_fn
_run_config: Optional[LLMRunConfig] = None


def set_run_config(config: LLMRunConfig) -> None:
    """Set the LLM run configuration."""
    global _run_config
    _run_config = config


def get_run_config() -> LLMRunConfig:
    """Get the LLM run configuration, or defaults if not set."""
    if _run_config is None:
        return LLMRunConfig()
    return _run_config
