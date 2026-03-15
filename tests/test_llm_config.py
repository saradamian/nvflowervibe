"""
Tests for LLM runtime configuration.

Covers LLMRunConfig dataclass defaults, custom values,
FederationConfig composition, and module-level config getter/setter.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.slow

from sfl.llm.config import LLMRunConfig, get_run_config, set_run_config
from sfl.types import FederationConfig
import sfl.llm.config as config_module


class TestLLMRunConfigDefaults:
    """LLMRunConfig should have sensible defaults."""

    def test_default_model_name(self):
        cfg = LLMRunConfig()
        assert cfg.model_name == "gpt2"

    def test_default_federation(self):
        cfg = LLMRunConfig()
        assert cfg.num_clients == 2
        assert cfg.num_rounds == 3
        assert cfg.fraction_fit == 1.0
        assert cfg.fraction_evaluate == 1.0

    def test_default_training_params(self):
        cfg = LLMRunConfig()
        assert cfg.learning_rate == 5e-5
        assert cfg.local_epochs == 1
        assert cfg.batch_size == 2
        assert cfg.max_length == 128

    def test_default_lora_disabled(self):
        cfg = LLMRunConfig()
        assert cfg.use_lora is False
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16

    def test_default_dataset_is_none(self):
        cfg = LLMRunConfig()
        assert cfg.dataset_name is None
        assert cfg.text_column == "text"
        assert cfg.max_samples is None
        assert cfg.save_dir is None


class TestLLMRunConfigCustom:
    """Custom values flow through FederationConfig composition."""

    def test_custom_values_and_delegation(self):
        cfg = LLMRunConfig(
            federation=FederationConfig(
                num_clients=4,
                num_rounds=10,
                fraction_fit=0.5,
                fraction_evaluate=0.5,
            ),
            model_name="gpt2-medium",
            learning_rate=1e-3,
            local_epochs=3,
            batch_size=16,
            max_length=256,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
        )
        assert cfg.num_clients == 4
        assert cfg.num_rounds == 10
        assert cfg.fraction_fit == 0.5
        assert cfg.model_name == "gpt2-medium"
        assert cfg.learning_rate == 1e-3
        assert cfg.use_lora is True
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32

    def test_federation_validation_propagates(self):
        """FederationConfig validation fires through LLMRunConfig."""
        with pytest.raises(ValueError, match="num_clients must be at least 1"):
            LLMRunConfig(federation=FederationConfig(num_clients=0))


class TestRunConfigModule:
    """Module-level get/set config helpers."""

    def setup_method(self):
        config_module._run_config = None

    def teardown_method(self):
        config_module._run_config = None

    def test_get_returns_default_when_not_set(self):
        cfg = get_run_config()
        assert isinstance(cfg, LLMRunConfig)
        assert cfg.num_clients == 2
        assert cfg.model_name == "gpt2"

    def test_set_and_get(self):
        custom = LLMRunConfig(
            federation=FederationConfig(num_clients=10, num_rounds=5),
        )
        set_run_config(custom)
        assert get_run_config() is custom
        assert get_run_config().num_clients == 10
        assert get_run_config().num_rounds == 5
