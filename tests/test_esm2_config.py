"""
Tests for ESM2 runtime configuration.

Covers ESM2RunConfig dataclass defaults, custom values,
FederationConfig composition, and module-level config getter/setter.
"""

import pytest

from sfl.esm2.config import ESM2RunConfig, get_run_config, set_run_config
from sfl.types import FederationConfig
import sfl.esm2.config as config_module


class TestESM2RunConfig:
    """ESM2RunConfig dataclass behaviour."""

    def test_custom_values_and_delegation(self):
        """Custom values flow through FederationConfig composition."""
        cfg = ESM2RunConfig(
            federation=FederationConfig(
                num_clients=4,
                num_rounds=10,
                fraction_fit=0.5,
                fraction_evaluate=0.5,
            ),
            model_name="custom/model",
            learning_rate=1e-3,
            local_epochs=3,
            batch_size=16,
            max_length=256,
        )
        assert cfg.num_clients == 4
        assert cfg.num_rounds == 10
        assert cfg.model_name == "custom/model"
        assert cfg.fraction_fit == 0.5

    def test_federation_validation_propagates(self):
        """FederationConfig validation fires through ESM2RunConfig."""
        with pytest.raises(ValueError, match="num_clients must be at least 1"):
            ESM2RunConfig(federation=FederationConfig(num_clients=0))


class TestRunConfigModule:
    """Module-level get/set config helpers."""

    def setup_method(self):
        # Reset to None before each test
        config_module._run_config = None

    def test_get_returns_default_when_not_set(self):
        cfg = get_run_config()
        assert isinstance(cfg, ESM2RunConfig)
        assert cfg.num_clients == 2

    def test_set_and_get(self):
        custom = ESM2RunConfig(
            federation=FederationConfig(num_clients=10, num_rounds=5),
        )
        set_run_config(custom)
        assert get_run_config() is custom
        assert get_run_config().num_clients == 10
        assert get_run_config().num_rounds == 5


