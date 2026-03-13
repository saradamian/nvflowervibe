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

    def test_default_values(self):
        cfg = ESM2RunConfig()
        assert cfg.num_clients == 2
        assert cfg.num_rounds == 3
        assert cfg.model_name == "facebook/esm2_t6_8M_UR50D"
        assert cfg.learning_rate == 5e-5
        assert cfg.local_epochs == 1
        assert cfg.batch_size == 4
        assert cfg.max_length == 128
        assert cfg.fraction_fit == 1.0
        assert cfg.fraction_evaluate == 1.0

    def test_custom_values(self):
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
        assert cfg.learning_rate == 1e-3
        assert cfg.local_epochs == 3
        assert cfg.batch_size == 16
        assert cfg.max_length == 256
        assert cfg.fraction_fit == 0.5
        assert cfg.fraction_evaluate == 0.5

    def test_partial_override(self):
        cfg = ESM2RunConfig(
            federation=FederationConfig(num_clients=8),
            learning_rate=1e-4,
        )
        assert cfg.num_clients == 8
        assert cfg.learning_rate == 1e-4
        # rest stays default
        assert cfg.num_rounds == 1  # FederationConfig default
        assert cfg.batch_size == 4

    def test_composes_with_federation_config(self):
        fed = FederationConfig(num_clients=5, num_rounds=10)
        cfg = ESM2RunConfig(federation=fed)
        assert cfg.federation is fed
        assert isinstance(cfg.federation, FederationConfig)

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

    def test_set_overrides_previous(self):
        set_run_config(ESM2RunConfig(federation=FederationConfig(num_clients=3)))
        set_run_config(ESM2RunConfig(federation=FederationConfig(num_clients=7)))
        assert get_run_config().num_clients == 7

    def test_get_default_returns_fresh_instance(self):
        a = get_run_config()
        b = get_run_config()
        # Both should be default, but separate instances
        assert a is not b
        assert a.num_clients == b.num_clients
