"""
Tests for ESM2 federated server.

Covers server_fn configuration, FedAvg strategy initialization,
and initial parameter seeding from pretrained ESM2 weights.
"""

import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.slow

from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

import sfl.esm2.config as config_module
from sfl.esm2.config import ESM2RunConfig, set_run_config
from sfl.esm2.model import DEFAULT_MODEL_NAME, load_model, get_parameters
from sfl.esm2.server import server_fn
from sfl.types import FederationConfig


@pytest.fixture(autouse=True)
def reset_config():
    """Reset ESM2 config before each test."""
    config_module._run_config = None
    yield
    config_module._run_config = None


def _make_context(**run_config_overrides):
    ctx = MagicMock()
    ctx.run_config = run_config_overrides
    return ctx


class TestServerFn:

    def test_returns_components(self):
        components = server_fn(_make_context())
        assert components.strategy is not None
        assert components.config is not None

    def test_uses_fedavg(self):
        components = server_fn(_make_context())
        assert isinstance(components.strategy, FedAvg)

    def test_default_num_rounds(self):
        """Default from ESM2RunConfig.federation.num_rounds = 3."""
        components = server_fn(_make_context())
        assert components.config.num_rounds == 3

    def test_run_config_overrides_rounds(self):
        components = server_fn(_make_context(**{"num-server-rounds": 7}))
        assert components.config.num_rounds == 7

    def test_shared_config_overrides_rounds(self):
        set_run_config(ESM2RunConfig(
            federation=FederationConfig(num_rounds=5),
        ))
        components = server_fn(_make_context())
        assert components.config.num_rounds == 5

    def test_run_config_takes_priority_over_shared(self):
        """context.run_config should override shared ESM2RunConfig."""
        set_run_config(ESM2RunConfig(
            federation=FederationConfig(num_rounds=5),
        ))
        components = server_fn(_make_context(**{"num-server-rounds": 10}))
        assert components.config.num_rounds == 10

    def test_initial_params_match_model(self):
        """Strategy should be seeded with pretrained ESM2 weights."""
        components = server_fn(_make_context())
        strategy = components.strategy
        # FedAvg stores initial_parameters
        assert strategy.initial_parameters is not None
        init_ndarrays = parameters_to_ndarrays(strategy.initial_parameters)
        # Should have the same number of tensors as the model
        model = load_model(DEFAULT_MODEL_NAME)
        expected = get_parameters(model)
        assert len(init_ndarrays) == len(expected)
        for a, b in zip(init_ndarrays, expected):
            assert a.shape == b.shape
