"""
Structural integrity tests for the SFL package.

These tests guard the project's architecture: package layout, public API
exports, type contracts, and configuration schema. If a refactor breaks
the import graph or removes a public symbol, these tests catch it.

Run automatically on PR to main via CI.
"""

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest


# ── Package layout ──────────────────────────────────────────────────────────

EXPECTED_MODULES = [
    "sfl",
    "sfl.types",
    "sfl.client",
    "sfl.client.base",
    "sfl.client.sum_client",
    "sfl.server",
    "sfl.server.app",
    "sfl.server.strategy",
    "sfl.utils",
    "sfl.utils.config",
    "sfl.utils.logging",
]


class TestPackageStructure:
    """Verify that all expected modules are importable."""

    @pytest.mark.parametrize("module_name", EXPECTED_MODULES)
    def test_module_importable(self, module_name):
        mod = importlib.import_module(module_name)
        assert mod is not None


class TestPublicAPI:
    """Verify public symbols exported from each package."""

    def test_sfl_root_exports(self):
        import sfl

        assert hasattr(sfl, "__version__")
        assert hasattr(sfl, "FederationConfig")
        assert hasattr(sfl, "ClientConfig")
        assert hasattr(sfl, "ServerConfig")

    def test_client_package_exports(self):
        from sfl.client import BaseFederatedClient, SumClient, client_fn, app

        assert inspect.isclass(BaseFederatedClient)
        assert inspect.isclass(SumClient)
        assert callable(client_fn)
        assert app is not None

    def test_server_package_exports(self):
        from sfl.server import SumFedAvg, server_fn, app

        assert inspect.isclass(SumFedAvg)
        assert callable(server_fn)
        assert app is not None

    def test_utils_package_exports(self):
        from sfl.utils import load_config, get_config, get_logger, setup_logging

        assert callable(load_config)
        assert callable(get_config)
        assert callable(get_logger)
        assert callable(setup_logging)


# ── Type system contracts ───────────────────────────────────────────────────

class TestTypeContracts:
    """Verify type aliases and dataclass fields haven't shifted."""

    def test_type_aliases_exist(self):
        from sfl.types import Parameters, Metrics, Config, ClientUpdate

        # Verify they are valid type annotations (not None)
        assert Parameters is not None
        assert Metrics is not None
        assert Config is not None
        assert ClientUpdate is not None

    def test_federation_config_fields(self):
        from sfl.types import FederationConfig
        import dataclasses

        fields = {f.name for f in dataclasses.fields(FederationConfig)}
        assert fields >= {
            "num_clients",
            "num_rounds",
            "min_available_clients",
            "min_fit_clients",
        }

    def test_client_config_fields(self):
        from sfl.types import ClientConfig
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ClientConfig)}
        assert "base_secret" in fields

    def test_server_config_fields(self):
        from sfl.types import ServerConfig
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ServerConfig)}
        assert "initial_param" in fields

    def test_sfl_config_fields(self):
        from sfl.types import SFLConfig
        import dataclasses

        fields = {f.name for f in dataclasses.fields(SFLConfig)}
        assert fields >= {"federation", "client", "server", "nvflare", "logging"}

    def test_federation_config_defaults(self):
        from sfl.types import FederationConfig

        cfg = FederationConfig()
        assert cfg.num_clients == 2
        assert cfg.num_rounds == 1
        assert cfg.min_available_clients == 2
        assert cfg.min_fit_clients == 2

    def test_client_config_defaults(self):
        from sfl.types import ClientConfig

        cfg = ClientConfig()
        assert cfg.base_secret == 7.0

    def test_server_config_defaults(self):
        from sfl.types import ServerConfig

        cfg = ServerConfig()
        assert cfg.initial_param == 0.0


# ── Inheritance contracts ───────────────────────────────────────────────────

class TestInheritance:
    """Verify the class hierarchy is intact."""

    def test_sum_client_extends_base(self):
        from sfl.client.base import BaseFederatedClient
        from sfl.client.sum_client import SumClient

        assert issubclass(SumClient, BaseFederatedClient)

    def test_base_client_is_abstract(self):
        from sfl.client.base import BaseFederatedClient

        assert inspect.isabstract(BaseFederatedClient)

    def test_base_client_has_compute_update(self):
        from sfl.client.base import BaseFederatedClient

        assert hasattr(BaseFederatedClient, "compute_update")

    def test_sum_fedavg_extends_fedavg(self):
        from flwr.server.strategy import FedAvg
        from sfl.server.strategy import SumFedAvg

        assert issubclass(SumFedAvg, FedAvg)


# ── Configuration files ────────────────────────────────────────────────────

class TestConfigFiles:
    """Verify config files exist and have expected structure."""

    def test_default_yaml_exists(self):
        root = Path(__file__).parent.parent
        assert (root / "config" / "default.yaml").is_file()

    def test_default_yaml_has_required_sections(self):
        import yaml

        root = Path(__file__).parent.parent
        with open(root / "config" / "default.yaml") as f:
            cfg = yaml.safe_load(f)

        assert "federation" in cfg
        assert "client" in cfg
        assert "server" in cfg
        assert "logging" in cfg

    def test_pyproject_toml_exists(self):
        root = Path(__file__).parent.parent
        assert (root / "pyproject.toml").is_file()
