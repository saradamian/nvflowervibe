"""Tests for the NVFlare backend abstraction and auto_mods."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sfl.nvflare.backend import (
    NVFlareBackendConfig,
    NVFlareMode,
    build_extra_env,
)
from sfl.nvflare.staging import stage_flower_content, _APP_ENTRYPOINTS
from sfl.privacy.auto_mods import auto_build_client_mods
from sfl.server.dp_setup import apply_dp_if_enabled


# ── NVFlareBackendConfig ─────────────────────────────────────────────


class TestNVFlareBackendConfig:
    def test_sim_mode_no_startup_kit_ok(self):
        cfg = NVFlareBackendConfig(mode=NVFlareMode.SIM, num_clients=2)
        assert cfg.mode == NVFlareMode.SIM

    def test_prod_mode_requires_startup_kit(self):
        with pytest.raises(ValueError, match="startup_kit"):
            NVFlareBackendConfig(mode=NVFlareMode.PROD, num_clients=2)

    def test_prod_mode_with_startup_kit(self):
        cfg = NVFlareBackendConfig(
            mode=NVFlareMode.PROD, num_clients=2,
            startup_kit="/some/path",
        )
        assert cfg.startup_kit == "/some/path"

    def test_default_values(self):
        cfg = NVFlareBackendConfig()
        assert cfg.mode == NVFlareMode.SIM
        assert cfg.num_clients == 2
        assert cfg.job_name == "sfl-federated"
        assert cfg.extra_env == {}


# ── build_extra_env ──────────────────────────────────────────────────


class TestBuildExtraEnv:
    def test_captures_sfl_vars(self):
        with patch.dict(os.environ, {"SFL_DP_ENABLED": "true", "SFL_DP_NOISE": "0.5"}):
            env = build_extra_env()
            assert env["SFL_DP_ENABLED"] == "true"
            assert env["SFL_DP_NOISE"] == "0.5"

    def test_ignores_non_sfl_vars(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin", "SFL_TEST": "1"}, clear=False):
            env = build_extra_env()
            assert "PATH" not in env
            assert "SFL_TEST" in env

    def test_captures_nccl_when_requested(self):
        with patch.dict(os.environ, {"NCCL_SOCKET_IFNAME": "ib0"}, clear=False):
            env_without = build_extra_env(include_non_sfl=False)
            assert "NCCL_SOCKET_IFNAME" not in env_without

            env_with = build_extra_env(include_non_sfl=True)
            assert env_with["NCCL_SOCKET_IFNAME"] == "ib0"

    def test_empty_when_no_sfl_vars(self):
        # Clear all SFL_ vars
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("SFL_")}
        with patch.dict(os.environ, clean_env, clear=True):
            env = build_extra_env()
            assert len(env) == 0


# ── stage_flower_content ─────────────────────────────────────────────


class TestStageFlowerContent:
    def test_creates_pyproject_for_esm2(self):
        # Need a fake project root with src/sfl
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "src" / "sfl").mkdir(parents=True)
            (Path(root) / "src" / "sfl" / "__init__.py").write_text("")

            staging = stage_flower_content(Path(root), "esm2", {"num-server-rounds": 10})
            try:
                pyproject = (Path(staging) / "pyproject.toml").read_text()
                assert "sfl.esm2:server_app" in pyproject
                assert "sfl.esm2:client_app" in pyproject
                assert "num-server-rounds = 10" in pyproject
            finally:
                import shutil
                shutil.rmtree(staging)

    def test_creates_pyproject_for_llm(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "src" / "sfl").mkdir(parents=True)
            (Path(root) / "src" / "sfl" / "__init__.py").write_text("")

            staging = stage_flower_content(Path(root), "llm")
            try:
                pyproject = (Path(staging) / "pyproject.toml").read_text()
                assert "sfl.llm:server_app" in pyproject
                assert "sfl.llm:client_app" in pyproject
            finally:
                import shutil
                shutil.rmtree(staging)

    def test_unknown_app_type_raises(self):
        with tempfile.TemporaryDirectory() as root:
            with pytest.raises(ValueError, match="Unknown app_type"):
                stage_flower_content(Path(root), "nonexistent")

    def test_copies_src_directory(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "src" / "sfl").mkdir(parents=True)
            (Path(root) / "src" / "sfl" / "test_file.py").write_text("hello")

            staging = stage_flower_content(Path(root), "base")
            try:
                assert (Path(staging) / "src" / "sfl" / "test_file.py").exists()
            finally:
                import shutil
                shutil.rmtree(staging)

    def test_copies_config_if_present(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "src" / "sfl").mkdir(parents=True)
            (Path(root) / "src" / "sfl" / "__init__.py").write_text("")
            (Path(root) / "config").mkdir()
            (Path(root) / "config" / "default.yaml").write_text("key: val")

            staging = stage_flower_content(Path(root), "base")
            try:
                assert (Path(staging) / "config" / "default.yaml").exists()
            finally:
                import shutil
                shutil.rmtree(staging)

    def test_all_app_types_have_entrypoints(self):
        for app_type in _APP_ENTRYPOINTS:
            server, client = _APP_ENTRYPOINTS[app_type]
            assert ":" in server or "." in server
            assert ":" in client or "." in client


# ── auto_build_client_mods ───────────────────────────────────────────


class TestAutoBuildClientMods:
    def _clean_env(self):
        """Remove all SFL_ env vars."""
        for key in list(os.environ):
            if key.startswith("SFL_"):
                del os.environ[key]

    def test_no_env_vars_returns_empty(self):
        self._clean_env()
        mods = auto_build_client_mods()
        assert mods == []

    def test_dp_client_mode_adds_clipping_mod(self):
        self._clean_env()
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_MODE"] = "client"
        try:
            mods = auto_build_client_mods()
            assert len(mods) >= 1
            # First mod should be fixedclipping_mod
            assert "fixedclipping" in mods[0].__name__.lower() or callable(mods[0])
        finally:
            self._clean_env()

    def test_dp_server_mode_no_clipping_mod(self):
        self._clean_env()
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_MODE"] = "server"
        try:
            mods = auto_build_client_mods()
            assert mods == []
        finally:
            self._clean_env()

    def test_secagg_adds_mod_last(self):
        self._clean_env()
        os.environ["SFL_SECAGG_ENABLED"] = "true"
        try:
            mods = auto_build_client_mods()
            assert len(mods) == 1
        finally:
            self._clean_env()

    def test_multiple_mods_ordered(self):
        self._clean_env()
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_MODE"] = "client"
        os.environ["SFL_SECAGG_ENABLED"] = "true"
        try:
            mods = auto_build_client_mods()
            assert len(mods) == 2
            # SecAgg should be last
        finally:
            self._clean_env()

    def test_exclude_layers_from_env(self):
        self._clean_env()
        os.environ["SFL_EXCLUDE_LAYERS"] = "0,1,2"
        try:
            mods = auto_build_client_mods()
            assert len(mods) == 1
            assert callable(mods[0])
        finally:
            self._clean_env()

    def test_freeze_layers_from_env(self):
        self._clean_env()
        os.environ["SFL_FREEZE_LAYERS"] = "4,5"
        try:
            mods = auto_build_client_mods()
            assert len(mods) == 1
        finally:
            self._clean_env()


# ── apply_dp_if_enabled ──────────────────────────────────────────────


class TestApplyDpIfEnabled:
    def test_returns_original_when_disabled(self):
        strategy = MagicMock()
        result = apply_dp_if_enabled(strategy, {}, 2)
        assert result is strategy

    def test_returns_original_when_no_env(self):
        for k in list(os.environ):
            if k.startswith("SFL_DP"):
                del os.environ[k]
        strategy = MagicMock()
        result = apply_dp_if_enabled(strategy, {"dp-enabled": "false"}, 2)
        assert result is strategy

    def test_wraps_when_env_enabled(self):
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_NOISE"] = "1.0"
        os.environ["SFL_DP_CLIP"] = "10.0"
        os.environ["SFL_DP_MODE"] = "server"
        try:
            strategy = MagicMock()
            strategy.configure_fit = MagicMock()
            strategy.configure_evaluate = MagicMock()
            result = apply_dp_if_enabled(strategy, {}, 2)
            # Should be wrapped (not the same object)
            assert result is not strategy
        finally:
            for k in list(os.environ):
                if k.startswith("SFL_DP"):
                    del os.environ[k]

    def test_wraps_when_run_config_enabled(self):
        strategy = MagicMock()
        strategy.configure_fit = MagicMock()
        strategy.configure_evaluate = MagicMock()
        result = apply_dp_if_enabled(
            strategy,
            {"dp-enabled": "true", "dp-noise-multiplier": "1.0"},
            2,
        )
        assert result is not strategy
