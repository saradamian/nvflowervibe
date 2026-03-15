"""Tests for HPC production utilities: gRPC auth, checkpointing, metrics, resources."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sfl.utils.grpc_auth import (
    TLSConfig,
    TokenAuthConfig,
    load_tls_certificates,
    tls_config_from_env,
    token_config_from_env,
)
from sfl.utils.checkpoint import CheckpointManager
from sfl.utils.metrics import MetricsCollector, save_metrics_csv, save_metrics_json
from sfl.utils.resources import (
    ClientResources,
    ResourceConfig,
    build_backend_config,
    detect_resources,
)


# ── TLS Config ──────────────────────────────────────────────────────


class TestTLSConfig:
    def test_defaults(self):
        cfg = TLSConfig()
        assert cfg.ca_cert == ""
        assert cfg.server_cert == ""

    def test_load_missing_ca_raises(self):
        cfg = TLSConfig(ca_cert="/nonexistent/ca.pem")
        with pytest.raises(FileNotFoundError, match="CA certificate"):
            load_tls_certificates(cfg, role="server")

    def test_invalid_role_raises(self):
        cfg = TLSConfig(ca_cert="ca.pem")
        with pytest.raises(ValueError, match="role"):
            load_tls_certificates(cfg, role="admin")

    def test_load_certificates_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            ca = Path(d) / "ca.pem"
            cert = Path(d) / "server.pem"
            key = Path(d) / "server.key"
            ca.write_bytes(b"CA-DATA")
            cert.write_bytes(b"CERT-DATA")
            key.write_bytes(b"KEY-DATA")

            cfg = TLSConfig(
                ca_cert=str(ca),
                server_cert=str(cert),
                server_key=str(key),
            )
            ca_bytes, cert_bytes, key_bytes = load_tls_certificates(cfg, role="server")
            assert ca_bytes == b"CA-DATA"
            assert cert_bytes == b"CERT-DATA"
            assert key_bytes == b"KEY-DATA"

    def test_load_client_certificates(self):
        with tempfile.TemporaryDirectory() as d:
            ca = Path(d) / "ca.pem"
            cert = Path(d) / "client.pem"
            key = Path(d) / "client.key"
            ca.write_bytes(b"CA")
            cert.write_bytes(b"CLIENT-CERT")
            key.write_bytes(b"CLIENT-KEY")

            cfg = TLSConfig(
                ca_cert=str(ca),
                client_cert=str(cert),
                client_key=str(key),
            )
            ca_b, cert_b, key_b = load_tls_certificates(cfg, role="client")
            assert cert_b == b"CLIENT-CERT"

    def test_tls_config_from_env_returns_none(self):
        # No SFL_TLS_CA_CERT set
        for k in list(os.environ):
            if k.startswith("SFL_TLS_"):
                del os.environ[k]
        assert tls_config_from_env() is None

    def test_tls_config_from_env_reads_vars(self):
        os.environ["SFL_TLS_CA_CERT"] = "/path/ca.pem"
        os.environ["SFL_TLS_SERVER_CERT"] = "/path/server.pem"
        try:
            cfg = tls_config_from_env()
            assert cfg is not None
            assert cfg.ca_cert == "/path/ca.pem"
            assert cfg.server_cert == "/path/server.pem"
        finally:
            del os.environ["SFL_TLS_CA_CERT"]
            del os.environ["SFL_TLS_SERVER_CERT"]


class TestTokenAuthConfig:
    def test_defaults(self):
        cfg = TokenAuthConfig()
        assert cfg.token == ""
        assert cfg.header_key == "x-sfl-auth-token"

    def test_token_config_from_env_returns_none(self):
        os.environ.pop("SFL_AUTH_TOKEN", None)
        assert token_config_from_env() is None

    def test_token_config_from_env_reads_vars(self):
        os.environ["SFL_AUTH_TOKEN"] = "secret123"
        try:
            cfg = token_config_from_env()
            assert cfg is not None
            assert cfg.token == "secret123"
        finally:
            del os.environ["SFL_AUTH_TOKEN"]


# ── Checkpoint Manager ──────────────────────────────────────────────


class TestCheckpointManager:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            params = [np.array([1.0, 2.0]), np.array([3.0])]
            metrics = {"loss": 0.5, "accuracy": 0.9}

            mgr.save_round(5, params, metrics)

            result = mgr.load_latest()
            assert result is not None
            round_num, loaded_params, loaded_metrics = result
            assert round_num == 5
            assert len(loaded_params) == 2
            np.testing.assert_array_equal(loaded_params[0], params[0])
            assert loaded_metrics["loss"] == 0.5

    def test_load_latest_returns_none_on_empty(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            assert mgr.load_latest() is None

    def test_multiple_rounds_loads_latest(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            mgr.save_round(1, [np.array([1.0])], {"round": 1})
            mgr.save_round(3, [np.array([3.0])], {"round": 3})
            mgr.save_round(2, [np.array([2.0])], {"round": 2})

            round_num, _, metrics = mgr.load_latest()
            assert round_num == 3

    def test_cleanup_keeps_last_n(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            for i in range(5):
                mgr.save_round(i, [np.array([float(i)])], {})

            mgr.cleanup(keep_last_n=2)

            # Only rounds 3 and 4 should remain
            dirs = sorted(Path(d).glob("round_*"))
            assert len(dirs) == 2
            assert "round_0003" in dirs[0].name
            assert "round_0004" in dirs[1].name


# ── Metrics Collector ───────────────────────────────────────────────


class TestMetricsCollector:
    def test_record_and_get_history(self):
        mc = MetricsCollector()
        mc.record_round(1, {"loss": 0.5, "accuracy": 0.8})
        mc.record_round(2, {"loss": 0.3, "accuracy": 0.9})

        history = mc.get_history()
        assert history["loss"] == [0.5, 0.3]
        assert history["accuracy"] == [0.8, 0.9]

    def test_aggregate_client_metrics(self):
        mc = MetricsCollector()
        client_metrics = [
            {"loss": 0.4, "n_samples": 100},
            {"loss": 0.6, "n_samples": 200},
        ]
        agg = mc.aggregate_client_metrics(client_metrics)
        assert agg["loss"] == pytest.approx(0.5)

    def test_summary(self):
        mc = MetricsCollector()
        mc.record_round(1, {"loss": 0.5})
        mc.record_round(2, {"loss": 0.3})
        mc.record_round(3, {"loss": 0.1})

        s = mc.summary()
        assert "loss" in s
        assert s["loss"]["latest"] == 0.1
        assert s["loss"]["min"] == 0.1
        assert s["loss"]["max"] == 0.5

    def test_save_csv(self):
        mc = MetricsCollector()
        mc.record_round(1, {"loss": 0.5, "acc": 0.8})
        mc.record_round(2, {"loss": 0.3, "acc": 0.9})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            save_metrics_csv(mc, f.name)
            content = Path(f.name).read_text()
            os.unlink(f.name)

        assert "loss" in content
        assert "0.5" in content
        assert "0.3" in content

    def test_save_json(self):
        mc = MetricsCollector()
        mc.record_round(1, {"loss": 0.5})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_metrics_json(mc, f.name)
            data = json.loads(Path(f.name).read_text())
            os.unlink(f.name)

        assert "history" in data
        assert "loss" in data["history"]
        assert len(data["history"]["loss"]) == 1


# ── Resource Config ─────────────────────────────────────────────────


class TestResources:
    def test_client_resources_defaults(self):
        r = ClientResources()
        assert r.num_cpus == 1
        assert r.num_gpus == 0.0

    def test_detect_resources_returns_valid(self):
        r = detect_resources()
        assert isinstance(r, ClientResources)
        assert r.num_cpus >= 1

    def test_build_backend_config_cpu_only(self):
        cfg = ResourceConfig(
            default=ClientResources(num_cpus=2, num_gpus=0.0),
            auto_detect=False,
        )
        bc = build_backend_config(cfg, num_clients=4)
        assert "client_resources" in bc
        assert bc["client_resources"]["num_cpus"] == 2
        assert bc["client_resources"]["num_gpus"] == 0.0

    def test_build_backend_config_with_gpus(self):
        cfg = ResourceConfig(
            default=ClientResources(num_cpus=1, num_gpus=0.5),
            auto_detect=False,
        )
        bc = build_backend_config(cfg, num_clients=4)
        assert bc["client_resources"]["num_gpus"] == 0.5

    def test_resource_config_auto_detect(self):
        cfg = ResourceConfig(auto_detect=True)
        bc = build_backend_config(cfg, num_clients=2)
        assert "client_resources" in bc

    def test_exported_from_utils_init(self):
        from sfl.utils import (
            TLSConfig,
            TokenAuthConfig,
            CheckpointManager,
            MetricsCollector,
            ClientResources,
            ResourceConfig,
        )
        assert TLSConfig is not None
        assert CheckpointManager is not None
