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


# ── MetricsCollector with output_dir ──────────────────────────────


class TestMetricsCollectorExport:
    def test_init_accepts_output_dir_and_format(self):
        mc = MetricsCollector(output_dir="/tmp/test", export_format="json")
        assert mc._output_dir == "/tmp/test"
        assert mc._export_format == "json"

    def test_export_writes_csv(self):
        with tempfile.TemporaryDirectory() as d:
            mc = MetricsCollector(output_dir=d, export_format="csv")
            mc.record_round(1, {"loss": 0.5})
            mc.export()
            csv_path = Path(d) / "metrics.csv"
            assert csv_path.exists()
            content = csv_path.read_text()
            assert "loss" in content
            assert "0.5" in content

    def test_export_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            mc = MetricsCollector(output_dir=d, export_format="json")
            mc.record_round(1, {"loss": 0.3})
            mc.export()
            json_path = Path(d) / "metrics.json"
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert "history" in data

    def test_export_noop_without_output_dir(self):
        mc = MetricsCollector()
        mc.record_round(1, {"loss": 0.5})
        mc.export()  # should not raise

    def test_export_noop_without_data(self):
        with tempfile.TemporaryDirectory() as d:
            mc = MetricsCollector(output_dir=d, export_format="csv")
            mc.export()  # no data — should not write
            assert not (Path(d) / "metrics.csv").exists()


# ── build_strategy tests ──────────────────────────────────────────


class TestBuildStrategy:
    """Test build_strategy wiring for resume, metrics, and aggregation."""

    def _clean_env(self):
        """Remove SFL_* env vars to isolate tests."""
        for key in list(os.environ):
            if key.startswith("SFL_"):
                del os.environ[key]

    def test_default_strategy_is_fedavg(self):
        from flwr.common import ndarrays_to_parameters
        from flwr.server.strategy import FedAvg
        from sfl.server.dp_setup import build_strategy

        self._clean_env()
        params = ndarrays_to_parameters([np.array([1.0, 2.0])])
        strategy = build_strategy(
            initial_parameters=params,
            num_clients=2,
            run_config={},
        )
        # Should be FedAvg (no wrapping since no checkpoint/metrics/dp)
        assert isinstance(strategy, FedAvg)

    def test_krum_aggregation(self):
        from flwr.common import ndarrays_to_parameters
        from sfl.server.dp_setup import build_strategy
        from sfl.server.robust import MultiKrumFedAvg

        self._clean_env()
        os.environ["SFL_AGGREGATION"] = "krum"
        os.environ["SFL_KRUM_BYZANTINE"] = "1"
        params = ndarrays_to_parameters([np.array([1.0, 2.0])])
        strategy = build_strategy(
            initial_parameters=params,
            num_clients=4,
            run_config={},
        )
        assert isinstance(strategy, MultiKrumFedAvg)
        self._clean_env()

    def test_metrics_wrapping(self):
        from flwr.common import ndarrays_to_parameters
        from sfl.server.dp_setup import build_strategy

        self._clean_env()
        with tempfile.TemporaryDirectory() as d:
            os.environ["SFL_METRICS_DIR"] = d
            os.environ["SFL_METRICS_FORMAT"] = "csv"
            params = ndarrays_to_parameters([np.array([1.0])])
            strategy = build_strategy(
                initial_parameters=params,
                num_clients=2,
                run_config={},
            )
            # Should be wrapped in _MetricsWrapper
            assert hasattr(strategy, "_collector")
            assert strategy._collector._output_dir == d
        self._clean_env()

    def test_checkpoint_wrapping(self):
        from flwr.common import ndarrays_to_parameters
        from sfl.server.dp_setup import build_strategy

        self._clean_env()
        with tempfile.TemporaryDirectory() as d:
            os.environ["SFL_CHECKPOINT_DIR"] = d
            params = ndarrays_to_parameters([np.array([1.0])])
            strategy = build_strategy(
                initial_parameters=params,
                num_clients=2,
                run_config={},
            )
            # Should be wrapped in _CheckpointWrapper
            assert hasattr(strategy, "_mgr")
        self._clean_env()

    def test_resume_restores_parameters(self):
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from sfl.server.dp_setup import build_strategy

        self._clean_env()
        with tempfile.TemporaryDirectory() as d:
            # Save a checkpoint
            mgr = CheckpointManager(d)
            saved_params = [np.array([99.0, 88.0])]
            mgr.save_round(5, saved_params, {"loss": 0.1})

            os.environ["SFL_CHECKPOINT_DIR"] = d
            os.environ["SFL_RESUME"] = "true"

            # Build with different initial params
            initial = ndarrays_to_parameters([np.array([1.0, 2.0])])
            strategy = build_strategy(
                initial_parameters=initial,
                num_clients=2,
                run_config={},
            )
            # The inner strategy should have the checkpoint params
            inner = strategy._inner if hasattr(strategy, "_inner") else strategy
            result_params = parameters_to_ndarrays(inner.initial_parameters)
            np.testing.assert_array_equal(result_params[0], saved_params[0])

            # SFL_RESUME_ROUND should be set
            assert os.environ.get("SFL_RESUME_ROUND") == "5"
        self._clean_env()

    def test_resume_no_checkpoint_keeps_initial(self):
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from sfl.server.dp_setup import build_strategy

        self._clean_env()
        with tempfile.TemporaryDirectory() as d:
            os.environ["SFL_CHECKPOINT_DIR"] = d
            os.environ["SFL_RESUME"] = "true"

            initial = ndarrays_to_parameters([np.array([1.0, 2.0])])
            strategy = build_strategy(
                initial_parameters=initial,
                num_clients=2,
                run_config={},
            )
            inner = strategy._inner if hasattr(strategy, "_inner") else strategy
            result_params = parameters_to_ndarrays(inner.initial_parameters)
            np.testing.assert_array_equal(result_params[0], [1.0, 2.0])
        self._clean_env()


# ── HPC script correctness ────────────────────────────────────────


class TestHPCScriptCommands:
    """Verify HPC scripts use Flower 1.17.0 CLI commands and flags."""

    SCRIPT_DIR = Path(__file__).parent.parent / "examples" / "hpc"

    def test_server_uses_superlink_and_serverapp(self):
        """Server script must use flower-superlink + flwr-serverapp."""
        content = (self.SCRIPT_DIR / "submit_server.sbatch").read_text()
        assert "flower-superlink" in content
        assert "flwr-serverapp" in content
        # Must NOT use the old commands
        assert "flower-server-app" not in content
        assert "flower-client-app" not in content

    def test_client_uses_supernode_and_clientapp(self):
        """Client script must use flower-supernode + flwr-clientapp."""
        content = (self.SCRIPT_DIR / "submit_client.sbatch").read_text()
        assert "flower-supernode" in content
        assert "flwr-clientapp" in content
        # Must NOT use the old commands
        assert "flower-server-app" not in content
        assert "flower-client-app" not in content

    def test_server_superlink_uses_fleet_api_address(self):
        """SuperLink should use --fleet-api-address, not --server-address."""
        content = (self.SCRIPT_DIR / "submit_server.sbatch").read_text()
        assert "--fleet-api-address" in content
        assert "--server-address" not in content

    def test_client_supernode_uses_superlink_flag(self):
        """SuperNode should use --superlink, not --server-address."""
        content = (self.SCRIPT_DIR / "submit_client.sbatch").read_text()
        assert "--superlink" in content

    def test_server_tls_is_fail_closed(self):
        """Server should fail (exit 1) when certs are missing, not fall back to --insecure."""
        content = (self.SCRIPT_DIR / "submit_server.sbatch").read_text()
        assert "exit 1" in content
        # Should have SFL_INSECURE check as explicit opt-in
        assert "SFL_INSECURE" in content

    def test_client_tls_is_fail_closed(self):
        """Client should fail (exit 1) when certs are missing, not fall back to --insecure."""
        content = (self.SCRIPT_DIR / "submit_client.sbatch").read_text()
        assert "exit 1" in content
        assert "SFL_INSECURE" in content

    def test_launcher_propagates_privacy_vars_to_clients(self):
        """launch_federation.sh must pass SFL_* vars to both server and client jobs."""
        content = (self.SCRIPT_DIR / "launch_federation.sh").read_text()
        # SFL_EXPORT should include privacy vars
        assert "SFL_SECAGG_ENABLED" in content
        assert "SFL_DP_ENABLED" in content
        # The client sbatch block (multi-line) must reference $SFL_EXPORT
        # Find the section between "Submit clients" and the next section
        client_section_start = content.find("Submit clients")
        assert client_section_start != -1, "Missing 'Submit clients' section"
        client_section = content[client_section_start:client_section_start + 500]
        assert "SFL_EXPORT" in client_section, (
            "Client sbatch must use $SFL_EXPORT for privacy var propagation"
        )

    def test_launcher_rejects_unknown_args(self):
        """launch_federation.sh should error on unknown args, not silently drop them."""
        content = (self.SCRIPT_DIR / "launch_federation.sh").read_text()
        assert "Unknown argument" in content
