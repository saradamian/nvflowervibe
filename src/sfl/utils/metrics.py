"""
Metric aggregation and export utilities for federated learning.

Collects per-round server and client metrics, provides aggregation
helpers, and exports to CSV, JSON, or TensorBoard for HPC dashboards
and experiment tracking.

Usage::

    from sfl.utils.metrics import MetricsCollector, save_metrics_csv

    collector = MetricsCollector()
    collector.record_round(1, {"loss": 0.5}, [{"loss": 0.6}, {"loss": 0.4}])
    save_metrics_csv(collector, "metrics.csv")
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and aggregates per-round federated learning metrics.

    Stores server-level and (optionally) client-level metrics for each
    training round. Provides history retrieval and summary statistics
    for dashboards and experiment tracking.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        export_format: str = "csv",
    ) -> None:
        self._rounds: List[int] = []
        self._server_metrics: List[Dict[str, Any]] = []
        self._client_metrics: List[Optional[List[Dict[str, Any]]]] = []
        self._output_dir = output_dir
        self._export_format = export_format

    def record_round(
        self,
        round_num: int,
        server_metrics: Dict[str, Any],
        client_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store metrics for a single federated round.

        Args:
            round_num: The round number (1-indexed).
            server_metrics: Metrics returned by the server strategy.
            client_metrics: Optional list of per-client metric dicts.
                If provided, they are aggregated and merged into the
                server metrics under ``client_*`` prefixed keys.
        """
        merged = dict(server_metrics)
        if client_metrics:
            agg = self.aggregate_client_metrics(client_metrics)
            for key, value in agg.items():
                merged[f"client_{key}"] = value

        self._rounds.append(round_num)
        self._server_metrics.append(merged)
        self._client_metrics.append(client_metrics)

    def get_history(self) -> Dict[str, List]:
        """Return metric history keyed by metric name.

        Returns:
            Dict mapping each metric name to a list of per-round values.
            Missing values are filled with ``None``.
        """
        all_keys: set = set()
        for m in self._server_metrics:
            all_keys.update(m.keys())

        history: Dict[str, List] = {"round": list(self._rounds)}
        for key in sorted(all_keys):
            history[key] = [m.get(key) for m in self._server_metrics]
        return history

    @staticmethod
    def aggregate_client_metrics(client_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate a list of client metric dicts into a single summary.

        Numeric values are averaged. Non-numeric values are counted
        (most-common value is kept, with a ``_count`` suffix).

        Args:
            client_metrics: List of metric dicts, one per client.

        Returns:
            Aggregated metrics dict.
        """
        if not client_metrics:
            return {}

        numeric: Dict[str, List[float]] = defaultdict(list)
        non_numeric: Dict[str, List] = defaultdict(list)

        for cm in client_metrics:
            for key, value in cm.items():
                if isinstance(value, (int, float)):
                    numeric[key].append(float(value))
                else:
                    non_numeric[key].append(value)

        result: Dict[str, Any] = {}
        for key, values in numeric.items():
            result[key] = sum(values) / len(values)
        for key, values in non_numeric.items():
            result[f"{key}_count"] = len(values)

        return result

    def summary(self) -> Dict[str, Any]:
        """Return a summary with latest, min, max, and mean per metric.

        Returns:
            Dict with structure ``{metric_name: {latest, min, max, mean}}``.
        """
        history = self.get_history()
        result: Dict[str, Any] = {}

        for key, values in history.items():
            nums = [v for v in values if isinstance(v, (int, float))]
            if not nums:
                continue
            result[key] = {
                "latest": nums[-1],
                "min": min(nums),
                "max": max(nums),
                "mean": sum(nums) / len(nums),
            }

        return result

    def export(self) -> None:
        """Export collected metrics to the configured output directory.

        Uses ``output_dir`` and ``export_format`` from ``__init__``.
        Supported formats: ``csv``, ``json``, ``tensorboard``, ``all``.
        No-op if ``output_dir`` was not set.
        """
        if not self._output_dir or not self._rounds:
            return

        fmt = self._export_format.lower()
        if fmt in ("csv", "all"):
            save_metrics_csv(self, os.path.join(self._output_dir, "metrics.csv"))
        if fmt in ("json", "all"):
            save_metrics_json(self, os.path.join(self._output_dir, "metrics.json"))
        if fmt in ("tensorboard", "all"):
            save_metrics_tensorboard(self, self._output_dir)


# ── Export functions ─────────────────────────────────────────────────────


def save_metrics_csv(collector: MetricsCollector, filepath: str) -> None:
    """Export collected metrics to a CSV file (one row per round).

    Args:
        collector: A populated MetricsCollector.
        filepath: Destination file path.
    """
    history = collector.get_history()
    if not history.get("round"):
        logger.warning("No metrics to export — skipping CSV write")
        return

    columns = list(history.keys())
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        num_rows = len(history["round"])
        for i in range(num_rows):
            writer.writerow([history[col][i] for col in columns])

    logger.info("Metrics saved to %s (%d rounds)", filepath, num_rows)


def save_metrics_json(collector: MetricsCollector, filepath: str) -> None:
    """Export collected metrics to a JSON file.

    Args:
        collector: A populated MetricsCollector.
        filepath: Destination file path.
    """
    history = collector.get_history()
    if not history.get("round"):
        logger.warning("No metrics to export — skipping JSON write")
        return

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    payload = {
        "history": history,
        "summary": collector.summary(),
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Metrics saved to %s (%d rounds)", filepath, len(history["round"]))


def save_metrics_tensorboard(collector: MetricsCollector, log_dir: str) -> None:
    """Export collected metrics to TensorBoard event files.

    Requires ``tensorboard`` to be installed. Silently skips if the
    dependency is missing.

    Args:
        collector: A populated MetricsCollector.
        log_dir: TensorBoard log directory.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning(
            "tensorboard not installed — skipping TensorBoard export. "
            "Install with: pip install tensorboard"
        )
        return

    history = collector.get_history()
    if not history.get("round"):
        logger.warning("No metrics to export — skipping TensorBoard write")
        return

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    rounds = history["round"]
    for key, values in history.items():
        if key == "round":
            continue
        for step, value in zip(rounds, values):
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, global_step=step)

    writer.close()
    logger.info("Metrics saved to TensorBoard at %s (%d rounds)", log_dir, len(rounds))


# ── Strategy wrapper ────────────────────────────────────────────────────


def make_metrics_strategy(strategy: Strategy, collector: MetricsCollector) -> Strategy:
    """Wrap a Flower Strategy to auto-collect metrics.

    Intercepts ``aggregate_fit`` and ``aggregate_evaluate`` to record
    returned metrics into the collector. Follows the same delegation
    pattern as ``_AccountingWrapper`` in ``sfl.privacy.dp``.

    Args:
        strategy: Any Flower Strategy instance.
        collector: MetricsCollector to record into.

    Returns:
        A wrapped Strategy that transparently collects metrics.
    """
    return _MetricsWrapper(strategy, collector)


class _MetricsWrapper(Strategy):
    """Thin wrapper that records metrics from aggregate_fit / aggregate_evaluate."""

    def __init__(self, strategy: Strategy, collector: MetricsCollector) -> None:
        super().__init__()
        self._inner = strategy
        self._collector = collector

    def __repr__(self) -> str:
        return f"_MetricsWrapper({self._inner!r})"

    # ── Intercept aggregate_fit ──────────────────────────────────────────

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = self._inner.aggregate_fit(server_round, results, failures)

        if metrics:
            client_metrics = self._extract_client_metrics(results)
            self._collector.record_round(server_round, metrics, client_metrics)
            self._collector.export()

        return params, metrics

    # ── Intercept aggregate_evaluate ─────────────────────────────────────

    def aggregate_evaluate(self, server_round, results, failures):
        result = self._inner.aggregate_evaluate(server_round, results, failures)

        if result is not None:
            loss, metrics = result
            if metrics:
                eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
                eval_metrics["eval_loss"] = loss
                self._collector.record_round(server_round, eval_metrics)

        return result

    # ── Delegate everything else ─────────────────────────────────────────

    def initialize_parameters(self, client_manager):
        return self._inner.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self._inner.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self._inner.configure_evaluate(server_round, parameters, client_manager)

    def evaluate(self, server_round, parameters):
        return self._inner.evaluate(server_round, parameters)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_client_metrics(results) -> Optional[List[Dict[str, Any]]]:
        """Pull per-client metrics from fit results, if any."""
        client_metrics = []
        for _, fit_res in results:
            if fit_res.metrics:
                client_metrics.append(dict(fit_res.metrics))
        return client_metrics or None
