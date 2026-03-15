"""
Round-level checkpoint/resume for federated learning.

Saves model parameters and metadata after each round so that
training interrupted by HPC wall-time limits or crashes can
resume from the last completed round instead of restarting.

Checkpoint format::

    checkpoint_dir/
        round_0001/
            metadata.json   # round number, timestamp, metrics, config
            parameters.npz  # numpy parameter arrays
        round_0002/
            ...

All writes are atomic (write to tmp, then rename) to prevent
corruption from SIGKILL or wall-time kills mid-write.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages round-level checkpoints for federated learning.

    Thread-safe. Uses atomic writes to prevent corruption from
    wall-time kills or crashes.

    Args:
        checkpoint_dir: Directory to store checkpoint subdirectories.

    Example::

        mgr = CheckpointManager("/scratch/fl_checkpoints")
        mgr.save_round(1, parameters, {"loss": 0.5}, {"lr": 0.01})
        result = mgr.load_latest()
        if result is not None:
            round_num, params, metrics = result
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def checkpoint_dir(self) -> Path:
        return self._dir

    def _round_dir(self, round_num: int) -> Path:
        return self._dir / f"round_{round_num:04d}"

    def save_round(
        self,
        round_num: int,
        parameters: List[NDArray],
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a round checkpoint atomically.

        Writes to a temporary directory first, then renames to the
        final path. This ensures a partial write from a wall-time kill
        never leaves a corrupt checkpoint.

        Args:
            round_num: Completed round number.
            parameters: List of numpy parameter arrays.
            metrics: Round metrics dictionary.
            config: Optional training configuration snapshot.

        Returns:
            Path to the saved checkpoint directory.
        """
        with self._lock:
            target = self._round_dir(round_num)
            tmp_dir = Path(tempfile.mkdtemp(
                prefix=f".round_{round_num:04d}_",
                dir=self._dir,
            ))
            try:
                # Write metadata
                metadata = {
                    "round_num": round_num,
                    "timestamp": time.time(),
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "num_arrays": len(parameters),
                    "metrics": _sanitize_for_json(metrics),
                    "config": _sanitize_for_json(config) if config else {},
                }
                meta_path = tmp_dir / "metadata.json"
                meta_path.write_text(json.dumps(metadata, indent=2))

                # Write parameters
                npz_path = tmp_dir / "parameters.npz"
                np.savez(npz_path, *parameters)

                # Atomic swap: remove stale target if exists, then rename
                if target.exists():
                    shutil.rmtree(target)
                tmp_dir.rename(target)

                logger.info(
                    "Checkpoint saved: round=%d  arrays=%d  dir=%s",
                    round_num, len(parameters), target,
                )
                return target

            except BaseException:
                # Clean up partial tmp on failure
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

    def load_latest(self) -> Optional[Tuple[int, List[NDArray], Dict[str, Any]]]:
        """Load the most recent checkpoint.

        Returns:
            Tuple of (round_num, parameters, metrics) or ``None`` if
            no valid checkpoint exists.
        """
        with self._lock:
            rounds = self._list_rounds()
            if not rounds:
                logger.info("No checkpoints found in %s", self._dir)
                return None

            # Try from newest to oldest in case the latest is corrupt
            for round_num in reversed(rounds):
                result = self._load_round(round_num)
                if result is not None:
                    return result

            logger.warning("All checkpoints in %s are corrupt", self._dir)
            return None

    def cleanup(self, keep_last_n: int = 3) -> List[Path]:
        """Remove old checkpoints, keeping the most recent *n*.

        Args:
            keep_last_n: Number of most recent checkpoints to keep.

        Returns:
            List of removed checkpoint directories.
        """
        with self._lock:
            rounds = self._list_rounds()
            if len(rounds) <= keep_last_n:
                return []

            to_remove = rounds[:-keep_last_n]
            removed = []
            for round_num in to_remove:
                d = self._round_dir(round_num)
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d)
                logger.debug("Removed checkpoint: round=%d", round_num)

            if removed:
                logger.info(
                    "Cleaned up %d old checkpoint(s), kept last %d",
                    len(removed), keep_last_n,
                )
            return removed

    # ── Internal helpers ─────────────────────────────────────────────

    def _list_rounds(self) -> List[int]:
        """Return sorted list of round numbers with valid checkpoint dirs."""
        rounds = []
        for entry in self._dir.iterdir():
            if entry.is_dir() and entry.name.startswith("round_"):
                try:
                    num = int(entry.name.split("_", 1)[1])
                    rounds.append(num)
                except (ValueError, IndexError):
                    continue
        rounds.sort()
        return rounds

    def _load_round(self, round_num: int) -> Optional[Tuple[int, List[NDArray], Dict[str, Any]]]:
        """Load a specific round checkpoint. Returns None on corruption."""
        d = self._round_dir(round_num)
        meta_path = d / "metadata.json"
        npz_path = d / "parameters.npz"

        if not meta_path.exists() or not npz_path.exists():
            logger.warning("Incomplete checkpoint: round=%d", round_num)
            return None

        try:
            metadata = json.loads(meta_path.read_text())
            data = np.load(npz_path)
            parameters = [data[k] for k in sorted(data.files, key=_npz_sort_key)]
            data.close()

            logger.info(
                "Loaded checkpoint: round=%d  arrays=%d  timestamp=%s",
                round_num, len(parameters),
                metadata.get("timestamp_iso", "unknown"),
            )
            return round_num, parameters, metadata.get("metrics", {})

        except Exception as exc:
            logger.warning(
                "Corrupt checkpoint round=%d: %s", round_num, exc,
            )
            return None


def _npz_sort_key(name: str) -> int:
    """Sort npz keys like 'arr_0', 'arr_1', ... numerically."""
    try:
        return int(name.split("_", 1)[1])
    except (ValueError, IndexError):
        return 0


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy/non-serializable types so json.dumps doesn't choke."""
    if obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


# ── Strategy wrapper ────────────────────────────────────────────────────


class _CheckpointWrapper(Strategy):
    """Thin wrapper that auto-saves a checkpoint after each aggregate_fit.

    Intercepts ``aggregate_fit``, delegates to the inner strategy,
    then saves the aggregated parameters and metrics via the
    CheckpointManager.
    """

    def __init__(self, strategy: Strategy, checkpoint_mgr: CheckpointManager) -> None:
        super().__init__()
        self._inner = strategy
        self._mgr = checkpoint_mgr

    def __repr__(self) -> str:
        return f"_CheckpointWrapper({self._inner!r})"

    # ── Auto-checkpoint after aggregate_fit ──────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        params, metrics = self._inner.aggregate_fit(
            server_round, results, failures,
        )

        if params is not None:
            try:
                ndarrays = parameters_to_ndarrays(params)
                self._mgr.save_round(
                    round_num=server_round,
                    parameters=ndarrays,
                    metrics=dict(metrics) if metrics else {},
                )
                self._mgr.cleanup(keep_last_n=3)
            except Exception as exc:
                # Checkpoint failure must not kill training
                logger.error(
                    "Checkpoint save failed (round %d): %s",
                    server_round, exc,
                )

        return params, metrics

    # ── Delegate everything else ─────────────────────────────────────

    def initialize_parameters(self, client_manager):
        return self._inner.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self._inner.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self._inner.configure_evaluate(server_round, parameters, client_manager)

    def evaluate(self, server_round, parameters):
        return self._inner.evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        return self._inner.aggregate_evaluate(server_round, results, failures)


def make_checkpoint_strategy(
    strategy: Strategy,
    checkpoint_mgr: CheckpointManager,
) -> Strategy:
    """Wrap any Flower Strategy to auto-checkpoint after each round.

    After every successful ``aggregate_fit``, the aggregated
    parameters and metrics are saved to disk via *checkpoint_mgr*.
    Old checkpoints are automatically cleaned up (keeping the last 3).

    Checkpoint failures are logged but never propagate — training
    continues even if the filesystem is temporarily unavailable.

    Args:
        strategy: Any Flower Strategy instance.
        checkpoint_mgr: A configured CheckpointManager.

    Returns:
        Wrapped strategy that auto-saves after each round.

    Example::

        from sfl.utils.checkpoint import CheckpointManager, make_checkpoint_strategy

        mgr = CheckpointManager("/scratch/checkpoints")
        strategy = make_checkpoint_strategy(FedAvg(...), mgr)
    """
    logger.info(
        "Checkpoint strategy wrapper enabled: dir=%s",
        checkpoint_mgr.checkpoint_dir,
    )
    return _CheckpointWrapper(strategy, checkpoint_mgr)
