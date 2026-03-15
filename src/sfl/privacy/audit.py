"""
Privacy auditing utilities for federated learning.

Provides empirical verification that DP noise effectively prevents
gradient reconstruction attacks. Based on practices from Carlini et al.
(2023) and Tramer et al. (2022).

The core idea: insert a known "canary" gradient, share the update
through the privacy pipeline, then measure how much the canary is
detectable in the output. If the canary is undetectable, the DP
parameters are working as intended.

Usage:
    from sfl.privacy.audit import PrivacyAuditor

    auditor = PrivacyAuditor(noise_scale=1.0, clipping_norm=10.0)
    result = auditor.run_canary_audit(
        params=[np.zeros(100, dtype=np.float32)],
        num_trials=100,
    )
    print(result)  # AuditResult with detection_rate, etc.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from flwr.client.typing import ClientAppCallable
from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import recorddict_compat as compat
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message, Metadata

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AuditResult:
    """Result of a canary-based privacy audit.

    Attributes:
        detection_rate: Fraction of trials where the canary was
            detectable (cosine similarity > threshold) after DP.
            Should be near 0 for good privacy.
        mean_cosine_sim: Mean cosine similarity between the canary
            direction and the noised output across trials.
        max_cosine_sim: Worst-case cosine similarity.
        noise_scale: The DP noise scale used.
        clipping_norm: The clipping norm used.
        passed: True if detection_rate <= acceptable_rate.
    """
    detection_rate: float
    mean_cosine_sim: float
    max_cosine_sim: float
    noise_scale: float
    clipping_norm: float
    passed: bool

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"AuditResult({status}: detection={self.detection_rate:.1%}, "
            f"mean_cos={self.mean_cosine_sim:.4f}, "
            f"max_cos={self.max_cosine_sim:.4f}, "
            f"σ={self.noise_scale}, C={self.clipping_norm})"
        )


class PrivacyAuditor:
    """Empirically tests whether DP noise prevents gradient reconstruction.

    Inserts a known canary gradient, applies clipping + Gaussian noise
    (simulating the server-side DP pipeline), then checks if the canary
    direction is still detectable via cosine similarity.

    Args:
        noise_scale: Ratio σ of Gaussian noise std to clipping norm.
        clipping_norm: L2 clipping norm for updates.
        detection_threshold: Cosine similarity above which the canary
            is considered "detected". Default 0.1 (very conservative).
        acceptable_rate: Maximum detection rate for a passing audit.
    """

    def __init__(
        self,
        noise_scale: float = 1.0,
        clipping_norm: float = 10.0,
        detection_threshold: float = 0.1,
        acceptable_rate: float = 0.05,
    ):
        self.noise_scale = noise_scale
        self.clipping_norm = clipping_norm
        self.detection_threshold = detection_threshold
        self.acceptable_rate = acceptable_rate

    def run_canary_audit(
        self,
        params: List[np.ndarray],
        num_trials: int = 200,
        canary_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> AuditResult:
        """Run a canary-based privacy audit.

        For each trial:
        1. Generate a random canary direction
        2. Add canary to the base params (simulating a gradient update)
        3. Clip the combined update to ``clipping_norm``
        4. Add Gaussian noise with std = ``noise_scale * clipping_norm``
        5. Measure cosine similarity between output and canary direction

        Args:
            params: Base parameter arrays (e.g., current model weights).
            num_trials: Number of independent canary trials.
            canary_scale: Magnitude of the canary gradient.
            seed: Random seed for reproducibility.

        Returns:
            AuditResult with detection statistics.
        """
        rng = np.random.RandomState(seed)
        flat_base = np.concatenate([p.ravel() for p in params]).astype(np.float64)
        d = flat_base.size

        cos_sims = []
        for _ in range(num_trials):
            # Random canary direction
            canary = rng.randn(d)
            canary = canary / (np.linalg.norm(canary) + 1e-12) * canary_scale

            # Simulate update = base + canary
            update = flat_base + canary

            # Clip
            norm = np.linalg.norm(update)
            if norm > self.clipping_norm:
                update = update * (self.clipping_norm / norm)

            # Add DP noise
            noise_std = self.noise_scale * self.clipping_norm
            update = update + rng.normal(0, noise_std, size=d)

            # Cosine similarity with canary direction
            cos = np.dot(update, canary) / (
                np.linalg.norm(update) * np.linalg.norm(canary) + 1e-12
            )
            cos_sims.append(float(cos))

        cos_sims = np.array(cos_sims)
        detection_rate = float(np.mean(np.abs(cos_sims) > self.detection_threshold))
        mean_cos = float(np.mean(np.abs(cos_sims)))
        max_cos = float(np.max(np.abs(cos_sims)))
        passed = detection_rate <= self.acceptable_rate

        result = AuditResult(
            detection_rate=detection_rate,
            mean_cosine_sim=mean_cos,
            max_cosine_sim=max_cos,
            noise_scale=self.noise_scale,
            clipping_norm=self.clipping_norm,
            passed=passed,
        )

        if passed:
            logger.info("Privacy audit PASSED: %s", result)
        else:
            logger.warning("Privacy audit FAILED: %s", result)

        return result

    def run_pipeline_audit(
        self,
        params: List[np.ndarray],
        mods: List[Callable[[Message, Context, ClientAppCallable], Message]],
        num_trials: int = 200,
        canary_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> AuditResult:
        """Run a canary audit through the real Flower client mod chain.

        Unlike ``run_canary_audit`` which simulates a simplified
        clip→noise pipeline, this method sends canary-injected
        updates through the actual mod chain (percentile, SVT,
        compression, per-layer clip, etc.). This validates that
        the *real* pipeline masks canaries, not just an idealized
        simulation.

        For each trial:
        1. Generate a random canary direction
        2. Add canary to the base params
        3. Build a Flower FitRes Message with these params
        4. Pass through the full mod chain
        5. Extract output params and measure cosine similarity

        Args:
            params: Base parameter arrays (shapes define the model).
            mods: List of Flower client mods to chain, in order.
                Each mod has signature (Message, Context, Callable) → Message.
            num_trials: Number of independent canary trials.
            canary_scale: Magnitude of the canary gradient.
            seed: Random seed for reproducibility.

        Returns:
            AuditResult with detection statistics.
        """
        rng = np.random.RandomState(seed)

        # Pre-compute shapes for splitting flat vectors back to param arrays
        shapes = [p.shape for p in params]
        sizes = [p.size for p in params]
        flat_base = np.concatenate([p.ravel() for p in params]).astype(np.float64)
        d = flat_base.size

        cos_sims = []
        for _ in range(num_trials):
            # Random canary direction
            canary = rng.randn(d)
            canary = canary / (np.linalg.norm(canary) + 1e-12) * canary_scale

            # Inject canary into params
            injected_flat = flat_base + canary
            injected_params = []
            offset = 0
            for shape, size in zip(shapes, sizes):
                injected_params.append(
                    injected_flat[offset:offset + size]
                    .reshape(shape).astype(np.float32)
                )
                offset += size

            # Build a Flower FitRes message
            fit_res = FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(injected_params),
                num_examples=10,
                metrics={},
            )
            content = compat.fitres_to_recorddict(fit_res, keep_input=True)
            metadata = Metadata(
                run_id=0,
                message_id="audit-reply",
                src_node_id=1,
                dst_node_id=0,
                reply_to_message_id="audit-msg",
                group_id="",
                created_at=0.0,
                ttl=0.0,
                message_type=MessageType.TRAIN,
            )
            out_msg = Message(metadata=metadata, content=content)

            # Build a minimal input message
            in_metadata = Metadata(
                run_id=0,
                message_id="audit-msg",
                src_node_id=0,
                dst_node_id=1,
                reply_to_message_id="",
                group_id="",
                created_at=0.0,
                ttl=0.0,
                message_type=MessageType.TRAIN,
            )
            in_msg = Message(metadata=in_metadata, content=content)

            # Chain mods: each mod wraps call_next
            # The innermost call_next returns out_msg (the canary-injected update)
            def _make_call_next(msg_to_return):
                def _call_next(m, c):
                    return msg_to_return
                return _call_next

            # Mock context
            from unittest.mock import MagicMock
            ctx = MagicMock(spec=Context)

            # Apply mods in reverse order to build the chain
            call_next = _make_call_next(out_msg)
            for mod in reversed(mods):
                _captured_mod = mod
                _captured_next = call_next
                def _chained(m, c, _mod=_captured_mod, _next=_captured_next):
                    return _mod(m, c, _next)
                call_next = _chained

            # Run the chain
            result_msg = call_next(in_msg, ctx)

            # Extract output params
            result_fit_res = compat.recorddict_to_fitres(
                result_msg.content, keep_input=True,
            )
            result_params = parameters_to_ndarrays(result_fit_res.parameters)
            result_flat = np.concatenate(
                [p.ravel().astype(np.float64) for p in result_params]
            )

            # Cosine similarity with canary direction
            result_norm = np.linalg.norm(result_flat)
            canary_norm = np.linalg.norm(canary)
            if result_norm > 0 and canary_norm > 0:
                # If mod chain changed the dimensionality (e.g., partial freeze),
                # we can only compare up to the shorter length.
                min_len = min(len(result_flat), len(canary))
                cos = float(np.dot(result_flat[:min_len], canary[:min_len]) / (
                    np.linalg.norm(result_flat[:min_len]) *
                    np.linalg.norm(canary[:min_len]) + 1e-12
                ))
            else:
                cos = 0.0
            cos_sims.append(cos)

        cos_sims = np.array(cos_sims)
        detection_rate = float(np.mean(np.abs(cos_sims) > self.detection_threshold))
        mean_cos = float(np.mean(np.abs(cos_sims)))
        max_cos = float(np.max(np.abs(cos_sims)))
        passed = detection_rate <= self.acceptable_rate

        result = AuditResult(
            detection_rate=detection_rate,
            mean_cosine_sim=mean_cos,
            max_cosine_sim=max_cos,
            noise_scale=self.noise_scale,
            clipping_norm=self.clipping_norm,
            passed=passed,
        )

        if passed:
            logger.info("Pipeline audit PASSED: %s", result)
        else:
            logger.warning("Pipeline audit FAILED: %s", result)

        return result
