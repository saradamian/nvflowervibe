"""
Differential privacy wrappers for Flower strategies.

Wraps any Flower strategy with DP using Flower's built-in
DifferentialPrivacy wrappers. Supports both server-side and
client-side clipping modes.

Server-side DP:
  The server clips each client's update to a fixed L2 norm,
  then adds calibrated Gaussian noise to the aggregate.
  Simpler setup — no client-side mods needed.

Client-side DP:
  Each client clips its own update before sending. The server
  adds noise to the clipped aggregate. Requires the
  fixedclipping_mod on the ClientApp. Better privacy accounting
  because the server never sees unclipped updates.
"""

from dataclasses import dataclass
from typing import Literal

from flwr.server.strategy import Strategy
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    DifferentialPrivacyServerSideFixedClipping,
)

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from dp_accounting.pld.privacy_loss_distribution import (
        from_gaussian_mechanism,
    )
    _HAS_DP_ACCOUNTING = True
except ImportError:
    _HAS_DP_ACCOUNTING = False


def calibrate_gaussian_sigma(
    epsilon: float,
    delta: float,
    sensitivity: float,
) -> float:
    """Compute σ for the Gaussian mechanism via binary search on the PLD.

    Returns the smallest σ such that adding N(0, σ²) noise to a function
    with L2 sensitivity ``sensitivity`` satisfies (ε,δ)-DP.

    Based on the analytic Gaussian mechanism (Balle & Wang, 2018).

    Args:
        epsilon: Target ε (must be > 0).
        delta: Target δ (must be > 0).
        sensitivity: L2 sensitivity of the function.

    Returns:
        Noise standard deviation σ.

    Raises:
        ImportError: If dp-accounting is not installed.
        ValueError: If epsilon or delta are non-positive.
    """
    if not _HAS_DP_ACCOUNTING:
        raise ImportError(
            "dp-accounting is required for noise calibration. "
            "Install with: pip install dp-accounting"
        )
    if epsilon <= 0 or delta <= 0:
        raise ValueError(f"epsilon ({epsilon}) and delta ({delta}) must be positive")

    lo, hi = 1e-6, 1e6
    for _ in range(100):  # converges well within 100 bisection steps
        mid = (lo + hi) / 2.0
        pld = from_gaussian_mechanism(standard_deviation=mid / sensitivity)
        if pld.get_epsilon_for_delta(delta) > epsilon:
            lo = mid  # need more noise
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class DPConfig:
    """Differential privacy configuration.

    Args:
        noise_multiplier: Ratio of noise std to clipping norm.
            Higher = more private but noisier. Start with 0.1 for
            utility, use 1.0+ for strong privacy.
        clipping_norm: Max L2 norm of client updates. Updates
            exceeding this are scaled down. For ESM2 (8M params),
            10.0 is a reasonable starting point.
        num_sampled_clients: Number of clients sampled per round.
            Used in privacy accounting (the more clients, the
            more noise is divided, improving utility).
        mode: 'server' for server-side DP (simpler),
              'client' for client-side DP (stronger guarantees).
        target_delta: δ in (ε,δ)-DP for privacy accounting.
        max_epsilon: Stop training when ε exceeds this budget.
        num_total_clients: Total client pool size for subsampling
            amplification in privacy accounting.
        adaptive_clipping: Enable adaptive clipping norm (Andrew et al. 2021).
        target_quantile: Target fraction of unclipped updates (0–1).
        clip_learning_rate: Geometric step size for clip norm update.
    """
    noise_multiplier: float = 0.1
    clipping_norm: float = 10.0
    num_sampled_clients: int = 2
    mode: Literal["server", "client"] = "server"
    target_delta: float = 1e-5
    max_epsilon: float = 10.0
    num_total_clients: int = 2
    adaptive_clipping: bool = False
    target_quantile: float = 0.5
    clip_learning_rate: float = 0.2


def wrap_strategy_with_dp(
    strategy: Strategy,
    dp_config: DPConfig,
) -> Strategy:
    """Wrap a Flower strategy with differential privacy.

    Also creates a PrivacyAccountant (if dp-accounting is installed)
    and attaches it as ``strategy.privacy_accountant`` for per-round
    epsilon tracking.

    Args:
        strategy: Base strategy (e.g., FedAvg, SumFedAvg).
        dp_config: DP configuration.

    Returns:
        DP-wrapped strategy (with ``.privacy_accountant`` attribute
        if dp-accounting is available).
    """
    if dp_config.mode == "client":
        wrapped = DifferentialPrivacyClientSideFixedClipping(
            strategy=strategy,
            noise_multiplier=dp_config.noise_multiplier,
            clipping_norm=dp_config.clipping_norm,
            num_sampled_clients=dp_config.num_sampled_clients,
        )
        logger.info(
            f"DP enabled (client-side): noise={dp_config.noise_multiplier}, "
            f"clip={dp_config.clipping_norm}, sampled={dp_config.num_sampled_clients}"
        )
    else:
        wrapped = DifferentialPrivacyServerSideFixedClipping(
            strategy=strategy,
            noise_multiplier=dp_config.noise_multiplier,
            clipping_norm=dp_config.clipping_norm,
            num_sampled_clients=dp_config.num_sampled_clients,
        )
        logger.info(
            f"DP enabled (server-side): noise={dp_config.noise_multiplier}, "
            f"clip={dp_config.clipping_norm}, sampled={dp_config.num_sampled_clients}"
        )

    # Attach privacy accountant for per-round epsilon tracking
    try:
        from sfl.privacy.accountant import PrivacyAccountant
        sample_rate = dp_config.num_sampled_clients / dp_config.num_total_clients
        wrapped.privacy_accountant = PrivacyAccountant(
            noise_multiplier=dp_config.noise_multiplier,
            sample_rate=sample_rate,
            delta=dp_config.target_delta,
            max_epsilon=dp_config.max_epsilon,
        )
    except ImportError:
        logger.warning(
            "dp-accounting not installed — no per-round epsilon tracking. "
            "Install with: pip install dp-accounting"
        )
        wrapped.privacy_accountant = None

    # Wrap with adaptive clipping if requested
    if dp_config.adaptive_clipping:
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig
        ac_cfg = AdaptiveClipConfig(
            target_quantile=dp_config.target_quantile,
            learning_rate=dp_config.clip_learning_rate,
        )
        wrapped = AdaptiveClipWrapper(wrapped, ac_cfg)
        logger.info(
            f"Adaptive clipping enabled: target_quantile={ac_cfg.target_quantile}, "
            f"lr={ac_cfg.learning_rate}"
        )

    return wrapped
