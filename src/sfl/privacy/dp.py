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
    """
    noise_multiplier: float = 0.1
    clipping_norm: float = 10.0
    num_sampled_clients: int = 2
    mode: Literal["server", "client"] = "server"
    target_delta: float = 1e-5
    max_epsilon: float = 10.0
    num_total_clients: int = 2


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

    return wrapped
