"""
Secure aggregation configuration for Flower.

SecAgg+ ensures the server only sees the aggregate of client
updates — it cannot inspect any individual client's contribution.
This is orthogonal to DP and can be combined with it.

How it works:
  1. Each client splits its update into secret shares
  2. Shares are exchanged between clients (not the server)
  3. Server receives only the sum of shares, which equals
     the aggregate of all updates
  4. If a client drops out, the remaining shares can still
     reconstruct the aggregate (threshold scheme)

Flower implements this via SecAgg+ workflows on the server
and secaggplus_mod on clients.
"""

from dataclasses import dataclass

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SecAggConfig:
    """Secure aggregation configuration.

    Args:
        num_shares: Number of secret shares per client.
            Must be >= 2. More shares = more fault tolerance
            but higher communication cost.
        reconstruction_threshold: Minimum shares needed to
            reconstruct the aggregate. Must be <= num_shares.
            Lower = more fault tolerant but less secure.
        clipping_range: Range for clipping update values.
            Values outside [-range, range] are clipped.
        quantization_range: Precision for quantizing float
            updates to integers (required for secret sharing).
    """
    num_shares: int = 3
    reconstruction_threshold: int = 2
    clipping_range: float = 8.0
    quantization_range: int = 4194304


def build_secagg_config(cfg: SecAggConfig) -> dict:
    """Build kwargs for SecAggPlusWorkflow from config.

    Returns:
        Dict of kwargs to pass to SecAggPlusWorkflow().
    """
    logger.info(
        f"SecAgg+ enabled: shares={cfg.num_shares}, "
        f"threshold={cfg.reconstruction_threshold}, "
        f"clip_range={cfg.clipping_range}"
    )
    return {
        "num_shares": cfg.num_shares,
        "reconstruction_threshold": cfg.reconstruction_threshold,
        "clipping_range": cfg.clipping_range,
        "quantization_range": cfg.quantization_range,
    }
