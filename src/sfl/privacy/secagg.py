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

    def __post_init__(self):
        import math
        min_threshold = math.ceil(2 * self.num_shares / 3)
        if self.reconstruction_threshold < min_threshold:
            from sfl.utils.logging import get_logger
            get_logger(__name__).warning(
                "SecAgg+ reconstruction_threshold=%d is below the "
                "recommended minimum ceil(2*num_shares/3)=%d. With a "
                "low threshold, fewer colluding nodes can reconstruct "
                "individual updates. Consider threshold >= %d.",
                self.reconstruction_threshold, min_threshold, min_threshold,
            )
        if self.reconstruction_threshold > self.num_shares:
            raise ValueError(
                f"reconstruction_threshold ({self.reconstruction_threshold}) "
                f"must be <= num_shares ({self.num_shares})"
            )
        if self.num_shares < 2:
            raise ValueError(
                f"num_shares ({self.num_shares}) must be >= 2"
            )


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


def make_secagg_main(server_fn, secagg_cfg: SecAggConfig):
    """Create a ServerApp main function that wraps server_fn with SecAgg+.

    Flower's SecAgg+ requires the workflow execution path
    (DefaultWorkflow + SecAggPlusWorkflow) rather than the server_fn
    compatibility path. This helper bridges the two: it calls server_fn
    to build the strategy/config, then runs them through the SecAgg+
    workflow.

    Args:
        server_fn: The existing server_fn that returns ServerAppComponents.
        secagg_cfg: SecAgg+ configuration.

    Returns:
        A main function suitable for ServerApp.main() decorator.
    """
    secagg_kwargs = build_secagg_config(secagg_cfg)

    def secagg_main(grid, context):
        from flwr.server.client_manager import SimpleClientManager
        from flwr.server.compat import LegacyContext
        from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

        # Reuse server_fn for strategy/config building
        components = server_fn(context)

        legacy_ctx = LegacyContext(
            context=context,
            config=components.config,
            strategy=components.strategy,
            client_manager=components.client_manager or SimpleClientManager(),
        )

        workflow = DefaultWorkflow(
            fit_workflow=SecAggPlusWorkflow(**secagg_kwargs),
        )
        workflow(grid, legacy_ctx)

    return secagg_main
