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
from typing import Dict, List, Optional, Tuple, Union, Literal

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
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
        quantile_noise_multiplier: Noise multiplier for private quantile
            tracking. 0 = non-private (testing only).
    """
    noise_multiplier: float = 1.0
    clipping_norm: float = 10.0
    num_sampled_clients: int = 2
    mode: Literal["server", "client"] = "server"
    target_delta: float = 1e-5
    max_epsilon: float = 10.0
    num_total_clients: int = 2
    adaptive_clipping: bool = False
    target_quantile: float = 0.5
    clip_learning_rate: float = 0.2
    quantile_noise_multiplier: float = 0.0


def wrap_strategy_with_dp(
    strategy: Strategy,
    dp_config: DPConfig,
) -> Strategy:
    """Wrap a Flower strategy with differential privacy.

    Creates a DP-wrapped strategy with automatic per-round privacy
    accounting. When ``max_epsilon`` is reached, ``aggregate_fit``
    returns ``(None, {})`` to signal Flower to stop training, enforcing
    the formal (ε,δ)-DP guarantee.

    The privacy accountant is accessible via
    ``strategy.privacy_accountant``.

    Args:
        strategy: Base strategy (e.g., FedAvg, SumFedAvg).
        dp_config: DP configuration.

    Returns:
        DP-wrapped strategy with automatic budget enforcement.
    """
    if dp_config.noise_multiplier < 0.3:
        logger.warning(
            "noise_multiplier=%.2f is very low — this will produce ε >> 10 "
            "per round, offering negligible privacy. Consider σ >= 0.8 for "
            "meaningful privacy guarantees.",
            dp_config.noise_multiplier,
        )
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
    accountant = None
    try:
        from sfl.privacy.accountant import PrivacyAccountant
        sample_rate = dp_config.num_sampled_clients / dp_config.num_total_clients
        accountant = PrivacyAccountant(
            noise_multiplier=dp_config.noise_multiplier,
            sample_rate=sample_rate,
            delta=dp_config.target_delta,
            max_epsilon=dp_config.max_epsilon,
            num_total=dp_config.num_total_clients,
        )
    except ImportError:
        logger.warning(
            "dp-accounting not installed — no per-round epsilon tracking. "
            "Install with: pip install dp-accounting"
        )

    # Wrap with adaptive clipping if requested
    if dp_config.adaptive_clipping:
        from sfl.privacy.adaptive_clip import AdaptiveClipWrapper, AdaptiveClipConfig
        ac_cfg = AdaptiveClipConfig(
            target_quantile=dp_config.target_quantile,
            learning_rate=dp_config.clip_learning_rate,
            quantile_noise_multiplier=dp_config.quantile_noise_multiplier,
        )
        wrapped = AdaptiveClipWrapper(wrapped, ac_cfg)
        logger.info(
            f"Adaptive clipping enabled: target_quantile={ac_cfg.target_quantile}, "
            f"lr={ac_cfg.learning_rate}"
        )

    # Wrap with accounting + budget enforcement
    if accountant is not None:
        wrapped = _AccountingWrapper(wrapped, accountant)

    return wrapped


class _AccountingWrapper(Strategy):
    """Thin wrapper that auto-steps the privacy accountant after each round.

    Intercepts ``aggregate_fit`` to:
    1. Check budget *before* aggregating — if already exhausted, return
       ``(None, {})`` to stop training.
    2. Call the inner strategy's ``aggregate_fit``.
    3. Call ``accountant.step()`` — if budget is now exhausted, the *next*
       round will be stopped (this round's result is still returned so
       the model is saved correctly).
    """

    def __init__(self, strategy: Strategy, accountant) -> None:
        super().__init__()
        self._inner = strategy
        self.privacy_accountant = accountant

    def __repr__(self) -> str:
        return f"_AccountingWrapper({self._inner!r})"

    # ── Pre-check + post-step in aggregate_fit ───────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.privacy_accountant.budget_exhausted:
            logger.error(
                "Round %d SKIPPED: privacy budget already exhausted "
                "(ε=%.4f >= %.1f)",
                server_round,
                self.privacy_accountant.epsilon,
                self.privacy_accountant._max_epsilon,
            )
            return None, {}

        params, metrics = self._inner.aggregate_fit(
            server_round, results, failures,
        )

        if params is not None:
            from sfl.privacy.accountant import BudgetExhaustedError

            # Compose adaptive clipping quantile query cost if present.
            # The AdaptiveClipWrapper stores the DpEvent from each
            # round's private quantile estimate.
            from sfl.privacy.adaptive_clip import AdaptiveClipWrapper
            if isinstance(self._inner, AdaptiveClipWrapper):
                _adaptive = self._inner._last_quantile_dp_event
                if _adaptive is not None:
                    self.privacy_accountant._accountant.compose(_adaptive)

            try:
                eps = self.privacy_accountant.step(
                    num_participants=len(results),
                )
                metrics["dp_epsilon"] = eps
            except BudgetExhaustedError:
                # This round succeeded, but next round will be blocked.
                # Return this round's result so the model can be saved.
                metrics["dp_epsilon"] = self.privacy_accountant.epsilon
                metrics["dp_budget_exhausted"] = True

            # Compose with client-side DP-SGD epsilon if present
            client_epsilons = [
                res.metrics.get("dpsgd_epsilon", 0.0)
                for _, res in results
                if res.metrics
            ]
            max_client_eps = max(client_epsilons) if client_epsilons else 0.0
            if max_client_eps > 0:
                from sfl.privacy.accountant import compose_epsilon
                total_eps, total_delta = compose_epsilon(
                    eps_server=metrics.get("dp_epsilon", 0.0),
                    eps_client=max_client_eps,
                    delta_server=self.privacy_accountant._delta,
                    delta_client=self.privacy_accountant._delta,
                )
                metrics["dp_total_epsilon"] = total_eps
                metrics["dp_total_delta"] = total_delta
                metrics["dpsgd_epsilon_max"] = max_client_eps

            # Per-round privacy budget dashboard
            acc = self.privacy_accountant
            budget_pct = min(100.0, 100.0 * acc.epsilon / acc._max_epsilon)
            total_eps_display = metrics.get("dp_total_epsilon", metrics.get("dp_epsilon", 0.0))
            logger.info(
                "┌─ Privacy Budget ─────────────────────────────────────┐\n"
                "│ Round %-4d  ε = %-8.4f  δ = %-10.1e           │\n"
                "│ Budget:    %.1f%% used  (%.4f / %.1f)              │\n"
                "│ Remaining: ε = %-8.4f  (%d rounds completed)       │\n"
                "%s"
                "└──────────────────────────────────────────────────────┘",
                server_round,
                total_eps_display,
                acc._delta,
                budget_pct,
                acc.epsilon,
                acc._max_epsilon,
                max(0.0, acc._max_epsilon - acc.epsilon),
                acc.rounds,
                (
                    f"│ Client DP-SGD: ε_max = {max_client_eps:.4f}"
                    f"                       │\n"
                    if max_client_eps > 0 else ""
                ),
            )

        return params, metrics

    # ── Delegate everything else ─────────────────────────────────────────

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
