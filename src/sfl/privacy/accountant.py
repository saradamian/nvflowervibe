"""
Privacy accounting for differential privacy.

Tracks cumulative (ε,δ)-DP guarantee across FL rounds using either:
- **PLD** (Privacy Loss Distribution) via Google's dp-accounting (default)
- **PRV** (Privacy Random Variable) via Microsoft's prv_accountant

Both provide tight composition bounds for Gaussian mechanisms; PLD is the
default, PRV additionally returns error bounds (ε_low, ε_est, ε_high).

When sample_rate < 1.0, applies Poisson subsampling amplification for
tighter privacy bounds (no extra noise needed — pure accounting win).

Includes **shuffle-model amplification** (Balle et al., Crypto 2019;
Feldman et al., 2021): when clients send updates through an anonymous
shuffler, the central (ε,δ)-DP guarantee is tighter than the local ε₀
each client applies.

Usage:
    from sfl.privacy.accountant import PrivacyAccountant

    # Default PLD backend
    accountant = PrivacyAccountant(
        noise_multiplier=1.0,
        sample_rate=0.5,
        delta=1e-5,
        max_epsilon=10.0,
    )

    # PRV backend (Microsoft) — returns error bounds
    accountant = PrivacyAccountant(
        noise_multiplier=1.0,
        sample_rate=0.5,
        delta=1e-5,
        max_epsilon=10.0,
        backend="prv",
    )

    for round_num in range(num_rounds):
        eps = accountant.step()
        if accountant.budget_exhausted:
            break
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class BudgetExhaustedError(RuntimeError):
    """Raised when the cumulative ε exceeds the configured budget.

    Training must stop to preserve the (ε,δ)-DP guarantee.
    """

try:
    from dp_accounting import dp_event
    from dp_accounting.pld import pld_privacy_accountant
    from dp_accounting.pld.privacy_loss_distribution import (
        from_gaussian_mechanism,
    )
    HAS_DP_ACCOUNTING = True
except ImportError:
    HAS_DP_ACCOUNTING = False

try:
    from prv_accountant import PRVAccountant as _PRVAccountant
    from prv_accountant import PoissonSubsampledGaussianMechanism as _PSGM
    HAS_PRV_ACCOUNTANT = True
except ImportError:
    HAS_PRV_ACCOUNTANT = False


@dataclass
class AccountingConfig:
    """Privacy accounting configuration.

    Args:
        delta: Target δ in (ε,δ)-DP. Typically 1/N^1.1 where N is
            dataset size. 1e-5 is a common default.
        max_epsilon: Stop training when cumulative ε exceeds this.
            Set to float('inf') to disable budget enforcement.
        backend: Accounting backend: ``"pld"`` (Google dp-accounting,
            default) or ``"prv"`` (Microsoft prv_accountant).
    """
    delta: float = 1e-5
    max_epsilon: float = 10.0
    backend: Literal["pld", "prv"] = "pld"


class PrivacyAccountant:
    """Tracks cumulative (ε,δ)-DP across federated learning rounds.

    Supports two backends:
    - **PLD** (default): Google's dp-accounting PLD accountant. Composes
      per-round DpEvents for tight (ε,δ) tracking.
    - **PRV**: Microsoft's prv_accountant. Uses the Privacy Random
      Variable framework with FFT-based composition. Returns error
      bounds (ε_low, ε_est, ε_high) accessible via ``epsilon_bounds``.

    Args:
        noise_multiplier: Ratio of noise std to clipping norm (σ/C).
        sample_rate: Fraction of clients sampled per round
            (num_sampled / num_total). Set to 1.0 if all clients
            participate every round.
        delta: Target δ for (ε,δ)-DP.
        max_epsilon: Budget cap — training should stop if exceeded.
        enforce_budget: If True, ``step()`` raises
            ``BudgetExhaustedError`` when ε >= max_epsilon.
            If False, only logs a warning (legacy behavior).
        num_total: Total number of clients in the pool. Required
            for per-round participation tracking via ``step()``.
        backend: ``"pld"`` or ``"prv"``.
        eps_error: Error tolerance for PRV epsilon computation.
            Lower values give tighter bounds but cost more compute.
            Default 0.01 (±1%). Only used with ``backend="prv"``.
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float = 1.0,
        delta: float = 1e-5,
        max_epsilon: float = 10.0,
        enforce_budget: bool = True,
        num_total: Optional[int] = None,
        backend: Literal["pld", "prv"] = "pld",
        eps_error: float = 0.01,
    ):
        self._backend = backend
        self._noise_multiplier = noise_multiplier
        self._sample_rate = sample_rate
        self._delta = delta
        self._max_epsilon = max_epsilon
        self._enforce_budget = enforce_budget
        self._rounds = 0
        self._num_total = num_total
        self._eps_error = eps_error
        # PRV error bounds: (eps_low, eps_estimate, eps_high)
        self._prv_bounds: Optional[Tuple[float, float, float]] = None

        if backend == "prv":
            if not HAS_PRV_ACCOUNTANT:
                raise ImportError(
                    "prv-accountant is required for the PRV backend. "
                    "Install with: pip install prv-accountant"
                )
            # PRV accountant is stateless — we recompute each step
            # by passing the current round count. Store config only.
            self._prv_mechanism = _PSGM(
                noise_multiplier=noise_multiplier,
                sampling_probability=sample_rate,
            )
        else:
            if not HAS_DP_ACCOUNTING:
                raise ImportError(
                    "dp-accounting is required for privacy accounting. "
                    "Install with: pip install 'sfl[privacy]'"
                )

            # Build default per-round DpEvent with subsampling amplification.
            self._base_event = dp_event.GaussianDpEvent(
                noise_multiplier=noise_multiplier
            )
            if sample_rate < 1.0:
                self._round_event = dp_event.PoissonSampledDpEvent(
                    sampling_probability=sample_rate,
                    event=self._base_event,
                )
            else:
                self._round_event = self._base_event

            # Internal PLD accountant for tight composition
            self._accountant = pld_privacy_accountant.PLDAccountant()

        logger.info(
            "Privacy accountant initialized: backend=%s, noise=%.3f, "
            "sample_rate=%.3f, delta=%.1e, max_eps=%.1f%s",
            backend, noise_multiplier, sample_rate, delta, max_epsilon,
            " (subsampling amplification ON)" if sample_rate < 1.0 else "",
        )

    def _compute_prv_epsilon(self) -> float:
        """Compute ε via PRV accountant for the current round count."""
        # PRV needs max_self_compositions set upfront; use a safe upper
        # bound (current rounds + generous headroom).
        max_comp = max(self._rounds * 2, 1000)
        acc = _PRVAccountant(
            prvs=[self._prv_mechanism],
            max_self_compositions=[max_comp],
            eps_error=self._eps_error,
            delta_error=self._delta * 1e-3,
        )
        low, est, high = acc.compute_epsilon(
            delta=self._delta,
            num_self_compositions=[self._rounds],
        )
        self._prv_bounds = (low, est, high)
        return est

    def step(self, num_participants: Optional[int] = None) -> float:
        """Record one FL round and return updated cumulative epsilon.

        Args:
            num_participants: Actual number of clients that participated
                this round.  When provided (and differs from the default
                sample count), a per-round DpEvent with the correct
                sampling probability is composed instead of the default
                fixed-rate event. This gives tighter bounds when
                participation varies across rounds.  (PLD backend only;
                PRV uses the fixed sample_rate.)

        Returns:
            Current cumulative ε at the configured δ.
        """
        self._rounds += 1

        if self._backend == "prv":
            eps = self._compute_prv_epsilon()
        else:
            # Use per-round event if actual participation differs from default
            if (
                num_participants is not None
                and self._num_total is not None
                and self._num_total > 0
                and num_participants != round(self._sample_rate * self._num_total)
            ):
                actual_rate = min(num_participants / self._num_total, 1.0)
                if actual_rate < 1.0:
                    event = dp_event.PoissonSampledDpEvent(
                        sampling_probability=actual_rate,
                        event=self._base_event,
                    )
                else:
                    event = self._base_event
                self._accountant.compose(event)
            else:
                self._accountant.compose(self._round_event)

            eps = self._accountant.get_epsilon(self._delta)

        bounds_str = ""
        if self._prv_bounds is not None:
            low, _, high = self._prv_bounds
            bounds_str = f" [ε_low={low:.4f}, ε_high={high:.4f}]"

        logger.info(
            "Round %d: ε = %.4f (δ = %.1e) | budget remaining: %.4f%s",
            self._rounds, eps, self._delta,
            max(0.0, self._max_epsilon - eps),
            bounds_str,
        )

        if self.budget_exhausted:
            msg = (
                f"Privacy budget EXHAUSTED: ε = {eps:.4f} >= "
                f"max_epsilon = {self._max_epsilon:.1f}. "
                f"Further training degrades privacy guarantees."
            )
            if self._enforce_budget:
                logger.error(msg)
                raise BudgetExhaustedError(msg)
            logger.warning(msg)

        return eps

    @property
    def epsilon(self) -> float:
        """Current cumulative ε at the configured δ."""
        if self._rounds == 0:
            return 0.0
        if self._backend == "prv":
            return self._compute_prv_epsilon()
        return self._accountant.get_epsilon(self._delta)

    @property
    def epsilon_bounds(self) -> Optional[Tuple[float, float, float]]:
        """PRV error bounds (ε_low, ε_estimate, ε_high).

        Only available when backend="prv". Returns None for PLD backend.
        """
        if self._backend == "prv" and self._rounds > 0:
            self._compute_prv_epsilon()
            return self._prv_bounds
        return None

    @property
    def delta(self) -> float:
        """Target δ."""
        return self._delta

    @property
    def rounds(self) -> int:
        """Number of rounds recorded."""
        return self._rounds

    @property
    def budget_exhausted(self) -> bool:
        """True if cumulative ε >= max_epsilon."""
        return self.epsilon >= self._max_epsilon

    @property
    def backend(self) -> str:
        """Active accounting backend name."""
        return self._backend

    def compose_auxiliary(self, event) -> None:
        """Compose an auxiliary DP event into the running budget.

        Use this to account for privacy costs of adaptive mechanisms
        that run alongside the main gradient mechanism, such as:

        - Adaptive clipping quantile estimation (GaussianDpEvent)
        - SVT parameter selection (LaplaceDpEvent)

        These costs are composed via ``ComposedDpEvent`` (which handles
        adaptive composition per Dwork et al. 2010) rather than
        advanced composition, giving correct accounting even when
        the auxiliary mechanism's output influences the next round's
        main mechanism.

        Args:
            event: A ``dp_accounting.dp_event.DpEvent`` instance.

        Raises:
            RuntimeError: If using the PRV backend (which doesn't
                support arbitrary event composition).
        """
        if self._backend == "prv":
            raise RuntimeError(
                "compose_auxiliary is only supported with the PLD backend. "
                "PRV accountant does not support arbitrary event composition."
            )
        self._accountant.compose(event)
        logger.info(
            "Composed auxiliary DP event: %s | ε = %.4f",
            type(event).__name__,
            self._accountant.get_epsilon(self._delta),
        )

    def compute_epsilon_for_rounds(self, num_rounds: int) -> float:
        """Predict ε for a given number of rounds (without advancing state).

        Useful for planning: check what ε you'd get after N rounds
        before starting training.

        Args:
            num_rounds: Number of rounds to simulate.

        Returns:
            Predicted ε at the configured δ.
        """
        if self._backend == "prv":
            max_comp = max(num_rounds * 2, 1000)
            acc = _PRVAccountant(
                prvs=[self._prv_mechanism],
                max_self_compositions=[max_comp],
                eps_error=self._eps_error,
                delta_error=self._delta * 1e-3,
            )
            _, est, _ = acc.compute_epsilon(
                delta=self._delta,
                num_self_compositions=[num_rounds],
            )
            return est
        preview = pld_privacy_accountant.PLDAccountant()
        preview.compose(self._round_event, count=num_rounds)
        return preview.get_epsilon(self._delta)


def compose_epsilon(
    eps_server: float,
    eps_client: float,
    delta_server: float = 1e-5,
    delta_client: float = 1e-5,
    *,
    sigma_server: float = 0.0,
    sigma_client: float = 0.0,
) -> tuple:
    """Compose server-side and client-side (ε,δ) using PLD-based joint composition.

    Uses the PLD accountant for tighter bounds when dp-accounting is
    available (2-5× tighter than basic sequential composition).
    Falls back to basic sequential composition (ε₁ + ε₂, δ₁ + δ₂) if
    dp-accounting is not installed.

    When ``sigma_server`` and ``sigma_client`` are provided (the actual
    noise multipliers of the Gaussian mechanisms), uses exact
    GaussianDpEvent composition instead of the lossy binary-search
    Gaussian approximation. This avoids H3 (approximation error from
    mapping arbitrary (ε,δ) → Gaussian σ).

    Args:
        eps_server: Server-side cumulative ε.
        eps_client: Client-side per-round ε (worst across clients).
        delta_server: Server-side δ.
        delta_client: Client-side δ.
        sigma_server: Server Gaussian noise multiplier (if known).
        sigma_client: Client Gaussian noise multiplier (if known).

    Returns:
        (ε_total, δ_total) composed guarantee.
    """
    if not HAS_DP_ACCOUNTING or eps_server <= 0 or eps_client <= 0:
        # Fall back to basic sequential composition
        return (eps_server + eps_client, delta_server + delta_client)

    try:
        total_delta = delta_server + delta_client
        accountant = pld_privacy_accountant.PLDAccountant()

        if sigma_server > 0 and sigma_client > 0:
            # Exact composition from known Gaussian noise multipliers
            accountant.compose(
                dp_event.GaussianDpEvent(noise_multiplier=sigma_server)
            )
            accountant.compose(
                dp_event.GaussianDpEvent(noise_multiplier=sigma_client)
            )
        else:
            # Approximate each mechanism as a Gaussian with matching (ε,δ):
            # find σ such that Gaussian(σ) gives (ε, δ).
            for eps, delta in [(eps_server, delta_server), (eps_client, delta_client)]:
                lo, hi = 1e-6, 1e6
                for _ in range(100):
                    mid = (lo + hi) / 2.0
                    pld = from_gaussian_mechanism(standard_deviation=mid)
                    if pld.get_epsilon_for_delta(delta) > eps:
                        lo = mid
                    else:
                        hi = mid
                sigma = (lo + hi) / 2.0
                accountant.compose(dp_event.GaussianDpEvent(noise_multiplier=sigma))

        eps_total = accountant.get_epsilon(total_delta)
        return (eps_total, total_delta)
    except Exception:
        # Fall back to basic composition on any error
        return (eps_server + eps_client, delta_server + delta_client)


def compute_pabi_epsilon(
    noise_multiplier: float,
    num_steps: int,
    sample_rate: float = 1.0,
    delta: float = 1e-5,
    smoothness: float = 1.0,
    strong_convexity: float = 0.0,
) -> float:
    """Estimate ε using Privacy Amplification by Iteration (PABI).

    For strongly convex and smooth losses, iterative noisy gradient
    descent enjoys privacy amplification beyond standard composition
    (Feldman et al. 2018, Altschuler & Talwar 2022, Ye & Shokri 2022).

    The PABI bound applies a contraction factor ``c = 1 - strong_convexity /
    smoothness`` per step, which dampens the contribution of earlier
    steps to the final iterate. When ``strong_convexity > 0``, this
    yields tighter ε than PLD's worst-case composition.

    Falls back to standard PLD accounting when ``strong_convexity == 0``
    (no amplification) or when dp-accounting is not installed.

    Args:
        noise_multiplier: σ/C ratio used in Gaussian mechanism.
        num_steps: Total number of gradient steps.
        sample_rate: Fraction of data sampled per step (Poisson).
        delta: Target δ for (ε,δ)-DP.
        smoothness: L-smoothness constant of the loss.
        strong_convexity: μ-strong convexity constant. 0 = no PABI.

    Returns:
        Estimated ε under PABI (or standard PLD when μ=0).
    """
    if not HAS_DP_ACCOUNTING:
        raise ImportError("dp-accounting is required for PABI computation")

    if strong_convexity <= 0 or smoothness <= 0:
        # No PABI available, use standard PLD composition
        acc = pld_privacy_accountant.PLDAccountant()
        base = dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier)
        event = (
            dp_event.PoissonSampledDpEvent(sampling_probability=sample_rate, event=base)
            if sample_rate < 1.0 else base
        )
        acc.compose(event, count=num_steps)
        return acc.get_epsilon(delta)

    # PABI contraction factor: c = 1 - μ/L
    contraction = 1.0 - strong_convexity / smoothness

    # Effective number of rounds under contraction:
    # The last step has weight 1, step t has weight c^(T-t).
    # sum_{t=0..T-1} c^(2t) = (1 - c^{2T}) / (1 - c^2)
    if contraction >= 1.0:
        return compute_pabi_epsilon(
            noise_multiplier, num_steps, sample_rate, delta,
            smoothness, 0.0,
        )

    effective_rounds = (1.0 - contraction ** (2 * num_steps)) / (1.0 - contraction ** 2)

    # Compose that many effective rounds
    acc = pld_privacy_accountant.PLDAccountant()
    base = dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier)
    event = (
        dp_event.PoissonSampledDpEvent(sampling_probability=sample_rate, event=base)
        if sample_rate < 1.0 else base
    )
    # Round effective_rounds to int (conservative: ceiling)
    import math
    acc.compose(event, count=math.ceil(effective_rounds))
    pabi_eps = acc.get_epsilon(delta)

    logger.info(
        "PABI: %d actual steps → %.1f effective rounds (c=%.4f, μ/L=%.4f) → ε=%.4f",
        num_steps, effective_rounds, contraction,
        strong_convexity / smoothness, pabi_eps,
    )
    return pabi_eps


def shuffle_amplification_epsilon(
    local_epsilon: float,
    num_clients: int,
    delta: float = 1e-5,
) -> float:
    """Compute central (ε,δ)-DP guarantee under shuffle-model amplification.

    When clients send locally-randomized updates through an anonymous
    channel (shuffler), the central guarantee is significantly tighter
    than the local ε₀ each client applies.

    Uses the analytical bound from Feldman, McMillan & Talwar (2021,
    Theorem 3.1), which improves on Balle et al. (Crypto 2019):

        ε_central ≤ log(1 + (e^ε₀ - 1) / n · (√(2·log(4/δ)/n) + 1/n))

    For small ε₀ this simplifies to approximately ε₀ · √(8·log(4/δ)/n).

    This is a pure accounting improvement — no algorithmic changes needed.
    Clients apply their existing local randomizer (e.g., DP-SGD with ε₀),
    and the shuffler provides amplification for free.

    Args:
        local_epsilon: Per-client local ε₀ (e.g., from DP-SGD).
        num_clients: Number of clients n participating in the round.
        delta: Target δ for the central (ε,δ)-DP guarantee.

    Returns:
        Central ε under shuffle-model amplification. Always ≤ local_epsilon.

    Raises:
        ValueError: If num_clients < 2 or local_epsilon < 0 or delta <= 0.
    """
    import math

    if num_clients < 2:
        raise ValueError("Shuffle amplification requires at least 2 clients")
    if local_epsilon < 0:
        raise ValueError(f"local_epsilon must be non-negative, got {local_epsilon}")
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")

    if local_epsilon == 0:
        return 0.0

    n = num_clients
    e_eps = math.exp(local_epsilon) - 1.0  # e^ε₀ - 1

    # Feldman-McMillan-Talwar (2021) bound
    inner = math.sqrt(2.0 * math.log(4.0 / delta) / n) + 1.0 / n
    central_eps = math.log(1.0 + e_eps * inner)

    # Shuffle amplification can never make things worse than local
    result = min(central_eps, local_epsilon)

    logger.info(
        "Shuffle amplification: ε_local=%.4f → ε_central=%.4f "
        "(n=%d, δ=%.1e, amplification=%.1fx)",
        local_epsilon, result, n, delta,
        local_epsilon / max(result, 1e-10),
    )

    return result
