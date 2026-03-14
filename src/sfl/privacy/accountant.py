"""
Privacy accounting for differential privacy.

Tracks cumulative (ε,δ)-DP guarantee across FL rounds using Google's
dp-accounting library with the PLD (Privacy Loss Distribution) accountant,
which provides tighter composition bounds than RDP for Gaussian mechanisms.

When sample_rate < 1.0, applies Poisson subsampling amplification for
tighter privacy bounds (no extra noise needed — pure accounting win).

Usage:
    from sfl.privacy.accountant import PrivacyAccountant

    accountant = PrivacyAccountant(
        noise_multiplier=1.0,
        sample_rate=0.5,
        delta=1e-5,
        max_epsilon=10.0,
    )
    for round_num in range(num_rounds):
        # ... training round ...
        eps = accountant.step()
        if accountant.budget_exhausted:
            break
"""

from dataclasses import dataclass
from typing import Optional

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


@dataclass
class AccountingConfig:
    """Privacy accounting configuration.

    Args:
        delta: Target δ in (ε,δ)-DP. Typically 1/N^1.1 where N is
            dataset size. 1e-5 is a common default.
        max_epsilon: Stop training when cumulative ε exceeds this.
            Set to float('inf') to disable budget enforcement.
    """
    delta: float = 1e-5
    max_epsilon: float = 10.0


class PrivacyAccountant:
    """Tracks cumulative (ε,δ)-DP across federated learning rounds.

    Uses Google's dp-accounting PLD (Privacy Loss Distribution) accountant
    for tight composition of Gaussian mechanism privacy loss.

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
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float = 1.0,
        delta: float = 1e-5,
        max_epsilon: float = 10.0,
        enforce_budget: bool = True,
        num_total: Optional[int] = None,
    ):
        if not HAS_DP_ACCOUNTING:
            raise ImportError(
                "dp-accounting is required for privacy accounting. "
                "Install with: pip install 'sfl[privacy]'"
            )

        self._noise_multiplier = noise_multiplier
        self._sample_rate = sample_rate
        self._delta = delta
        self._max_epsilon = max_epsilon
        self._enforce_budget = enforce_budget
        self._rounds = 0
        self._num_total = num_total

        # Build default per-round DpEvent with subsampling amplification.
        self._base_event = dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier)
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
            "Privacy accountant initialized: noise=%.3f, sample_rate=%.3f, "
            "delta=%.1e, max_eps=%.1f%s",
            noise_multiplier, sample_rate, delta, max_epsilon,
            " (subsampling amplification ON)" if sample_rate < 1.0 else "",
        )

    def step(self, num_participants: Optional[int] = None) -> float:
        """Record one FL round and return updated cumulative epsilon.

        Args:
            num_participants: Actual number of clients that participated
                this round.  When provided (and differs from the default
                sample count), a per-round DpEvent with the correct
                sampling probability is composed instead of the default
                fixed-rate event. This gives tighter bounds when
                participation varies across rounds.

        Returns:
            Current cumulative ε at the configured δ.
        """
        self._rounds += 1

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

        logger.info(
            "Round %d: ε = %.4f (δ = %.1e) | budget remaining: %.4f",
            self._rounds, eps, self._delta,
            max(0.0, self._max_epsilon - eps),
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
        return self._accountant.get_epsilon(self._delta)

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

    def compute_epsilon_for_rounds(self, num_rounds: int) -> float:
        """Predict ε for a given number of rounds (without advancing state).

        Useful for planning: check what ε you'd get after N rounds
        before starting training.

        Args:
            num_rounds: Number of rounds to simulate.

        Returns:
            Predicted ε at the configured δ.
        """
        preview = pld_privacy_accountant.PLDAccountant()
        preview.compose(self._round_event, count=num_rounds)
        return preview.get_epsilon(self._delta)


def compose_epsilon(
    eps_server: float,
    eps_client: float,
    delta_server: float = 1e-5,
    delta_client: float = 1e-5,
) -> tuple:
    """Compose server-side and client-side (ε,δ) using PLD-based joint composition.

    Uses the PLD accountant for tighter bounds when dp-accounting is
    available (2-5× tighter than basic sequential composition).
    Falls back to basic sequential composition (ε₁ + ε₂, δ₁ + δ₂) if
    dp-accounting is not installed.

    When both server-side DP (aggregate noise) and client-side DP-SGD
    (per-example clipping+noise) are used, the total privacy guarantee
    for a single data point is bounded by their composition.

    Args:
        eps_server: Server-side cumulative ε.
        eps_client: Client-side per-round ε (worst across clients).
        delta_server: Server-side δ.
        delta_client: Client-side δ.

    Returns:
        (ε_total, δ_total) composed guarantee.
    """
    if not HAS_DP_ACCOUNTING or eps_server <= 0 or eps_client <= 0:
        # Fall back to basic sequential composition
        return (eps_server + eps_client, delta_server + delta_client)

    # Use PLD composition: convert each (ε,δ) to a PLD, then compose.
    # We model each mechanism as a generic (ε,δ)-DP mechanism by
    # constructing its privacy loss distribution from a Gaussian
    # mechanism with equivalent noise multiplier.
    try:
        total_delta = delta_server + delta_client
        accountant = pld_privacy_accountant.PLDAccountant()

        # Approximate each mechanism as a Gaussian with matching (ε,δ):
        # find σ such that Gaussian(σ) gives (ε, δ).
        for eps, delta in [(eps_server, delta_server), (eps_client, delta_client)]:
            # Binary search for noise_multiplier giving this ε at this δ
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
