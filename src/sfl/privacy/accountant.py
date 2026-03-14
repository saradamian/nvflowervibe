"""
Privacy accounting for differential privacy.

Tracks cumulative (ε,δ)-DP guarantee across FL rounds using Google's
dp-accounting library with the PLD (Privacy Loss Distribution) accountant,
which provides tighter composition bounds than RDP for Gaussian mechanisms.

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

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

try:
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
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float = 1.0,
        delta: float = 1e-5,
        max_epsilon: float = 10.0,
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
        self._rounds = 0

        # Create the per-round PLD for Gaussian mechanism.
        # standard_deviation = noise_multiplier (sensitivity is normalized to 1).
        self._round_pld = from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
        )
        self._composed_pld = None

        logger.info(
            "Privacy accountant initialized: noise=%.3f, sample_rate=%.3f, "
            "delta=%.1e, max_eps=%.1f",
            noise_multiplier, sample_rate, delta, max_epsilon,
        )

    def step(self) -> float:
        """Record one FL round and return updated cumulative epsilon.

        Should be called once after each aggregation round.

        Returns:
            Current cumulative ε at the configured δ.
        """
        self._rounds += 1
        if self._composed_pld is None:
            self._composed_pld = self._round_pld
        else:
            self._composed_pld = self._composed_pld.compose(self._round_pld)

        eps = self._composed_pld.get_epsilon_for_delta(self._delta)

        logger.info(
            "Round %d: ε = %.4f (δ = %.1e) | budget remaining: %.4f",
            self._rounds, eps, self._delta,
            max(0.0, self._max_epsilon - eps),
        )

        if self.budget_exhausted:
            logger.warning(
                "Privacy budget EXHAUSTED: ε = %.4f >= max_epsilon = %.1f. "
                "Further training degrades privacy guarantees.",
                eps, self._max_epsilon,
            )

        return eps

    @property
    def epsilon(self) -> float:
        """Current cumulative ε at the configured δ."""
        if self._composed_pld is None:
            return 0.0
        return self._composed_pld.get_epsilon_for_delta(self._delta)

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
        composed = self._round_pld.self_compose(num_rounds)
        return composed.get_epsilon_for_delta(self._delta)
