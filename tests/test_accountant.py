"""Tests for privacy accountant (PLD-based DP accounting)."""

import pytest

try:
    from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism
    _has_dp_accounting = True
except ImportError:
    _has_dp_accounting = False

pytestmark = pytest.mark.skipif(
    not _has_dp_accounting, reason="dp-accounting not installed"
)

from sfl.privacy.accountant import PrivacyAccountant, BudgetExhaustedError


class TestPrivacyAccountant:

    def test_epsilon_increases_with_rounds(self):
        """Epsilon must increase monotonically as rounds accumulate."""
        acc = PrivacyAccountant(noise_multiplier=1.0, delta=1e-5, enforce_budget=False)
        epsilons = [acc.step() for _ in range(5)]
        for i in range(1, len(epsilons)):
            assert epsilons[i] > epsilons[i - 1]

    def test_higher_noise_lower_epsilon(self):
        """More noise → lower epsilon for the same number of rounds."""
        acc_low = PrivacyAccountant(noise_multiplier=0.5, delta=1e-5, enforce_budget=False)
        acc_high = PrivacyAccountant(noise_multiplier=2.0, delta=1e-5, enforce_budget=False)
        for _ in range(5):
            acc_low.step()
            acc_high.step()
        assert acc_high.epsilon < acc_low.epsilon

    def test_budget_exhausted(self):
        """Budget should be flagged exhausted when epsilon exceeds max."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, max_epsilon=5.0,
            enforce_budget=False,  # legacy mode: warn only
        )
        assert not acc.budget_exhausted
        # Keep stepping until budget exhausted (epsilon ~4.4 per round)
        for _ in range(10):
            acc.step()
            if acc.budget_exhausted:
                break
        assert acc.budget_exhausted
        assert acc.epsilon >= 5.0

    def test_enforce_budget_raises(self):
        """enforce_budget=True should raise BudgetExhaustedError."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, max_epsilon=5.0,
            enforce_budget=True,
        )
        with pytest.raises(BudgetExhaustedError):
            for _ in range(20):
                acc.step()

    def test_enforce_budget_false_no_raise(self):
        """enforce_budget=False should NOT raise, only warn."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, max_epsilon=5.0,
            enforce_budget=False,
        )
        # Should complete all rounds without raising
        for _ in range(20):
            acc.step()
        assert acc.budget_exhausted

    def test_compute_epsilon_for_rounds_does_not_advance(self):
        """Predicting epsilon for N rounds shouldn't change internal state."""
        acc = PrivacyAccountant(noise_multiplier=1.0, delta=1e-5)
        predicted = acc.compute_epsilon_for_rounds(10)
        assert predicted > 0
        assert acc.rounds == 0
        assert acc.epsilon == 0.0

    def test_initial_state(self):
        """Fresh accountant should have zero rounds and zero epsilon."""
        acc = PrivacyAccountant(noise_multiplier=1.0, delta=1e-5)
        assert acc.rounds == 0
        assert acc.epsilon == 0.0
        assert acc.delta == 1e-5
        assert not acc.budget_exhausted

    def test_subsampling_lowers_epsilon(self):
        """sample_rate < 1.0 should give strictly lower ε than 1.0."""
        acc_full = PrivacyAccountant(noise_multiplier=1.0, sample_rate=1.0, delta=1e-5, enforce_budget=False)
        acc_half = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.5, delta=1e-5, enforce_budget=False)
        for _ in range(10):
            acc_full.step()
            acc_half.step()
        assert acc_half.epsilon < acc_full.epsilon

    def test_subsampling_compute_epsilon_for_rounds(self):
        """compute_epsilon_for_rounds should also reflect subsampling."""
        acc_full = PrivacyAccountant(noise_multiplier=1.0, sample_rate=1.0, delta=1e-5)
        acc_half = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.5, delta=1e-5)
        eps_full = acc_full.compute_epsilon_for_rounds(20)
        eps_half = acc_half.compute_epsilon_for_rounds(20)
        assert eps_half < eps_full
        # State should not advance
        assert acc_full.rounds == 0
        assert acc_half.rounds == 0
