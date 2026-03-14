"""Tests for privacy accountant (PLD and PRV backends)."""

import pytest

try:
    from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism
    _has_dp_accounting = True
except ImportError:
    _has_dp_accounting = False

try:
    from prv_accountant import PRVAccountant as _PRVAccountant
    _has_prv_accountant = True
except ImportError:
    _has_prv_accountant = False

pytestmark = pytest.mark.skipif(
    not _has_dp_accounting, reason="dp-accounting not installed"
)

from sfl.privacy.accountant import PrivacyAccountant, BudgetExhaustedError, compose_epsilon


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


class TestComposeEpsilon:
    """Tests for sequential composition of client+server DP."""

    def test_basic_composition(self):
        """PLD composition should be tighter than basic sequential (ε₁+ε₂)."""
        total_eps, total_delta = compose_epsilon(
            eps_server=2.0, eps_client=3.0,
            delta_server=1e-5, delta_client=1e-5,
        )
        # PLD composition gives tighter bound than basic (2+3=5)
        assert total_eps < 5.0
        assert total_eps > 0
        assert total_delta == pytest.approx(2e-5)

    def test_zero_client_epsilon(self):
        """No client DP → total equals server DP."""
        total_eps, total_delta = compose_epsilon(
            eps_server=4.0, eps_client=0.0,
            delta_server=1e-5, delta_client=1e-5,
        )
        assert total_eps == pytest.approx(4.0)


class TestParticipationTracking:
    """Tests for per-round participation tracking in step()."""

    def test_fewer_participants_lower_epsilon(self):
        """Fewer actual participants → lower ε (tighter subsampling)."""
        acc_default = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=0.5, delta=1e-5,
            enforce_budget=False, num_total=10,
        )
        acc_fewer = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=0.5, delta=1e-5,
            enforce_budget=False, num_total=10,
        )
        for _ in range(5):
            acc_default.step()  # uses default 5/10 = 0.5
            acc_fewer.step(num_participants=2)  # actual 2/10 = 0.2

        assert acc_fewer.epsilon < acc_default.epsilon

    def test_default_participants_unchanged(self):
        """Passing num_participants matching default gives same result."""
        acc_a = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=0.5, delta=1e-5,
            enforce_budget=False, num_total=10,
        )
        acc_b = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=0.5, delta=1e-5,
            enforce_budget=False, num_total=10,
        )
        for _ in range(5):
            acc_a.step()
            acc_b.step(num_participants=5)  # 5/10 = 0.5, matches default
        assert acc_a.epsilon == pytest.approx(acc_b.epsilon, rel=1e-4)

    def test_none_participants_uses_default(self):
        """step(num_participants=None) should behave like step()."""
        acc_a = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=1.0, delta=1e-5,
            enforce_budget=False,
        )
        acc_b = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=1.0, delta=1e-5,
            enforce_budget=False,
        )
        for _ in range(3):
            acc_a.step()
            acc_b.step(num_participants=None)
        assert acc_a.epsilon == pytest.approx(acc_b.epsilon)


class TestPLDComposition:
    """Tests for PLD-based joint composition (B1)."""

    def test_pld_tighter_than_basic(self):
        """PLD joint composition should beat naive ε₁+ε₂ for non-trivial ε."""
        total_eps, total_delta = compose_epsilon(
            eps_server=3.0, eps_client=4.0,
            delta_server=1e-5, delta_client=1e-5,
        )
        assert total_eps < 7.0, "PLD composition should be tighter than basic (3+4=7)"
        assert total_eps > 0

    def test_pld_composition_symmetry(self):
        """compose_epsilon(a,b) ~ compose_epsilon(b,a)."""
        eps_ab, _ = compose_epsilon(
            eps_server=1.5, eps_client=2.5,
            delta_server=1e-5, delta_client=1e-5,
        )
        eps_ba, _ = compose_epsilon(
            eps_server=2.5, eps_client=1.5,
            delta_server=1e-5, delta_client=1e-5,
        )
        assert eps_ab == pytest.approx(eps_ba, rel=1e-2)

    def test_pld_delta_additive(self):
        """Total delta should equal delta_server + delta_client."""
        _, total_delta = compose_epsilon(
            eps_server=2.0, eps_client=3.0,
            delta_server=1e-5, delta_client=2e-5,
        )
        assert total_delta == pytest.approx(3e-5)


@pytest.mark.skipif(not _has_prv_accountant, reason="prv-accountant not installed")
class TestPRVAccountant:
    """Tests for the Microsoft PRV accountant backend."""

    def test_prv_epsilon_increases_with_rounds(self):
        """PRV epsilon must increase monotonically."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
            backend="prv",
        )
        epsilons = [acc.step() for _ in range(5)]
        for i in range(1, len(epsilons)):
            assert epsilons[i] > epsilons[i - 1]

    def test_prv_higher_noise_lower_epsilon(self):
        """More noise → lower epsilon for the PRV backend."""
        acc_low = PrivacyAccountant(
            noise_multiplier=0.5, delta=1e-5, enforce_budget=False,
            backend="prv",
        )
        acc_high = PrivacyAccountant(
            noise_multiplier=2.0, delta=1e-5, enforce_budget=False,
            backend="prv",
        )
        for _ in range(5):
            acc_low.step()
            acc_high.step()
        assert acc_high.epsilon < acc_low.epsilon

    def test_prv_matches_pld(self):
        """PRV and PLD should give similar epsilon values."""
        acc_pld = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
            backend="pld",
        )
        acc_prv = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
            backend="prv",
        )
        for _ in range(10):
            acc_pld.step()
            acc_prv.step()
        # PRV returns estimate; should be close to PLD (within 1.0)
        assert abs(acc_prv.epsilon - acc_pld.epsilon) < 1.0

    def test_prv_epsilon_bounds(self):
        """PRV backend should expose error bounds."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, enforce_budget=False,
            backend="prv",
        )
        for _ in range(5):
            acc.step()
        bounds = acc.epsilon_bounds
        assert bounds is not None
        low, est, high = bounds
        assert low <= est <= high
        assert low > 0

    def test_prv_budget_exhausted(self):
        """PRV backend should respect budget enforcement."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, max_epsilon=5.0,
            enforce_budget=True, backend="prv",
        )
        with pytest.raises(BudgetExhaustedError):
            for _ in range(20):
                acc.step()

    def test_prv_compute_epsilon_for_rounds(self):
        """PRV prediction should not advance state."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, backend="prv",
        )
        predicted = acc.compute_epsilon_for_rounds(10)
        assert predicted > 0
        assert acc.rounds == 0

    def test_prv_subsampling_lowers_epsilon(self):
        """PRV with subsampling should give lower ε."""
        acc_full = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=1.0, delta=1e-5,
            enforce_budget=False, backend="prv",
        )
        acc_half = PrivacyAccountant(
            noise_multiplier=1.0, sample_rate=0.5, delta=1e-5,
            enforce_budget=False, backend="prv",
        )
        for _ in range(10):
            acc_full.step()
            acc_half.step()
        assert acc_half.epsilon < acc_full.epsilon

    def test_prv_backend_property(self):
        """Backend property should return 'prv'."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, backend="prv",
        )
        assert acc.backend == "prv"

    def test_pld_backend_property(self):
        """Backend property should return 'pld' for default."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, backend="pld",
        )
        assert acc.backend == "pld"

    def test_pld_no_epsilon_bounds(self):
        """PLD backend should return None for epsilon_bounds."""
        acc = PrivacyAccountant(
            noise_multiplier=1.0, delta=1e-5, backend="pld",
        )
        acc.step()
        assert acc.epsilon_bounds is None
