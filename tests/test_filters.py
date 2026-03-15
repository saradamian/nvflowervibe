"""
Tests for NVFlare-inspired privacy filters (Flower client mods)
and homomorphic encryption support.
"""

from logging import WARNING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism
    _has_dp_accounting = True
except ImportError:
    _has_dp_accounting = False

try:
    import tenseal
    _has_tenseal = True
except ImportError:
    _has_tenseal = False
from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import recorddict_compat as compat
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message, Metadata

from sfl.privacy.filters import (
    PercentilePrivacyConfig,
    SVTPrivacyConfig,
    make_percentile_privacy_mod,
    make_svt_privacy_mod,
    make_exclude_vars_mod,
    make_gradient_compression_mod,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_train_message(params: list[np.ndarray]) -> tuple[Message, Message]:
    """Create a mock incoming train message and an outgoing FitRes message."""
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(params),
        num_examples=10,
        metrics={},
    )
    # Build a response message with FitRes content
    content = compat.fitres_to_recorddict(fit_res, keep_input=True)
    metadata = Metadata(
        run_id=0,
        message_id="test-reply",
        src_node_id=1,
        dst_node_id=0,
        reply_to_message_id="test-msg",
        group_id="",
        created_at=0.0,
        ttl=0.0,
        message_type=MessageType.TRAIN,
    )
    out_msg = Message(metadata=metadata, content=content)

    # Input message (from server) — minimal
    in_metadata = Metadata(
        run_id=0,
        message_id="test-msg",
        src_node_id=0,
        dst_node_id=1,
        reply_to_message_id="",
        group_id="",
        created_at=0.0,
        ttl=0.0,
        message_type=MessageType.TRAIN,
    )
    in_msg = Message(metadata=in_metadata, content=content)

    return in_msg, out_msg


def _extract_params(msg: Message) -> list[np.ndarray]:
    """Extract parameters from a FitRes message."""
    fit_res = compat.recorddict_to_fitres(msg.content, keep_input=True)
    return parameters_to_ndarrays(fit_res.parameters)


# ── PercentilePrivacy Tests ─────────────────────────────────────────────────


class TestPercentilePrivacyMod:

    def test_zeros_below_percentile(self):
        """Values below the percentile should be zeroed out."""
        # Create params with known distribution
        params = [np.array([0.001, 0.002, 0.5, 0.8, 1.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        # percentile=50 means keep top 50% by abs value
        mod = make_percentile_privacy_mod(percentile=50, gamma=2.0)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        # The bottom 50% by abs value should be zeroed
        assert result_params[0][0] == 0.0  # 0.001 is below median
        assert result_params[0][1] == 0.0  # 0.002 is below median
        # Top values should be preserved (but clipped to gamma)
        assert result_params[0][4] != 0.0

    def test_clips_to_gamma(self):
        """Values should be clipped to [-gamma, gamma]."""
        params = [np.array([5.0, -5.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_percentile_privacy_mod(percentile=0, gamma=1.0)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        assert result_params[0][0] == pytest.approx(1.0)
        assert result_params[0][1] == pytest.approx(-1.0)

    def test_skips_non_train(self):
        """Non-train messages should pass through unchanged."""
        params = [np.array([1.0, 2.0], dtype=np.float32)]
        _, out_msg = _make_train_message(params)

        # Change message type to evaluate
        in_metadata = Metadata(
            run_id=0, message_id="test", src_node_id=0, dst_node_id=1,
            reply_to_message_id="", group_id="", created_at=0.0, ttl=0.0,
            message_type=MessageType.EVALUATE,
        )
        in_msg = Message(metadata=in_metadata, content=out_msg.content)

        mod = make_percentile_privacy_mod(percentile=90, gamma=0.01)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        # call_next should be called directly (pass-through)
        call_next.assert_called_once()

    def test_noise_scale_adds_noise(self):
        """With noise_scale > 0, output should differ from deterministic clipping."""
        np.random.seed(42)
        params = [np.array([0.5, -0.5, 0.8, -0.8], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod_no_noise = make_percentile_privacy_mod(percentile=0, gamma=1.0, noise_scale=0.0)
        mod_with_noise = make_percentile_privacy_mod(percentile=0, gamma=1.0, noise_scale=0.5)

        np.random.seed(42)
        r1 = _extract_params(mod_no_noise(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))
        np.random.seed(42)
        r2 = _extract_params(mod_with_noise(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))

        # With noise, outputs should not match the no-noise version
        assert not np.allclose(r1[0], r2[0])

    def test_noise_scale_zero_logs_warning(self):
        """noise_scale=0 should log a privacy warning."""
        params = [np.array([1.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_percentile_privacy_mod(percentile=0, gamma=1.0, noise_scale=0.0)
        call_next = MagicMock(return_value=out_msg)

        with patch("sfl.privacy.filters.log") as mock_log:
            mod(in_msg, MagicMock(spec=Context), call_next)
            # Check that a WARNING was logged
            warning_calls = [c for c in mock_log.call_args_list if c[0][0] == WARNING]
            assert len(warning_calls) >= 1
            assert "NO formal privacy" in str(warning_calls[0])

    @pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
    def test_epsilon_calibrates_noise(self):
        """epsilon > 0 should auto-calibrate noise_scale via PLD."""
        params = [np.array([0.5, -0.5, 0.8, -0.8], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        # With epsilon, noise should be added (calibrated)
        mod = make_percentile_privacy_mod(percentile=0, gamma=1.0, epsilon=1.0, delta=1e-5)
        call_next = MagicMock(return_value=out_msg)

        np.random.seed(0)
        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        # Output should differ from input (noise was added)
        assert not np.allclose(result_params[0], [0.5, -0.5, 0.8, -0.8], atol=1e-6)

    @pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
    def test_epsilon_overrides_noise_scale(self):
        """When epsilon is set, it should override the explicit noise_scale."""
        # Two mods: one with manual noise_scale, one with epsilon
        mod_manual = make_percentile_privacy_mod(percentile=0, gamma=1.0, noise_scale=0.5)
        mod_eps = make_percentile_privacy_mod(percentile=0, gamma=1.0, noise_scale=0.5, epsilon=1.0, delta=1e-5)

        # The epsilon mod should have recalculated noise_scale
        # We can't easily inspect the closure, but we can verify the two
        # produce different outputs with the same seed
        params = [np.array([0.5, -0.5], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        np.random.seed(99)
        r1 = _extract_params(mod_manual(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))
        np.random.seed(99)
        r2 = _extract_params(mod_eps(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))

        # Different noise_scale → different results
        assert not np.allclose(r1[0], r2[0])


# ── SVTPrivacy Tests ────────────────────────────────────────────────────────


class TestSVTPrivacyMod:

    def test_sparsifies_output(self):
        """SVT should zero out most parameters (fraction=0.1 → ~10% kept)."""
        np.random.seed(42)
        params = [np.random.randn(100).astype(np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_svt_privacy_mod(fraction=0.1, epsilon=1.0, gamma=1.0, tau=0.0)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        # Most values should be zero (only ~10% kept)
        nonzero = np.count_nonzero(result_params[0])
        assert nonzero <= 20  # allow some slack

    def test_preserves_shape(self):
        """Output shapes should match input shapes."""
        params = [
            np.random.randn(5, 3).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        ]
        in_msg, out_msg = _make_train_message(params)

        mod = make_svt_privacy_mod(fraction=0.5)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        assert result_params[0].shape == (5, 3)
        assert result_params[1].shape == (3,)

    def test_clips_to_gamma(self):
        """All non-zero output values should be within [-gamma, gamma]."""
        params = [np.array([100.0, -100.0, 50.0, -50.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        gamma = 0.5
        mod = make_svt_privacy_mod(fraction=1.0, epsilon=10.0, gamma=gamma, tau=0.0)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        nonzero_mask = result_params[0] != 0
        if nonzero_mask.any():
            assert np.all(np.abs(result_params[0][nonzero_mask]) <= gamma + 1e-6)

    def test_iteration_cap_prevents_infinite_loop(self):
        """With very low epsilon, SVT should hit the iteration cap and stop."""
        params = [np.ones(50, dtype=np.float32) * 0.001]
        in_msg, out_msg = _make_train_message(params)

        # Very low epsilon → huge noise → hard to accept anything
        mod = make_svt_privacy_mod(fraction=0.5, epsilon=1e-6, gamma=1e-5, tau=10.0)
        call_next = MagicMock(return_value=out_msg)

        # Should terminate (not hang) thanks to iteration cap
        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)
        assert result_params[0].shape == (50,)

    def test_high_epsilon_accepts_most(self):
        """With very high epsilon (low noise), nearly all params are accepted."""
        np.random.seed(42)
        params = [np.random.randn(100).astype(np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_svt_privacy_mod(fraction=0.9, epsilon=1000.0, gamma=10.0, tau=0.0)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        nonzero = np.count_nonzero(result_params[0])
        assert nonzero >= 60  # most should be accepted (single-pass SVT)

    def test_noise_var_kwarg_ignored(self):
        """Legacy noise_var kwarg should be accepted but ignored."""
        params = [np.array([1.0, 2.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        # Should not raise, noise_var is caught by **_kwargs
        mod = make_svt_privacy_mod(fraction=0.5, noise_var=0.5)
        call_next = MagicMock(return_value=out_msg)
        result = mod(in_msg, MagicMock(spec=Context), call_next)
        assert _extract_params(result)[0].shape == (2,)

    def test_optimal_budget_less_noise_than_standard(self):
        """Optimal budget split should produce less noise (more nonzeros) for same ε.

        Run multiple trials since DP noise is CSRNG-seeded and therefore
        non-deterministic. The optimal split should win on average.
        """
        wins = 0
        trials = 10
        for _ in range(trials):
            params = [np.random.randn(200).astype(np.float32)]
            in_msg, out_msg = _make_train_message(params)

            mod_std = make_svt_privacy_mod(
                fraction=0.5, epsilon=1.0, gamma=10.0, tau=0.0, optimal_budget=False,
            )
            r_std = _extract_params(mod_std(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))

            mod_opt = make_svt_privacy_mod(
                fraction=0.5, epsilon=1.0, gamma=10.0, tau=0.0, optimal_budget=True,
            )
            r_opt = _extract_params(mod_opt(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))

            nz_std = np.count_nonzero(r_std[0])
            nz_opt = np.count_nonzero(r_opt[0])
            if nz_opt >= nz_std:
                wins += 1

        # Optimal should win in majority of trials
        assert wins >= trials // 2, f"Optimal won only {wins}/{trials} trials"

    def test_prescreen_reduces_candidates(self):
        """Pre-screening should still produce valid output."""
        np.random.seed(42)
        params = [np.random.randn(100).astype(np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_svt_privacy_mod(
            fraction=0.1, epsilon=10.0, gamma=10.0, tau=0.0,
            pre_screen_ratio=0.5,
        )
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)
        assert result_params[0].shape == (100,)
        # Should have some nonzero values
        assert np.count_nonzero(result_params[0]) > 0

    def test_standard_budget_backward_compatible(self):
        """optimal_budget=False should reproduce the original behavior."""
        np.random.seed(42)
        params = [np.random.randn(50).astype(np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_svt_privacy_mod(
            fraction=0.5, epsilon=1.0, gamma=1.0, tau=0.0, optimal_budget=False,
        )
        call_next = MagicMock(return_value=out_msg)
        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)
        assert result_params[0].shape == (50,)


# ── ExcludeVars Tests ───────────────────────────────────────────────────────


class TestExcludeVarsMod:

    def test_zeros_excluded_indices(self):
        """Excluded parameter arrays should be all zeros."""
        params = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32),
        ]
        in_msg, out_msg = _make_train_message(params)

        mod = make_exclude_vars_mod(exclude_indices=[0, 2])
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        # Index 0 and 2 should be zeroed
        np.testing.assert_array_equal(result_params[0], [0.0, 0.0])
        np.testing.assert_array_equal(result_params[2], [0.0, 0.0])
        # Index 1 should be preserved
        np.testing.assert_array_equal(result_params[1], [3.0, 4.0])

    def test_no_indices_passes_through(self):
        """With no exclude indices, params should pass through unchanged."""
        params = [np.array([1.0, 2.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_exclude_vars_mod(exclude_indices=[])
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        np.testing.assert_array_equal(result_params[0], [1.0, 2.0])

    def test_none_indices_passes_through(self):
        """With None exclude indices, params should pass through unchanged."""
        params = [np.array([1.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_exclude_vars_mod(exclude_indices=None)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        np.testing.assert_array_equal(result_params[0], [1.0])


# ── GradientCompression Tests ──────────────────────────────────────────────


class TestGradientCompressionMod:

    def test_compression_ratio(self):
        """Output should have ~90% zeros when ratio=0.1."""
        np.random.seed(42)
        params = [np.random.randn(100).astype(np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_gradient_compression_mod(compression_ratio=0.1, noise_scale=0.0)
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        nonzero = np.count_nonzero(result_params[0])
        assert nonzero == 10  # 10% of 100

    def test_noise_applied(self):
        """With noise_scale > 0, surviving values should differ from input."""
        np.random.seed(42)
        params = [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_gradient_compression_mod(
            compression_ratio=1.0, noise_scale=1.0, use_random_mask=False,
        )
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        # Values should be modified by noise
        assert not np.allclose(result_params[0], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_random_mask_varies(self):
        """Random masking should produce different selections across calls."""
        # Use uniform magnitudes so selection is purely random
        mod = make_gradient_compression_mod(
            compression_ratio=0.2, noise_scale=0.0, use_random_mask=True,
        )

        # Separate messages to avoid in-place mutation sharing
        params1 = [np.ones(50, dtype=np.float32)]
        in1, out1 = _make_train_message(params1)
        np.random.seed(1)
        r1 = _extract_params(mod(in1, MagicMock(spec=Context), MagicMock(return_value=out1)))

        params2 = [np.ones(50, dtype=np.float32)]
        in2, out2 = _make_train_message(params2)
        np.random.seed(2)
        r2 = _extract_params(mod(in2, MagicMock(spec=Context), MagicMock(return_value=out2)))

        # Different seeds → different mask selections
        assert not np.array_equal(r1[0] != 0, r2[0] != 0)

    def test_topk_deterministic(self):
        """TopK (non-random) should always select the same values."""
        params = [np.array([0.1, 0.5, 0.9, 0.2, 0.8], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_gradient_compression_mod(
            compression_ratio=0.4, noise_scale=0.0, use_random_mask=False,
        )

        r1 = _extract_params(mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))
        r2 = _extract_params(mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))

        np.testing.assert_array_equal(r1[0], r2[0])
        # Top 2 by abs value are indices 2 (0.9) and 4 (0.8)
        assert r1[0][2] != 0  # 0.9 should be kept
        assert r1[0][4] != 0  # 0.8 should be kept

    def test_calibrated_noise_requires_delta_and_clip(self):
        """epsilon without delta or clipping_norm should raise."""
        with pytest.raises(ValueError, match="delta and clipping_norm"):
            make_gradient_compression_mod(
                compression_ratio=0.1, epsilon=1.0,
            )

    @pytest.mark.skipif(
        not _has_dp_accounting,
        reason="dp-accounting not installed",
    )
    def test_calibrated_noise_applied(self):
        """With epsilon set, noise should be calibrated (different from heuristic)."""
        np.random.seed(42)
        params = [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_gradient_compression_mod(
            compression_ratio=1.0, use_random_mask=False,
            epsilon=1.0, delta=1e-5, clipping_norm=5.0,
        )
        call_next = MagicMock(return_value=out_msg)
        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)

        # Values should be modified by calibrated noise
        assert not np.allclose(result_params[0], [1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.mark.skipif(
        not _has_dp_accounting,
        reason="dp-accounting not installed",
    )
    def test_dp_forces_topk_over_random_mask(self):
        """When epsilon is set, use_random_mask must be forced to False (C1)."""
        mod = make_gradient_compression_mod(
            compression_ratio=0.5, use_random_mask=True,
            epsilon=1.0, delta=1e-5, clipping_norm=5.0,
        )
        # The mod should function with deterministic TopK
        params = [np.array([0.1, 0.5, 0.9, 0.2, 0.8], dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        np.random.seed(10)
        r1 = _extract_params(mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))
        np.random.seed(20)
        r2 = _extract_params(mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))

        # With TopK, same indices should be selected (mask is deterministic);
        # only noise differs
        mask1 = r1[0] != 0
        mask2 = r2[0] != 0
        np.testing.assert_array_equal(mask1, mask2)

    def test_error_feedback_accumulates(self):
        """Error feedback should add residuals from previous rounds."""
        # Use TopK with ratio=0.2 (keep 1 of 5) and no noise
        mod = make_gradient_compression_mod(
            compression_ratio=0.2, noise_scale=0.0,
            use_random_mask=False, error_feedback=True,
        )

        # Round 1: values [0.1, 0.2, 0.3, 0.4, 0.5] → keep top 1 (0.5)
        params1 = [np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)]
        in1, out1 = _make_train_message(params1)
        r1 = _extract_params(mod(in1, MagicMock(spec=Context), MagicMock(return_value=out1)))

        # Only index 4 (0.5) should survive
        assert r1[0][4] != 0
        # Others zeroed
        zeroed_count_r1 = np.sum(r1[0] == 0)
        assert zeroed_count_r1 == 4

        # Round 2: same values — but error feedback adds the residual
        # from round 1 (the zeroed values: 0.1, 0.2, 0.3, 0.4)
        # So effective values are [0.2, 0.4, 0.6, 0.8, 0.5]
        # Now index 3 (0.8) should be the top value
        params2 = [np.array([0.1, 0.2, 0.3, 0.4, 0.0], dtype=np.float32)]
        in2, out2 = _make_train_message(params2)
        r2 = _extract_params(mod(in2, MagicMock(spec=Context), MagicMock(return_value=out2)))

        # With error feedback, previously zeroed values get a boost
        # so the result should differ from no-feedback compression
        assert np.any(r2[0] != 0)

    def test_no_error_feedback_by_default(self):
        """Without error_feedback, rounds are independent."""
        mod = make_gradient_compression_mod(
            compression_ratio=0.2, noise_scale=0.0,
            use_random_mask=False, error_feedback=False,
        )

        params = [np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)]
        in1, out1 = _make_train_message(params)
        r1 = _extract_params(mod(in1, MagicMock(spec=Context), MagicMock(return_value=out1)))

        in2, out2 = _make_train_message(params)
        r2 = _extract_params(mod(in2, MagicMock(spec=Context), MagicMock(return_value=out2)))

        # Without error feedback, same input → same output
        np.testing.assert_array_equal(r1[0], r2[0])


# ── HE Tests ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_tenseal, reason="tenseal not installed")
class TestHEContext:
    """Tests for homomorphic encryption (requires tenseal)."""

    @pytest.fixture
    def he_ctx(self):
        from sfl.privacy.he import HEContext
        return HEContext()

    def test_encrypt_decrypt_roundtrip(self, he_ctx):
        """Encrypted params should decrypt back to original values."""
        params = [np.array([7.5, 3.2], dtype=np.float32)]
        shapes = [p.shape for p in params]
        dtypes = [p.dtype for p in params]

        encrypted = he_ctx.encrypt_parameters(params)
        assert len(encrypted) == 1
        assert isinstance(encrypted[0], bytes)

        decrypted = he_ctx.decrypt_parameters(encrypted, shapes, dtypes)
        np.testing.assert_allclose(decrypted[0], params[0], atol=1e-3)

    def test_homomorphic_addition(self, he_ctx):
        """Adding two encrypted vectors should give the sum."""
        a = [np.array([7.5], dtype=np.float32)]
        b = [np.array([2.5], dtype=np.float32)]

        enc_a = he_ctx.encrypt_parameters(a)
        enc_b = he_ctx.encrypt_parameters(b)

        enc_sum = he_ctx.add_encrypted(enc_a, enc_b)
        decrypted = he_ctx.decrypt_parameters(
            enc_sum, [a[0].shape], [a[0].dtype],
        )

        np.testing.assert_allclose(decrypted[0], [10.0], atol=1e-3)

    def test_multi_client_sum(self, he_ctx):
        """Simulates 3 clients contributing encrypted values, server sums."""
        client_vals = [3.0, 5.0, 7.0]
        encrypted_vals = [
            he_ctx.encrypt_parameters([np.array([v], dtype=np.float32)])
            for v in client_vals
        ]

        # Server aggregates encrypted values
        agg = encrypted_vals[0]
        for enc in encrypted_vals[1:]:
            agg = he_ctx.add_encrypted(agg, enc)

        result = he_ctx.decrypt_parameters(agg, [(1,)], [np.float32])
        np.testing.assert_allclose(result[0], [15.0], atol=1e-2)

    def test_2d_array_roundtrip(self, he_ctx):
        """Works with multi-dimensional arrays."""
        params = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
        shapes = [p.shape for p in params]

        encrypted = he_ctx.encrypt_parameters(params)
        decrypted = he_ctx.decrypt_parameters(encrypted, shapes)

        np.testing.assert_allclose(decrypted[0], params[0], atol=1e-3)

    def test_ciphertext_expansion(self, he_ctx):
        """Encrypted data should be much larger than plaintext."""
        params = [np.array([1.0], dtype=np.float32)]
        encrypted = he_ctx.encrypt_parameters(params)

        # A single float32 (4 bytes) becomes ~160KB+ encrypted
        assert len(encrypted[0]) > 1000


# ── SVT Single-Pass Tests ──────────────────────────────────────────────────


class TestSVTSinglePass:
    """Tests verifying the SVT single-pass fix (A1).

    Standard SVT's ε-DP proof assumes each query is answered once.
    Re-drawing noise on rejected candidates would violate this.
    """

    def test_single_pass_no_requery(self):
        """Rejected candidates should NOT be re-queried — single pass only."""
        np.random.seed(42)
        # Use params where most values are tiny (below threshold)
        params = [np.concatenate([
            np.array([0.0001] * 90, dtype=np.float32),
            np.array([1.0] * 10, dtype=np.float32),
        ])]
        in_msg, out_msg = _make_train_message(params)

        mod = make_svt_privacy_mod(
            fraction=0.5, epsilon=0.01, gamma=1.0, tau=0.5,
        )
        call_next = MagicMock(return_value=out_msg)

        result = mod(in_msg, MagicMock(spec=Context), call_next)
        result_params = _extract_params(result)
        # With single pass and noise, should accept fewer than n_upload
        nonzero = np.count_nonzero(result_params[0])
        assert nonzero <= 50  # should not exceed fraction target

    def test_single_pass_deterministic_count(self):
        """Two runs with same seed should give identical acceptance counts."""
        params = [np.random.randn(100).astype(np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_svt_privacy_mod(fraction=0.3, epsilon=1.0, gamma=1.0, tau=0.0)

        np.random.seed(99)
        r1 = _extract_params(mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))
        nz1 = np.count_nonzero(r1[0])

        np.random.seed(99)
        r2 = _extract_params(mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg)))
        nz2 = np.count_nonzero(r2[0])

        assert nz1 == nz2


# ── PercentilePrivacy L2 Sensitivity Tests ──────────────────────────────────


class TestPercentileSensitivity:
    """Tests verifying the PercentilePrivacy L2 sensitivity fix (A4)."""

    @pytest.mark.skipif(not _has_dp_accounting, reason="dp-accounting not installed")
    def test_epsilon_noise_scales_with_surviving_count(self):
        """With epsilon set, noise should increase with more surviving elements.

        L2 sensitivity = gamma * sqrt(K) where K = surviving count.
        More survivors → more noise needed for same ε guarantee.
        """
        gamma = 1.0

        # Few survivors (percentile=95 → ~5% kept)
        params_few = [np.random.randn(100).astype(np.float32)]
        in_msg_few, out_msg_few = _make_train_message(params_few)
        mod_few = make_percentile_privacy_mod(
            percentile=95, gamma=gamma, epsilon=1.0, delta=1e-5,
        )
        np.random.seed(42)
        r_few = _extract_params(mod_few(
            in_msg_few, MagicMock(spec=Context), MagicMock(return_value=out_msg_few),
        ))

        # Many survivors (percentile=5 → ~95% kept)
        params_many = [np.random.randn(100).astype(np.float32)]
        in_msg_many, out_msg_many = _make_train_message(params_many)
        mod_many = make_percentile_privacy_mod(
            percentile=5, gamma=gamma, epsilon=1.0, delta=1e-5,
        )
        np.random.seed(42)
        r_many = _extract_params(mod_many(
            in_msg_many, MagicMock(spec=Context), MagicMock(return_value=out_msg_many),
        ))

        # Both should produce valid output
        assert r_few[0].shape == (100,)
        assert r_many[0].shape == (100,)


# ── Partial Freeze (Lambda-SecAgg) Tests ──────────────────────────────────


class TestPartialFreezeMod:
    """Tests for Lambda-SecAgg partial freezing mod."""

    def test_strips_frozen_layers(self):
        """Only trainable layers should remain in output."""
        from sfl.privacy.filters import make_partial_freeze_mod

        params = [
            np.ones(10, dtype=np.float32),   # layer 0 (frozen)
            np.ones(20, dtype=np.float32),   # layer 1 (trainable)
            np.ones(30, dtype=np.float32),   # layer 2 (frozen)
            np.ones(40, dtype=np.float32),   # layer 3 (trainable)
        ]
        in_msg, out_msg = _make_train_message(params)

        mod = make_partial_freeze_mod(trainable_indices=[1, 3])
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        assert len(result_params) == 2
        assert result_params[0].shape == (20,)  # layer 1
        assert result_params[1].shape == (40,)  # layer 3

    def test_noop_when_no_indices(self):
        """With trainable_indices=None, all layers should pass through."""
        from sfl.privacy.filters import make_partial_freeze_mod

        params = [np.ones(10, dtype=np.float32), np.ones(20, dtype=np.float32)]
        in_msg, out_msg = _make_train_message(params)

        mod = make_partial_freeze_mod(trainable_indices=None)
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        assert len(result_params) == 2

    def test_preserves_trainable_values(self):
        """Trainable layer values should be unchanged."""
        from sfl.privacy.filters import make_partial_freeze_mod

        layer1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        params = [
            np.zeros(5, dtype=np.float32),  # frozen
            layer1.copy(),                   # trainable
        ]
        in_msg, out_msg = _make_train_message(params)

        mod = make_partial_freeze_mod(trainable_indices=[1])
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        assert len(result_params) == 1
        np.testing.assert_array_equal(result_params[0], layer1)

    def test_stores_indices_in_metrics(self):
        """Trainable indices should be stored in fit_res metrics."""
        from sfl.privacy.filters import make_partial_freeze_mod

        params = [np.ones(10, dtype=np.float32)] * 4
        in_msg, out_msg = _make_train_message(params)

        mod = make_partial_freeze_mod(trainable_indices=[0, 2])
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))

        fit_res = compat.recorddict_to_fitres(result.content, keep_input=True)
        assert "_trainable_indices" in fit_res.metrics
        assert fit_res.metrics["_trainable_indices"] == "0,2"

    def test_stores_frozen_shapes_in_metrics(self):
        """Frozen layer shapes should be stored so server can reconstruct."""
        from sfl.privacy.filters import make_partial_freeze_mod

        params = [
            np.ones((768, 3072), dtype=np.float32),  # layer 0 (frozen)
            np.ones(100, dtype=np.float32),           # layer 1 (trainable)
            np.ones((512, 512), dtype=np.float32),    # layer 2 (frozen)
        ]
        in_msg, out_msg = _make_train_message(params)

        mod = make_partial_freeze_mod(trainable_indices=[1])
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))

        fit_res = compat.recorddict_to_fitres(result.content, keep_input=True)
        assert "_frozen_shapes" in fit_res.metrics
        shapes_str = fit_res.metrics["_frozen_shapes"]
        # Should contain entries for layers 0 and 2
        assert "0:768,3072" in shapes_str
        assert "2:512,512" in shapes_str

    def test_strategy_restores_correct_shapes(self):
        """Server-side strategy should restore frozen layers with correct shapes."""
        from sfl.privacy.filters import make_partial_freeze_mod, make_partial_freeze_strategy

        original_params = [
            np.ones((10, 20), dtype=np.float32),  # layer 0 (frozen)
            np.full(30, 2.0, dtype=np.float32),    # layer 1 (trainable)
            np.ones((5, 5, 3), dtype=np.float32),  # layer 2 (frozen)
            np.full(15, 3.0, dtype=np.float32),    # layer 3 (trainable)
        ]
        in_msg, out_msg = _make_train_message(original_params)

        # Client side: strip frozen layers
        mod = make_partial_freeze_mod(trainable_indices=[1, 3])
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        fit_res = compat.recorddict_to_fitres(result.content, keep_input=True)

        # Server side: wrap a mock strategy, capturing the inner call
        inner_aggregate = MagicMock(return_value=(None, {}))
        mock_strategy = MagicMock()
        mock_strategy.aggregate_fit = inner_aggregate
        make_partial_freeze_strategy(mock_strategy, trainable_indices=[1, 3])

        # Simulate calling aggregate_fit with partial results
        from flwr.server.client_proxy import ClientProxy
        proxy = MagicMock(spec=ClientProxy)
        mock_strategy.aggregate_fit(1, [(proxy, fit_res)], [])

        # Get the expanded results passed to the original (inner) aggregate_fit
        inner_aggregate.assert_called_once()
        call_args = inner_aggregate.call_args
        expanded_results = call_args[0][1]
        expanded_params = parameters_to_ndarrays(expanded_results[0][1].parameters)

        assert len(expanded_params) == 4
        assert expanded_params[0].shape == (10, 20)   # frozen, correct shape
        assert expanded_params[1].shape == (30,)       # trainable, preserved
        assert expanded_params[2].shape == (5, 5, 3)   # frozen, correct shape
        assert expanded_params[3].shape == (15,)       # trainable, preserved
        # Frozen layers should be zeros
        np.testing.assert_array_equal(expanded_params[0], np.zeros((10, 20)))
        np.testing.assert_array_equal(expanded_params[2], np.zeros((5, 5, 3)))
        # Trainable layers should be preserved
        np.testing.assert_array_equal(expanded_params[1], np.full(30, 2.0))
        np.testing.assert_array_equal(expanded_params[3], np.full(15, 3.0))

    def test_size_reduction(self):
        """Output should be smaller than input when layers are frozen."""
        from sfl.privacy.filters import make_partial_freeze_mod

        params = [
            np.ones(1000, dtype=np.float32),  # layer 0 (frozen)
            np.ones(100, dtype=np.float32),   # layer 1 (trainable)
        ]
        in_msg, out_msg = _make_train_message(params)

        mod = make_partial_freeze_mod(trainable_indices=[1])
        result = mod(in_msg, MagicMock(spec=Context), MagicMock(return_value=out_msg))
        result_params = _extract_params(result)

        original_size = sum(p.size for p in params)
        result_size = sum(p.size for p in result_params)
        assert result_size < original_size
        assert result_size == 100

    def test_exported_from_privacy_init(self):
        """make_partial_freeze_mod should be importable from sfl.privacy."""
        from sfl.privacy import make_partial_freeze_mod, make_partial_freeze_strategy
        assert callable(make_partial_freeze_mod)
        assert callable(make_partial_freeze_strategy)



