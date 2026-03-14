"""
Tests for NVFlare-inspired privacy filters (Flower client mods)
and homomorphic encryption support.
"""

from unittest.mock import MagicMock, patch
import unittest

import numpy as np
import pytest

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


# ── HE Tests ───────────────────────────────────────────────────────────────


@unittest.skipUnless(_has_tenseal, "tenseal not installed")
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



