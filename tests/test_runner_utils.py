"""Tests for sfl.privacy.runner_utils — shared CLI args, mod building, env validation."""

import argparse
import os
from unittest.mock import patch

import pytest

from sfl.privacy.runner_utils import add_privacy_args, build_privacy_mods, validate_env_vars


def _make_parser(**overrides):
    """Create a parser with privacy args and parse with given overrides."""
    parser = argparse.ArgumentParser()
    add_privacy_args(parser)
    argv = []
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    return parser.parse_args(argv)


class TestAddPrivacyArgs:
    """add_privacy_args() populates all expected CLI flags."""

    def test_defaults(self):
        args = _make_parser()
        assert args.dp is False
        assert args.dp_noise == 1.0
        assert args.dp_clip == 10.0
        assert args.dp_mode == "server"
        assert args.secagg is False
        assert args.aggregation == "fedavg"
        assert args.dpsgd is False

    def test_dp_flags(self):
        args = _make_parser(dp=True, dp_noise=0.5, dp_clip=5.0, dp_mode="client")
        assert args.dp is True
        assert args.dp_noise == 0.5
        assert args.dp_clip == 5.0
        assert args.dp_mode == "client"

    def test_filter_flags(self):
        args = _make_parser(percentile_privacy=10, svt_privacy=True)
        assert args.percentile_privacy == 10
        assert args.svt_privacy is True

    def test_compression_flags(self):
        args = _make_parser(compress=0.1, compress_topk=True, compress_error_feedback=True)
        assert args.compress == 0.1
        assert args.compress_topk is True
        assert args.compress_error_feedback is True

    def test_aggregation_flags(self):
        args = _make_parser(aggregation="krum", krum_byzantine=2)
        assert args.aggregation == "krum"
        assert args.krum_byzantine == 2

    def test_freeze_and_per_layer_clip(self):
        args = _make_parser(freeze_layers="4,5,6", per_layer_clip=2.0)
        assert args.freeze_layers == "4,5,6"
        assert args.per_layer_clip == 2.0


class TestBuildPrivacyMods:
    """build_privacy_mods() returns correct mods and sets env vars."""

    def setup_method(self):
        # Clear any SFL_ env vars before each test
        for key in list(os.environ):
            if key.startswith("SFL_"):
                del os.environ[key]

    def teardown_method(self):
        for key in list(os.environ):
            if key.startswith("SFL_"):
                del os.environ[key]

    def test_no_flags_returns_empty(self):
        args = _make_parser()
        mods = build_privacy_mods(args)
        assert isinstance(mods, list)
        # fedavg is default, so SFL_AGGREGATION should be set
        assert os.environ.get("SFL_AGGREGATION") == "fedavg"

    def test_dp_sets_env_vars(self):
        args = _make_parser(dp=True, dp_noise=0.5, dp_clip=5.0)
        build_privacy_mods(args)
        assert os.environ["SFL_DP_ENABLED"] == "true"
        assert os.environ["SFL_DP_NOISE"] == "0.5"
        assert os.environ["SFL_DP_CLIP"] == "5.0"

    def test_dp_client_mode_adds_clipping_mod(self):
        args = _make_parser(dp=True, dp_mode="client")
        mods = build_privacy_mods(args)
        assert len(mods) >= 1  # at least fixedclipping_mod

    def test_percentile_adds_mod(self):
        args = _make_parser(percentile_privacy=10)
        mods = build_privacy_mods(args)
        assert len(mods) == 1
        assert callable(mods[0])

    def test_svt_adds_mod(self):
        args = _make_parser(svt_privacy=True)
        mods = build_privacy_mods(args)
        assert len(mods) == 1

    def test_compression_adds_mod(self):
        args = _make_parser(compress=0.1)
        mods = build_privacy_mods(args)
        assert len(mods) == 1

    def test_dpsgd_sets_env_vars(self):
        args = _make_parser(dpsgd=True, dpsgd_clip=2.0, dpsgd_noise=0.8)
        build_privacy_mods(args)
        assert os.environ["SFL_DPSGD_ENABLED"] == "true"
        assert os.environ["SFL_DPSGD_CLIP"] == "2.0"
        assert os.environ["SFL_DPSGD_NOISE"] == "0.8"

    def test_multiple_mods_compose(self):
        args = _make_parser(
            percentile_privacy=10,
            svt_privacy=True,
            compress=0.1,
        )
        mods = build_privacy_mods(args)
        assert len(mods) == 3

    def test_secagg_is_last_mod(self):
        args = _make_parser(percentile_privacy=10, secagg=True)
        mods = build_privacy_mods(args)
        assert len(mods) == 2
        # SecAgg should be the last mod
        assert mods[-1].__name__ == "secaggplus_mod"


class TestValidateEnvVars:
    """validate_env_vars() catches misconfigurations."""

    def setup_method(self):
        for key in list(os.environ):
            if key.startswith("SFL_"):
                del os.environ[key]

    def teardown_method(self):
        for key in list(os.environ):
            if key.startswith("SFL_"):
                del os.environ[key]

    def test_no_vars_passes(self):
        validate_env_vars()  # should not raise

    def test_valid_vars_pass(self):
        os.environ["SFL_DP_ENABLED"] = "true"
        os.environ["SFL_DP_NOISE"] = "1.0"
        os.environ["SFL_DP_CLIP"] = "10.0"
        os.environ["SFL_DP_MODE"] = "server"
        validate_env_vars()  # should not raise

    def test_invalid_float_raises(self):
        os.environ["SFL_DP_NOISE"] = "abc"
        with pytest.raises(ValueError, match="SFL_DP_NOISE.*not a valid float"):
            validate_env_vars()

    def test_invalid_bool_raises(self):
        os.environ["SFL_DP_ENABLED"] = "yes"
        with pytest.raises(ValueError, match="SFL_DP_ENABLED.*not a valid boolean"):
            validate_env_vars()

    def test_invalid_choice_raises(self):
        os.environ["SFL_DP_MODE"] = "hybrid"
        with pytest.raises(ValueError, match="SFL_DP_MODE.*not valid"):
            validate_env_vars()

    def test_negative_clip_raises(self):
        os.environ["SFL_DP_CLIP"] = "-1.0"
        with pytest.raises(ValueError, match="SFL_DP_CLIP.*strictly positive"):
            validate_env_vars()

    def test_multiple_errors_reported(self):
        os.environ["SFL_DP_NOISE"] = "abc"
        os.environ["SFL_DP_CLIP"] = "xyz"
        with pytest.raises(ValueError, match="SFL_DP_NOISE") as exc_info:
            validate_env_vars()
        assert "SFL_DP_CLIP" in str(exc_info.value)

    def test_build_then_validate_roundtrip(self):
        """build_privacy_mods() sets valid env vars that validate cleanly."""
        args = _make_parser(
            dp=True, dp_noise=0.5, dp_clip=5.0,
            aggregation="krum", krum_byzantine=2,
        )
        build_privacy_mods(args)
        validate_env_vars()  # should not raise
