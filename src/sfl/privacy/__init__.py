"""
Privacy module for SFL federated learning.

Provides differential privacy (DP), secure aggregation (SecAgg),
and NVFlare-inspired privacy filters as Flower client mods.

DP modes:
  - Server-side: server clips and adds noise after aggregation
  - Client-side: clients clip updates locally, server adds noise

Privacy filters (Flower mods):
  - PercentilePrivacy: only share top-percentile weight diffs
  - SVTPrivacy: sparse vector technique with ε-DP guarantees
  - ExcludeVars: zero out specific parameter layers

SecAgg:
  - SecAgg+ protocol ensures the server only sees the aggregate,
    not individual client updates
"""

from sfl.privacy.dp import wrap_strategy_with_dp, DPConfig, calibrate_gaussian_sigma
from sfl.privacy.adaptive_clip import (
    AdaptiveClipWrapper,
    AdaptiveClipConfig,
    make_per_layer_clip_mod,
)
from sfl.privacy.filters import (
    make_percentile_privacy_mod,
    make_svt_privacy_mod,
    make_exclude_vars_mod,
    make_gradient_compression_mod,
    make_partial_freeze_mod,
    make_partial_freeze_strategy,
    make_adapter_mask_mod,
    PercentilePrivacyConfig,
    SVTPrivacyConfig,
    GradientCompressionConfig,
)
from sfl.privacy.secagg import build_secagg_config, make_secagg_main, SecAggConfig

# Optional: privacy accounting (requires dp-accounting)
try:
    from sfl.privacy.accountant import (
        PrivacyAccountant,
        AccountingConfig,
        BudgetExhaustedError,
        compose_epsilon,
        shuffle_amplification_epsilon,
        HAS_PRV_ACCOUNTANT,
    )
except ImportError:
    HAS_PRV_ACCOUNTANT = False

# Privacy auditing (no extra deps)
from sfl.privacy.audit import PrivacyAuditor, AuditResult

# Runner utilities (shared arg-parsing and mod building)
from sfl.privacy.runner_utils import add_privacy_args, build_privacy_mods, validate_env_vars

# Optional: homomorphic encryption (requires tenseal)
try:
    from sfl.privacy.he import HEContext, HEConfig
except ImportError:
    pass

__all__ = [
    "wrap_strategy_with_dp",
    "DPConfig",
    "AdaptiveClipWrapper",
    "AdaptiveClipConfig",
    "make_per_layer_clip_mod",
    "calibrate_gaussian_sigma",
    "build_secagg_config",
    "SecAggConfig",
    "make_percentile_privacy_mod",
    "make_svt_privacy_mod",
    "make_exclude_vars_mod",
    "make_gradient_compression_mod",
    "PercentilePrivacyConfig",
    "SVTPrivacyConfig",
    "GradientCompressionConfig",
    "make_partial_freeze_mod",
    "make_partial_freeze_strategy",
    "make_adapter_mask_mod",
    "PrivacyAccountant",
    "AccountingConfig",
    "BudgetExhaustedError",
    "compose_epsilon",
    "shuffle_amplification_epsilon",
    "HAS_PRV_ACCOUNTANT",
    "HEContext",
    "HEConfig",
    "PrivacyAuditor",
    "AuditResult",
    "add_privacy_args",
    "build_privacy_mods",
    "validate_env_vars",
]
