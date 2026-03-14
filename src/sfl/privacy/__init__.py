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

from sfl.privacy.dp import wrap_strategy_with_dp, DPConfig
from sfl.privacy.filters import (
    make_percentile_privacy_mod,
    make_svt_privacy_mod,
    make_exclude_vars_mod,
    PercentilePrivacyConfig,
    SVTPrivacyConfig,
)
from sfl.privacy.secagg import build_secagg_config, SecAggConfig

# HE is optional — requires tenseal
try:
    from sfl.privacy.he import HEContext, HEConfig
except ImportError:
    pass

__all__ = [
    "wrap_strategy_with_dp",
    "DPConfig",
    "build_secagg_config",
    "SecAggConfig",
    "make_percentile_privacy_mod",
    "make_svt_privacy_mod",
    "make_exclude_vars_mod",
    "PercentilePrivacyConfig",
    "SVTPrivacyConfig",
    "HEContext",
    "HEConfig",
]
