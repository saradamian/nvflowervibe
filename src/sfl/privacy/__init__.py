"""
Privacy module for SFL federated learning.

Provides differential privacy (DP) and secure aggregation (SecAgg)
utilities that can be applied to any Flower strategy.

DP modes:
  - Server-side: server clips and adds noise after aggregation
  - Client-side: clients clip updates locally, server adds noise

SecAgg:
  - SecAgg+ protocol ensures the server only sees the aggregate,
    not individual client updates
"""

from sfl.privacy.dp import wrap_strategy_with_dp, DPConfig
from sfl.privacy.secagg import build_secagg_config, SecAggConfig

__all__ = [
    "wrap_strategy_with_dp",
    "DPConfig",
    "build_secagg_config",
    "SecAggConfig",
]
