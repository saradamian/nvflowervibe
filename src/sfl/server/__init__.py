"""
SFL Server Module

Provides federated learning server implementations using the Flower framework.
Includes custom aggregation strategies and the server application.
"""

from sfl.server.strategy import SumFedAvg
from sfl.server.app import server_fn, app

__all__ = [
    "SumFedAvg",
    "server_fn",
    "app",
]
