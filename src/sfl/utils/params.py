"""
Parameter serialization utilities.

Helpers for converting between different parameter precision formats
for communication efficiency (e.g., sending float16 over the wire
and converting to float32 for training).
"""

from typing import List

import numpy as np


def downcast_parameters(params: List[np.ndarray], dtype=np.float16) -> List[np.ndarray]:
    """Downcast parameter arrays to a lower precision dtype.

    Useful for reducing communication bandwidth by 2x when sending
    parameters between clients and server.

    Args:
        params: List of parameter arrays (any dtype).
        dtype: Target dtype (default: float16 for 2x compression).

    Returns:
        List of arrays cast to target dtype.
    """
    return [p.astype(dtype) for p in params]


def upcast_parameters(params: List[np.ndarray], dtype=np.float32) -> List[np.ndarray]:
    """Upcast parameter arrays to higher precision for computation.

    Args:
        params: List of parameter arrays (any dtype).
        dtype: Target dtype (default: float32).

    Returns:
        List of arrays cast to target dtype.
    """
    return [p.astype(dtype) for p in params]
