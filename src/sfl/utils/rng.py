"""Cryptographically-seeded random number generators for DP noise.

Using a CSRNG-seeded numpy RandomState ensures that DP noise is not
predictable from the default (time-based or zero) seed, which could
otherwise allow an adversary to reconstruct the noise and strip the
privacy guarantee.
"""

import os

import numpy as np


def secure_rng() -> np.random.RandomState:
    """Return a numpy RandomState seeded from os.urandom (CSRNG).

    This should be used for all DP noise generation paths (Gaussian,
    Laplace) to ensure unpredictable noise. Non-DP paths (e.g., JL
    projection for dimensionality reduction) can use deterministic seeds.
    """
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    return np.random.RandomState(seed)
