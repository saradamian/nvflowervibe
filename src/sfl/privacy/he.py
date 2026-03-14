"""
Homomorphic encryption support for federated learning.

Uses TenSEAL's CKKS scheme to encrypt model parameters so the
server can aggregate them without ever seeing plaintext values.

How it works:
  1. A shared TenSEAL context (with CKKS keys) is created once
  2. Clients encrypt their parameter updates before sending
  3. The server sums encrypted vectors (CKKS supports addition)
  4. The server decrypts only the aggregate — individual client
     contributions are never visible in plaintext

Limitations:
  - Requires ``pip install tenseal`` (not installed by default)
  - Each encrypted float32 expands to ~160KB of ciphertext,
    making it impractical for large models (ESM2 has 8M params)
  - Best suited for small-parameter demos (like the sum demo)
  - Only addition and scalar multiplication are supported on
    ciphertext — weighted averaging requires careful encoding
  - For production HE with large models, use NVFlare's native
    FedJob pipeline with HEModelEncryptor/HEModelDecryptor

Usage:
    from sfl.privacy.he import HEContext

    he_ctx = HEContext()
    encrypted = he_ctx.encrypt_parameters(ndarrays)
    # ... server aggregates encrypted params via CKKS addition ...
    decrypted = he_ctx.decrypt_parameters(encrypted, shapes)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False


@dataclass
class HEConfig:
    """Homomorphic encryption configuration.

    Args:
        poly_modulus_degree: Polynomial modulus degree. Higher = more
            security and precision but slower. 8192 is standard.
        coeff_mod_bit_sizes: Coefficient modulus bit sizes. Controls
            precision and multiplication depth.
        global_scale: CKKS encoding scale (2^40 is standard).
    """
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(
        default_factory=lambda: [60, 40, 40, 60]
    )
    global_scale: float = 2**40


class HEContext:
    """Manages TenSEAL CKKS context for encrypting/decrypting parameters.

    Creates a CKKS encryption context with public and secret keys.
    In a real deployment, the secret key would be managed separately
    (e.g., by a trusted key authority), but for simulation both
    encryption and decryption use the same context.
    """

    def __init__(self, config: Optional[HEConfig] = None):
        if not HAS_TENSEAL:
            raise ImportError(
                "tenseal is required for homomorphic encryption. "
                "Install with: pip install tenseal"
            )

        cfg = config or HEConfig()
        self._ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=cfg.poly_modulus_degree,
            coeff_mod_bit_sizes=cfg.coeff_mod_bit_sizes,
        )
        self._ctx.global_scale = cfg.global_scale
        self._ctx.generate_galois_keys()

        logger.info(
            "HE context created: poly_mod=%d, scale=2^%d",
            cfg.poly_modulus_degree,
            int(np.log2(cfg.global_scale)),
        )

    def encrypt_parameters(
        self, ndarrays: List[np.ndarray],
    ) -> List[bytes]:
        """Encrypt a list of NDArrays into serialized CKKS ciphertexts.

        Each NDArray is flattened, converted to float64, encrypted as
        a CKKS vector, and serialized to bytes.

        Args:
            ndarrays: Model parameters as numpy arrays.

        Returns:
            List of serialized encrypted vectors (bytes).
        """
        encrypted = []
        for arr in ndarrays:
            flat = arr.ravel().astype(np.float64).tolist()
            enc_vec = ts.ckks_vector(self._ctx, flat)
            encrypted.append(enc_vec.serialize())
        return encrypted

    def decrypt_parameters(
        self,
        encrypted: List[bytes],
        shapes: List[Tuple[int, ...]],
        dtypes: Optional[List[np.dtype]] = None,
    ) -> List[np.ndarray]:
        """Decrypt serialized CKKS ciphertexts back to NDArrays.

        Args:
            encrypted: List of serialized encrypted vectors.
            shapes: Original shapes of each parameter array.
            dtypes: Original dtypes (defaults to float32).

        Returns:
            List of decrypted NDArrays with original shapes.
        """
        if dtypes is None:
            dtypes = [np.float32] * len(encrypted)

        decrypted = []
        for enc_bytes, shape, dtype in zip(encrypted, shapes, dtypes):
            enc_vec = ts.lazy_ckks_vector_from(enc_bytes)
            enc_vec.link_context(self._ctx)
            flat = np.array(enc_vec.decrypt(), dtype=dtype)
            # CKKS may return more values than needed (padded)
            n_elements = int(np.prod(shape))
            decrypted.append(flat[:n_elements].reshape(shape))
        return decrypted

    def add_encrypted(
        self, enc_a: List[bytes], enc_b: List[bytes],
    ) -> List[bytes]:
        """Add two sets of encrypted parameters element-wise.

        CKKS supports homomorphic addition, so the server can sum
        encrypted client updates without decrypting them.

        Args:
            enc_a: First set of serialized encrypted vectors.
            enc_b: Second set of serialized encrypted vectors.

        Returns:
            Element-wise sum as serialized encrypted vectors.
        """
        result = []
        for a_bytes, b_bytes in zip(enc_a, enc_b):
            vec_a = ts.lazy_ckks_vector_from(a_bytes)
            vec_a.link_context(self._ctx)
            vec_b = ts.lazy_ckks_vector_from(b_bytes)
            vec_b.link_context(self._ctx)
            summed = vec_a + vec_b
            result.append(summed.serialize())
        return result
