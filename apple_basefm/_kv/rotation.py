"""QR rotation matrix for TurboQuant KV cache pre-quantization.

Applying a random orthogonal rotation to K and V before quantization
distributes outlier channels evenly across all dimensions, reducing affine
quantization error without changing the bit budget. The same matrix is shared
across all layers and both K and V, matching the TurboQuant paper's approach.

A fixed seed (42) is intentional — the quality benefit comes from the rotation
structure, not from randomness at inference time, and determinism makes
quantization behaviour reproducible across runs.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def make_rotation_matrix(head_dim: int):
    """Generate a random orthogonal rotation matrix via QR decomposition.

    Shape: (head_dim, head_dim). Properties:
      - Orthogonal: Q @ Q.T = I  (lossless; inverse = transpose)
      - Distributes outlier channels evenly (Gaussian → iid after rotation)
      - Deterministic: seed=42 ensures reproducible quantization behaviour

    Args:
        head_dim: Attention head dimension of the model.

    Returns:
        An (head_dim, head_dim) orthogonal array in the active MLX default dtype.

    Raises:
        RuntimeError: If mlx is not installed, or if MLX < 0.16 is detected
            (mx.linalg.qr is not available before 0.16).
    """
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "mlx is required for TurboQuantV2Cache rotation. "
            "Install with: pip install 'apple-basefm[mlx]'"
        ) from exc

    try:
        # Fixed seed: same matrix every run — reproducibility over variety.
        mx.random.seed(42)
        gaussian = mx.random.normal(shape=(head_dim, head_dim))
        Q, _ = mx.linalg.qr(gaussian)
    except AttributeError as exc:
        if "qr" in str(exc).lower():
            raise RuntimeError(
                "mx.linalg.qr requires MLX >= 0.16. "
                "Upgrade with: pip install 'mlx-lm>=0.22.0'"
            ) from exc
        raise

    logger.debug("Generated %dx%d QR rotation matrix for KV cache", head_dim, head_dim)
    return Q
