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
import time

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
        ValueError: If head_dim is less than 1.
        RuntimeError: If mlx is not installed, or if MLX < 0.16 is detected
            (mx.linalg.qr is not available before 0.16), or if QR
            decomposition fails for any other reason (e.g. bad tensor shape).
    """
    if head_dim < 1:
        raise ValueError(
            f"make_rotation_matrix: head_dim must be >= 1, got {head_dim!r}."
        )
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "mlx is required for TurboQuantV2Cache rotation. "
            "Install with: pip install 'apple-basefm[mlx]'"
        ) from exc

    try:
        # Try the keyed PRNG API (MLX ≥ 0.16 style) first; fall back to the
        # global-seed API for older releases or test stubs that don't expose
        # mx.random.key.
        try:
            key = mx.random.key(42)
            gaussian = mx.random.normal(shape=(head_dim, head_dim), key=key)
        except AttributeError:
            mx.random.seed(42)
            gaussian = mx.random.normal(shape=(head_dim, head_dim))
        t0 = time.perf_counter()
        # mx.linalg.qr is CPU-only in current MLX releases — explicitly pin
        # the op to the CPU stream rather than inheriting the GPU default.
        with mx.stream(mx.cpu):
            Q, _ = mx.linalg.qr(gaussian)
            mx.eval(Q)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Generated %dx%d QR rotation matrix for KV cache in %.3fs",
            head_dim, head_dim, elapsed,
            extra={"elapsed_ms": round(elapsed * 1000, 2), "backend": "mlx"},
        )
    except (AttributeError, TypeError) as exc:
        exc_str = str(exc).lower()
        if "qr" in exc_str or "linalg" in exc_str:
            raise RuntimeError(
                "mx.linalg.qr requires MLX >= 0.16. "
                "Upgrade with: pip install 'mlx-lm>=0.22.0'"
            ) from exc
        raise RuntimeError(
            f"make_rotation_matrix failed for head_dim={head_dim}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"make_rotation_matrix failed for head_dim={head_dim}: {exc}"
        ) from exc

    logger.debug(
        "Generated %dx%d QR rotation matrix for KV cache",
        head_dim, head_dim,
        extra={"backend": "mlx"},
    )
    return Q
