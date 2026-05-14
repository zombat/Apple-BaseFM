"""SDPA patch for TurboQuant V2 rotated KV cache.

When use_rotation=True, cache_v2 stores K and V in rotated space (K@R, V@R).
mlx-lm's SDPA then computes Q @ (K·R)ᵀ instead of the correct Q @ Kᵀ.

This module patches mlx.fast.scaled_dot_product_attention to apply the
compensating query rotation Q' = Q @ R before scoring, so that:

    Q' @ (K·R)ᵀ = (Q·R) @ (Rᵀ·Kᵀ) = Q·(R·Rᵀ)·Kᵀ = Q·Kᵀ  ✓

Note on notation: cache_v2's docstring uses column-vector convention "R.T @ Q";
in MLX's row-vector layout (batch, heads, seq, head_dim) this is Q @ R.

Usage (managed automatically by AppleLocalLM when use_rotation=True):

    with rotated_sdpa_context(rotation_matrix):
        mlx_lm.generate(...)
"""
from __future__ import annotations

import contextlib
import logging
from typing import Generator

logger = logging.getLogger(__name__)

_original_sdpa = None
_original_qsdpa = None
_active_rotation = None


def install(rotation_matrix) -> None:
    """Patch SDPA functions to rotate queries by R before scoring.

    Patches two functions so that Q rotation is applied regardless of which
    attention path mlx-lm chooses at runtime:

    1. ``mlx.fast.scaled_dot_product_attention`` — used when the KV cache does
       NOT expose ``.bits`` (fp16/bf16 path).
    2. ``mlx_lm.models.base.quantized_scaled_dot_product_attention`` — used
       when the KV cache DOES expose ``.bits`` (quantized path, e.g.
       TurboQuantV2Cache).  mlx-lm looks up this name in ``base``'s module
       namespace at call time, so patching the module attribute takes effect
       immediately without needing to re-import model files.

    Must be called after TurboQuantV2Cache.build() (which generates the
    rotation matrix) and before mlx_lm.generate() / mlx_lm.stream_generate().
    Call uninstall() after generation to restore the original functions.

    Args:
        rotation_matrix: The orthogonal rotation matrix from make_rotation_matrix(),
            shape (head_dim, head_dim). Must not be None.

    Raises:
        RuntimeError: If the patch is already installed, if mlx is not available,
            or if mlx.fast.scaled_dot_product_attention is not found.
        ValueError: If rotation_matrix is None.
    """
    global _original_sdpa, _original_qsdpa, _active_rotation  # noqa: PLW0603

    if rotation_matrix is None:
        raise ValueError("rotation_matrix must not be None")
    if _original_sdpa is not None:
        raise RuntimeError(
            "attention_v2 SDPA patch is already installed. "
            "Call uninstall() before re-installing."
        )

    try:
        import mlx.core as _mx
        _mlx_fast = _mx.fast
    except ImportError as exc:
        raise RuntimeError(
            "mlx is required for the TurboQuant V2 SDPA patch. "
            "Install with: pip install 'apple-basefm[mlx]'"
        ) from exc

    original = getattr(_mlx_fast, "scaled_dot_product_attention", None)
    if original is None:
        raise RuntimeError(
            "mlx.core.fast.scaled_dot_product_attention not found. "
            "Upgrade MLX: pip install 'mlx-lm>=0.22.0'"
        )

    _original_sdpa = original
    _active_rotation = rotation_matrix

    def _patched_sdpa(queries, keys, values, scale, mask=None, **kwargs):
        rot = _active_rotation
        q_dtype = getattr(queries, "dtype", None)
        if q_dtype is not None and getattr(rot, "dtype", None) != q_dtype:
            rot = rot.astype(q_dtype)
        queries = queries @ rot
        return _original_sdpa(queries, keys, values, scale=scale, mask=mask, **kwargs)

    _mlx_fast.scaled_dot_product_attention = _patched_sdpa

    # Also patch the quantized SDPA path.  mlx-lm's
    # scaled_dot_product_attention (base.py) looks up
    # quantized_scaled_dot_product_attention by name in base's module
    # namespace, so replacing the attribute is sufficient.
    try:
        import mlx_lm.models.base as _base
        original_qsdpa = getattr(_base, "quantized_scaled_dot_product_attention", None)
        if original_qsdpa is not None:
            _original_qsdpa = original_qsdpa

            def _patched_qsdpa(queries, q_keys, q_values, scale, mask=None, group_size=64, bits=8):
                rot = _active_rotation
                q_dtype = getattr(queries, "dtype", None)
                if q_dtype is not None and getattr(rot, "dtype", None) != q_dtype:
                    rot = rot.astype(q_dtype)
                queries = queries @ rot
                return _original_qsdpa(
                    queries, q_keys, q_values,
                    scale=scale, mask=mask, group_size=group_size, bits=bits,
                )

            _base.quantized_scaled_dot_product_attention = _patched_qsdpa
    except ImportError:
        pass  # mlx_lm not available; only the mx.fast path is patched

    logger.debug(
        "attention_v2: SDPA patch installed (head_dim=%d)",
        getattr(rotation_matrix, "shape", [None])[0],
    )


def uninstall() -> None:
    """Restore patched SDPA functions to their originals. No-op if not installed."""
    global _original_sdpa, _original_qsdpa, _active_rotation  # noqa: PLW0603

    if _original_sdpa is None:
        return
    try:
        import mlx.core as _mx
        _mx.fast.scaled_dot_product_attention = _original_sdpa
    except ImportError:
        pass
    _original_sdpa = None

    if _original_qsdpa is not None:
        try:
            import mlx_lm.models.base as _base
            _base.quantized_scaled_dot_product_attention = _original_qsdpa
        except ImportError:
            pass
        _original_qsdpa = None

    _active_rotation = None
    logger.debug("attention_v2: SDPA patch uninstalled")


def is_installed() -> bool:
    """Return True if the SDPA patch is currently active."""
    return _original_sdpa is not None


@contextlib.contextmanager
def rotated_sdpa_context(rotation_matrix) -> Generator[None, None, None]:
    """Context manager: install the SDPA patch, yield, then always uninstall.

    Args:
        rotation_matrix: Passed directly to install(). Must not be None.
    """
    install(rotation_matrix)
    try:
        yield
    finally:
        uninstall()
