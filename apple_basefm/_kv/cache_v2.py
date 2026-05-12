"""TurboQuant V2 affine KV-cache quantization backend.

TurboQuantV2Cache implements the KVCacheStrategy protocol. It wraps
mlx_lm.models.cache.QuantizedKVCache (mlx-lm's own quantized cache) and
optionally applies a shared QR rotation matrix to K and V tensors before
quantization, distributing outlier channels for lower quantization error.

Design note — rotation correctness:
    The rotation is applied inside update_and_fetch() before delegating to
    QuantizedKVCache. QuantizedKVCache dequantizes before returning, so the
    values returned to the model's attention are in *rotated* space. This means
    attention scores are computed as Q @ (K·R)ᵀ rather than Q @ Kᵀ, which
    changes scores by a rotation. Full correctness (exact attention equivalence)
    requires the paired SDPA patch in _kv/attention_v2.py (deferred). Without
    it, the rotation still reduces quantization error but is not lossless.
    Use use_rotation=False (LEAN mode) for the strictest numerical equivalence
    to standard mlx-lm --kv-bits behaviour.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# MLX mx.quantize only supports these bit widths.
_VALID_BITS: frozenset[int] = frozenset({2, 4, 8})


@dataclass
class TurboQuantV2Cache:
    """V2 affine KV-cache quantization with optional QR rotation.

    Implements the KVCacheStrategy protocol.

    Each call to build() returns a fresh list of _TurboQuantV2LayerCache
    objects, one per transformer layer. Each layer cache wraps mlx-lm's
    QuantizedKVCache and applies the shared rotation matrix (if configured)
    before every quantize/store/dequantize cycle.

    The rotation matrix is computed once on the first build() call and
    cached on this instance. Subsequent build() calls (one per forward())
    reuse the same matrix but create fresh layer caches — there is no
    shared mutable state between forward() passes.

    Args:
        bits: Bits per quantized element. Must be in {2, 4, 8}. Default: 4.
        group_size: Number of elements per quantization group. Default: 64.
        use_rotation: Apply QR rotation before quantization. Default: True.
            Set False for LEAN mode (fastest, exact numerical equivalence
            to mlx-lm --kv-bits).
        use_normalization: Reserved for future attention_v2.py SDPA patch.
            Currently a no-op. Default: True.
        step: Pre-allocation step size forwarded to QuantizedKVCache. Controls
            how many extra token slots are allocated when the cache grows.
            Default: 256.

    Raises:
        ValueError: At construction time if bits, group_size, or step is
            outside valid ranges.
    """

    bits: int = 4
    group_size: int = 64
    use_rotation: bool = True
    use_normalization: bool = True  # reserved — wired in attention_v2.py (future)
    step: int = 256

    def __post_init__(self) -> None:
        # --- Governance: validate all numeric parameters at the boundary ---
        if self.bits not in _VALID_BITS:
            raise ValueError(
                f"TurboQuantV2Cache: bits must be one of {sorted(_VALID_BITS)}, "
                f"got {self.bits!r}. MLX mx.quantize only supports these values."
            )
        if self.group_size < 1:
            raise ValueError(
                f"TurboQuantV2Cache: group_size must be >= 1, got {self.group_size!r}."
            )
        if self.step < 1:
            raise ValueError(
                f"TurboQuantV2Cache: step must be >= 1, got {self.step!r}."
            )
        if self.use_normalization:
            logger.debug(
                "TurboQuantV2Cache: use_normalization=True is reserved for "
                "attention_v2.py (future); currently a no-op."
            )
        # Lazy rotation matrix — computed once on first build() call, then reused.
        self._rotation = None

    def build(self, n_layers: int, head_dim: int) -> list:
        """Return a fresh per-layer cache list for one generation pass.

        Args:
            n_layers: Number of transformer layers in the model.
            head_dim: Attention head dimension.

        Returns:
            A list of n_layers _TurboQuantV2LayerCache instances.

        Raises:
            RuntimeError: If mlx or mlx-lm is not installed, or if rotation
                matrix generation fails.
        """
        # Lazy init: compute rotation matrix once, reuse across all forward() calls.
        if self.use_rotation and self._rotation is None:
            from .rotation import make_rotation_matrix

            self._rotation = make_rotation_matrix(head_dim)

        if self.group_size > head_dim:
            logger.debug(
                "TurboQuantV2Cache: group_size (%d) > head_dim (%d); "
                "MLX will treat the entire head as one quantization group.",
                self.group_size,
                head_dim,
            )

        # Import check — surfaces a clear error before entering generation.
        try:
            from mlx_lm.models.cache import QuantizedKVCache  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is required for TurboQuantV2Cache. "
                "Install with: pip install 'apple-basefm[mlx]'"
            ) from exc

        return [
            _TurboQuantV2LayerCache(
                bits=self.bits,
                group_size=self.group_size,
                step=self.step,
                rotation=self._rotation,
            )
            for _ in range(n_layers)
        ]

    def describe(self) -> str:
        mode = "rotated" if self.use_rotation else "LEAN"
        return (
            f"TurboQuantV2Cache(bits={self.bits}, mode={mode}, "
            f"group_size={self.group_size})"
        )


class _TurboQuantV2LayerCache:
    """Per-layer cache wrapper for one transformer layer.

    Applies the shared QR rotation matrix to K and V (if configured), then
    delegates all quantization, storage, and retrieval to an inner
    mlx_lm.models.cache.QuantizedKVCache. This ensures:
      - Correct quantized buffer layout (scales, biases, data all stored)
      - Correct return type: dequantized float tensors, as models expect
      - Correct mlx-lm interface: state, offset, nbytes, is_trimmable()

    The inner QuantizedKVCache is lazily constructed on the first
    update_and_fetch() call to avoid binding to the MLX runtime at list
    construction time.
    """

    def __init__(
        self,
        bits: int,
        group_size: int,
        step: int,
        rotation,
    ) -> None:
        self._bits = bits
        self._group_size = group_size
        self._step = step
        self.rotation = rotation
        self._inner = None  # lazy — mlx_lm imported at first update_and_fetch

    def _ensure_inner(self) -> None:
        if self._inner is not None:
            return
        from mlx_lm.models.cache import QuantizedKVCache

        inner = QuantizedKVCache(bits=self._bits, group_size=self._group_size)
        inner.step = self._step
        self._inner = inner

    def update_and_fetch(self, keys, values):
        """Rotate (if configured), quantize, store, and return dequantized K/V.

        The returned tensors are floats in rotated space if use_rotation=True.
        Full correctness (attention computed as Q@Kᵀ) requires the SDPA patch
        in attention_v2.py (deferred). Without it, scores are computed in
        rotated space, which reduces quantization error but is not lossless.

        Args:
            keys: Key tensor for the current step (from the model).
            values: Value tensor for the current step (from the model).

        Returns:
            (all_keys, all_values): Dequantized float tensors for all stored
            tokens including the current step, ready for use in SDPA.
        """
        self._ensure_inner()
        if self.rotation is not None:
            keys = keys @ self.rotation
            values = values @ self.rotation
        return self._inner.update_and_fetch(keys, values)

    @property
    def state(self):
        """Quantized buffer arrays for mx.eval() in mlx-lm's generation loop."""
        if self._inner is None:
            return []
        return self._inner.state

    @property
    def offset(self) -> int:
        """Number of tokens currently stored in this layer's cache."""
        if self._inner is None:
            return 0
        return self._inner.offset

    @property
    def nbytes(self) -> int:
        """Total bytes occupied by quantized K and V buffers."""
        if self._inner is None:
            return 0
        return self._inner.nbytes
