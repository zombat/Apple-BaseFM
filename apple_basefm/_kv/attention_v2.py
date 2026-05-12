"""
SDPA patch for TurboQuant V2 rotated KV cache.

NOT YET IMPLEMENTED. When shipped, this module will:
  - Monkey-patch mlx-lm's SDPA dispatch to use mx.quantized_matmul
    directly against the rotated K/V buffers
  - Apply the inverse rotation (R.T @ Q) to queries before scoring,
    so attention is computed correctly in rotated space
  - Enable TurboQuantV2Cache(use_rotation=True) as the default preset

Until then, use use_rotation=False (LEAN mode).
"""
raise NotImplementedError(
    "attention_v2.py is not yet implemented. "
    "Use TurboQuantV2Cache(use_rotation=False) (LEAN mode)."
)
