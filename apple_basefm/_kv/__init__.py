"""TurboQuant KV cache backend for AppleLocalLM.

This subpackage has no imports from the rest of apple_basefm — it depends
only on mlx and mlx-lm, both of which are lazy-imported inside methods.
This means the package is importable on Linux (for unit tests) without
MLX installed.

Public API::

    from apple_basefm._kv import KVCacheStrategy, TurboQuantV2Cache
"""
from ._base import KVCacheStrategy
from .cache_v2 import TurboQuantV2Cache
# attention_v2 is not imported here — it lazily imports mlx.fast which is
# unavailable on Linux. Import it directly: from apple_basefm._kv import attention_v2

__all__ = ["KVCacheStrategy", "TurboQuantV2Cache"]
