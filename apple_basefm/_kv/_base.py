"""KVCacheStrategy protocol — the seam between AppleLocalLM and any cache backend.

Any object that implements build() and describe() satisfies this protocol.
AppleLocalLM checks isinstance(strategy, KVCacheStrategy) at construction time
so misconfigured objects surface immediately, not mid-generation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class KVCacheStrategy(Protocol):
    """Structural protocol for KV cache backends.

    Implement this to supply a custom cache to AppleLocalLM::

        class MyCache:
            def build(self, n_layers: int, head_dim: int) -> list:
                ...
            def describe(self) -> str:
                return "MyCache()"

        lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit",
                          kv_cache=MyCache())
    """

    def build(self, n_layers: int, head_dim: int) -> list:
        """Return a per-layer cache list ready to pass to mlx_lm.generate().

        Called once per forward() invocation — must return a *fresh* list on
        every call. Callers must not share returned cache instances across
        concurrent forward() calls.

        Args:
            n_layers: Number of transformer layers in the loaded model.
            head_dim: Attention head dimension of the loaded model.

        Returns:
            A list of n_layers cache objects that each implement
            update_and_fetch(keys, values) as expected by mlx-lm.
        """
        ...

    def describe(self) -> str:
        """Human-readable description for __repr__ and logging."""
        ...
