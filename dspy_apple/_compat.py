"""Compatibility shim: provides BaseLM and DSPy runtime hooks.

When dspy is installed, all symbols resolve to the real DSPy objects so
AppleFoundationLM and AppleLocalLM are fully-conformant BaseLM subclasses.

When dspy is NOT installed, a minimal stub is provided so the adapters
can be used standalone (direct __call__, no optimizer support).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BaseLM shim
# ---------------------------------------------------------------------------

try:
    from dspy import BaseLM  # type: ignore[import]

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

    class BaseLM:  # type: ignore[no-redef]
        """Minimal stub used when dspy is not installed.

        Implements only the __call__ protocol so adapters can be used
        directly without DSPy's optimizer, history, or callback machinery.
        """

        def __init__(self, model: str, **kwargs: Any) -> None:
            self.model = model
            self.history: list[dict] = []

        def __call__(self, messages: list[dict], **kwargs: Any) -> list[str]:
            return self.forward(messages=messages, **kwargs)

        async def acall(self, messages: list[dict], **kwargs: Any) -> list[str]:
            return await self.aforward(messages=messages, **kwargs)

        def forward(self, messages: list[dict], **kwargs: Any) -> Any:
            raise NotImplementedError("Subclasses must implement forward()")

        async def aforward(self, messages: list[dict], **kwargs: Any) -> Any:
            raise NotImplementedError("Subclasses must implement aforward()")

        # Capability properties — subclasses may override
        @property
        def supports_function_calling(self) -> bool:
            return False

        @property
        def supports_response_schema(self) -> bool:
            return False

        @property
        def supports_reasoning(self) -> bool:
            return False

        @property
        def supported_params(self) -> list[str]:
            return []


# ---------------------------------------------------------------------------
# Cache shim
# ---------------------------------------------------------------------------


class _NullCache:
    """No-op cache used when dspy is not installed."""

    def get(self, key: Any) -> Any:
        return None

    def put(self, key: Any, value: Any) -> None:
        pass


def get_dspy_cache() -> Any:
    """Return dspy.cache when available, else a no-op stub."""
    if DSPY_AVAILABLE:
        try:
            import dspy

            return dspy.cache
        except AttributeError:
            pass
    return _NullCache()


# ---------------------------------------------------------------------------
# Settings shims
# ---------------------------------------------------------------------------


def get_send_stream() -> Any:
    """Return dspy.settings.send_stream, or None if unavailable."""
    if not DSPY_AVAILABLE:
        return None
    try:
        import dspy

        return getattr(dspy.settings, "send_stream", None)
    except Exception as exc:
        logger.debug("get_send_stream: dspy.settings access failed (%s), returning None", exc)
        return None


def get_caller_predict() -> Any:
    """Return dspy.settings.caller_predict, or None if unavailable."""
    if not DSPY_AVAILABLE:
        return None
    try:
        import dspy

        return getattr(dspy.settings, "caller_predict", None)
    except Exception as exc:
        logger.debug("get_caller_predict: dspy.settings access failed (%s), returning None", exc)
        return None


def get_usage_tracker() -> Any:
    """Return dspy.settings.usage_tracker, or None if unavailable."""
    if not DSPY_AVAILABLE:
        return None
    try:
        import dspy

        return getattr(dspy.settings, "usage_tracker", None)
    except Exception as exc:
        logger.debug("get_usage_tracker: dspy.settings access failed (%s), returning None", exc)
        return None


__all__ = [
    "DSPY_AVAILABLE",
    "BaseLM",
    "get_dspy_cache",
    "get_send_stream",
    "get_caller_predict",
    "get_usage_tracker",
]
