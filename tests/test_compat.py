"""Tests for apple_basefm._compat stub code paths when dspy is NOT installed.

Coverage targets:
  - BaseLM stub class (lines 26-65): __init__, __call__, acall, forward, aforward,
    capability properties
  - _NullCache + get_dspy_cache() (lines 77-80)
  - get_send_stream(), get_caller_predict(), get_usage_tracker() (lines 90-129)
    when DSPY_AVAILABLE is False
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helper — reload _compat with dspy absent from sys.modules
# ---------------------------------------------------------------------------


def _reload_compat_without_dspy():
    """Temporarily block dspy and reload _compat to force DSPY_AVAILABLE=False."""
    # Setting sys.modules["dspy"] = None tells Python's import machinery that
    # dspy failed to import; any subsequent `import dspy` or `from dspy import X`
    # will raise ImportError immediately without re-executing the package.
    # Simply popping the key is insufficient because Python will just re-import
    # the (installed) package, which succeeds and sets DSPY_AVAILABLE=True.
    _MISSING = object()
    saved = sys.modules.get("dspy", _MISSING)
    sys.modules["dspy"] = None  # type: ignore[assignment]
    sys.modules.pop("apple_basefm._compat", None)
    try:
        return importlib.import_module("apple_basefm._compat")
    finally:
        # Restore dspy so other tests are not affected.
        if saved is _MISSING:
            sys.modules.pop("dspy", None)
        else:
            sys.modules["dspy"] = saved
        # Force a clean reload of _compat on next import so the real
        # DSPY_AVAILABLE=True module is restored for subsequent tests.
        sys.modules.pop("apple_basefm._compat", None)
        importlib.import_module("apple_basefm._compat")


# ---------------------------------------------------------------------------
# DSPY_AVAILABLE flag
# ---------------------------------------------------------------------------


class TestDspyAvailableFlag:
    def test_false_when_dspy_absent(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.DSPY_AVAILABLE is False


# ---------------------------------------------------------------------------
# Stub BaseLM
# ---------------------------------------------------------------------------


class TestStubBaseLM:
    def test_init_stores_model(self) -> None:
        mod = _reload_compat_without_dspy()
        lm = mod.BaseLM(model="stub-model")
        assert lm.model == "stub-model"
        assert lm.history == []

    def test_call_delegates_to_forward_raises(self) -> None:
        mod = _reload_compat_without_dspy()
        lm = mod.BaseLM(model="stub")
        with pytest.raises(NotImplementedError):
            lm(messages=[{"role": "user", "content": "hi"}])

    def test_acall_delegates_to_aforward_raises(self) -> None:
        mod = _reload_compat_without_dspy()
        lm = mod.BaseLM(model="stub")
        with pytest.raises(NotImplementedError):
            asyncio.run(lm.acall(messages=[{"role": "user", "content": "hi"}]))

    def test_forward_raises_not_implemented(self) -> None:
        mod = _reload_compat_without_dspy()
        lm = mod.BaseLM(model="stub")
        with pytest.raises(NotImplementedError):
            lm.forward(messages=[])

    def test_aforward_raises_not_implemented(self) -> None:
        mod = _reload_compat_without_dspy()
        lm = mod.BaseLM(model="stub")
        with pytest.raises(NotImplementedError):
            asyncio.run(lm.aforward(messages=[]))

    def test_supports_function_calling_false(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.BaseLM(model="x").supports_function_calling is False

    def test_supports_response_schema_false(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.BaseLM(model="x").supports_response_schema is False

    def test_supports_reasoning_false(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.BaseLM(model="x").supports_reasoning is False

    def test_supported_params_empty(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.BaseLM(model="x").supported_params == []


# ---------------------------------------------------------------------------
# _NullCache via get_dspy_cache()
# ---------------------------------------------------------------------------


class TestNullCache:
    def test_get_returns_none(self) -> None:
        mod = _reload_compat_without_dspy()
        cache = mod.get_dspy_cache()
        assert cache.get("any_key") is None

    def test_put_is_noop(self) -> None:
        mod = _reload_compat_without_dspy()
        cache = mod.get_dspy_cache()
        cache.put("key", "value")  # must not raise


# ---------------------------------------------------------------------------
# Shims return None when dspy is absent
# ---------------------------------------------------------------------------


class TestShimsWithoutDspy:
    def test_get_send_stream_returns_none(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.get_send_stream() is None

    def test_get_caller_predict_returns_none(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.get_caller_predict() is None

    def test_get_usage_tracker_returns_none(self) -> None:
        mod = _reload_compat_without_dspy()
        assert mod.get_usage_tracker() is None
