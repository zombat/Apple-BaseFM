"""Unit tests for apple_basefm._session (token accumulation).

Tests cover:
  - Accumulation across multiple calls
  - No-op outside an active token_session context
  - Nesting — inner session is isolated from outer
  - Passing an existing accumulator (resume across blocks)
  - TypeError when accumulator is the wrong type
  - call_count increments correctly
  - ContextVar is reset on exception from the body
  - reset_usage() resets per-instance counter
  - _accumulate handles missing / non-int / negative attributes safely
  - lm.usage tracks per-instance lifetime totals via _build_response
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from apple_basefm._session import (
    _SessionAccumulator,
    _accumulate,
    _current_session,
    token_session,
)


# ---------------------------------------------------------------------------
# _SessionAccumulator
# ---------------------------------------------------------------------------


class TestSessionAccumulator:
    def test_defaults_are_zero(self) -> None:
        acc = _SessionAccumulator()
        assert acc.prompt_tokens == 0
        assert acc.completion_tokens == 0
        assert acc.total_tokens == 0
        assert acc.call_count == 0


# ---------------------------------------------------------------------------
# token_session context manager
# ---------------------------------------------------------------------------


class TestTokenSession:
    def test_accumulates_across_calls(self) -> None:
        """_accumulate() called twice inside a session adds both counts."""
        usage1 = MagicMock(prompt_tokens=10, completion_tokens=5)
        usage2 = MagicMock(prompt_tokens=20, completion_tokens=15)

        with token_session() as session:
            _accumulate(usage1)
            _accumulate(usage2)

        assert session.prompt_tokens == 30
        assert session.completion_tokens == 20
        assert session.total_tokens == 50
        assert session.call_count == 2

    def test_no_op_outside_context(self) -> None:
        """_accumulate outside a session must not raise and must leave acc untouched."""
        acc = _SessionAccumulator()
        # No active session — ensure ContextVar holds None
        assert _current_session.get() is None
        usage = MagicMock(prompt_tokens=99, completion_tokens=99)
        _accumulate(usage)  # should be a silent no-op
        # acc was never entered, so still zero
        assert acc.total_tokens == 0

    def test_nested_sessions_are_isolated(self) -> None:
        """Inner token_session must not pollute the outer accumulator."""
        usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with token_session() as outer:
            _accumulate(usage)  # outer: 15 total
            with token_session() as inner:
                _accumulate(usage)  # inner: 15 total, outer stays at 15

        assert outer.total_tokens == 15
        assert outer.call_count == 1
        assert inner.total_tokens == 15
        assert inner.call_count == 1

    def test_outer_resumes_after_inner(self) -> None:
        """After inner session exits, the outer session must be active again."""
        usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with token_session() as outer:
            _accumulate(usage)            # outer: 1 call
            with token_session():
                _accumulate(usage)        # inner only
            _accumulate(usage)            # outer: 2nd call

        assert outer.call_count == 2
        assert outer.total_tokens == 30

    def test_pass_existing_accumulator(self) -> None:
        """An existing accumulator can be reused across multiple blocks."""
        acc = _SessionAccumulator()
        usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with token_session(accumulator=acc):
            _accumulate(usage)
        with token_session(accumulator=acc):
            _accumulate(usage)

        assert acc.total_tokens == 30
        assert acc.call_count == 2

    def test_wrong_accumulator_type_raises(self) -> None:
        """Passing a non-_SessionAccumulator raises TypeError."""
        with pytest.raises(TypeError, match="_SessionAccumulator"):
            with token_session(accumulator="bad"):  # type: ignore[arg-type]
                pass

    def test_contextvar_reset_on_exception(self) -> None:
        """ContextVar must be reset even if the body raises."""
        usage = MagicMock(prompt_tokens=5, completion_tokens=3)

        with pytest.raises(ZeroDivisionError):
            with token_session():
                _accumulate(usage)
                raise ZeroDivisionError("deliberate")

        # After the exception exits the context, no session should be active
        assert _current_session.get() is None

    def test_yields_accumulator(self) -> None:
        """The context manager must yield the same _SessionAccumulator instance."""
        acc = _SessionAccumulator()
        with token_session(accumulator=acc) as yielded:
            assert yielded is acc

    def test_yields_fresh_accumulator_when_none(self) -> None:
        """Without an explicit accumulator, a fresh one is created and yielded."""
        with token_session() as session:
            assert isinstance(session, _SessionAccumulator)


# ---------------------------------------------------------------------------
# _accumulate
# ---------------------------------------------------------------------------


class TestAccumulate:
    def test_handles_missing_attributes(self) -> None:
        """Usage objects without the expected attributes must not crash."""
        with token_session() as session:
            _accumulate(object())  # bare object, no token attrs

        assert session.total_tokens == 0
        assert session.call_count == 1  # call_count still increments

    def test_clamps_negative_values(self) -> None:
        """Negative token counts must be clamped to zero."""
        usage = MagicMock(prompt_tokens=-10, completion_tokens=-5)

        with token_session() as session:
            _accumulate(usage)

        assert session.prompt_tokens == 0
        assert session.completion_tokens == 0
        assert session.total_tokens == 0

    def test_handles_float_token_values(self) -> None:
        """Float token counts must be coerced to int via int()."""
        usage = MagicMock(prompt_tokens=7.9, completion_tokens=3.1)

        with token_session() as session:
            _accumulate(usage)

        assert session.prompt_tokens == 7
        assert session.completion_tokens == 3
        assert session.total_tokens == 10


# ---------------------------------------------------------------------------
# _AppleBaseLM.usage (per-instance lifetime counter)
# ---------------------------------------------------------------------------


class TestInstanceUsage:
    def _make_lm(self) -> object:
        """Return an _AppleBaseLM instance with fake mlx patched."""
        import importlib
        import platform
        import sys
        import types

        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = MagicMock()  # type: ignore[attr-defined]
        fake_mlx_lm = types.ModuleType("mlx_lm")
        fake_mlx_lm.load = MagicMock(return_value=(MagicMock(), MagicMock()))  # type: ignore[attr-defined]
        fake_mlx_lm.generate = MagicMock(return_value="hello world")  # type: ignore[attr-defined]
        fake_mlx_lm.stream_generate = MagicMock(return_value=iter([]))  # type: ignore[attr-defined]
        fake_mlx_lm.sample_utils = MagicMock()  # type: ignore[attr-defined]
        sys.modules.setdefault("mlx", fake_mlx)
        sys.modules.setdefault("mlx.core", fake_mlx.core)
        sys.modules.setdefault("mlx_lm", fake_mlx_lm)
        sys.modules.setdefault("mlx_lm.sample_utils", fake_mlx_lm.sample_utils)

        if "apple_basefm.apple_local" in sys.modules:
            del sys.modules["apple_basefm.apple_local"]

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            mod = importlib.import_module("apple_basefm.apple_local")
            return mod.AppleLocalLM("mlx-community/test-model-4bit")

    def test_usage_initialises_to_zero(self) -> None:
        lm = self._make_lm()
        assert lm.usage.total_tokens == 0
        assert lm.usage.call_count == 0

    def test_reset_usage(self) -> None:
        lm = self._make_lm()
        # Manually push counts into the accumulator
        lm.usage.prompt_tokens = 50
        lm.usage.completion_tokens = 25
        lm.usage.total_tokens = 75
        lm.usage.call_count = 3

        lm.reset_usage()

        assert lm.usage.total_tokens == 0
        assert lm.usage.call_count == 0

    def test_usage_accumulates_via_build_response(self) -> None:
        """_build_response() must update lm.usage on each call."""
        from apple_basefm._base import _AppleBaseLM
        from apple_basefm._response import _FMUsage

        # Instantiate base directly (it only needs model= passed through BaseLM)
        lm = _AppleBaseLM(model="test-model")
        usage = _FMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        lm._build_response("hello", usage=usage)
        assert lm.usage.prompt_tokens == 10
        assert lm.usage.completion_tokens == 5
        assert lm.usage.total_tokens == 15
        assert lm.usage.call_count == 1

        lm._build_response("world", usage=usage)
        assert lm.usage.prompt_tokens == 20
        assert lm.usage.total_tokens == 30
        assert lm.usage.call_count == 2
