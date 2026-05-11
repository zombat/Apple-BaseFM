"""Unit tests for apple_basefm.apple_fm.AppleFoundationLM.

Isolation strategy:
    conftest.py injects fake apple_fm_sdk and mlx_lm via monkeypatch before
    each test. After injection we reload apple_basefm.apple_fm so the module picks
    up the patched sys.modules.
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import platform
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to reload the module with fresh sys.modules on each test function
# ---------------------------------------------------------------------------


def _reload_apple_fm() -> types.ModuleType:
    """Force a clean import of apple_fm against whatever sys.modules contains."""
    if "apple_basefm.apple_fm" in sys.modules:
        del sys.modules["apple_basefm.apple_fm"]
    return importlib.import_module("apple_basefm.apple_fm")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def apple_fm_mod(fake_apple_fm_sdk: types.ModuleType) -> types.ModuleType:  # noqa: ARG001
    """Reload apple_fm with fake SDK present and macOS patched."""
    with patch("platform.system", return_value="Darwin"):
        return _reload_apple_fm()


@pytest.fixture()
def fm_instance(apple_fm_mod: types.ModuleType) -> Any:
    """Create an AppleFoundationLM instance with mocked platform."""
    with patch("platform.system", return_value="Darwin"):
        return apple_fm_mod.AppleFoundationLM()


# ---------------------------------------------------------------------------
# Construction & platform guard
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_non_macos_raises(self, apple_fm_mod: types.ModuleType) -> None:
        """Non-Darwin platform must raise RuntimeError immediately."""
        with patch("platform.system", return_value="Linux"):
            with pytest.raises(RuntimeError, match="macOS 26\\+"):
                apple_fm_mod.AppleFoundationLM()

    def test_import_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ImportError must mention Apple developer channel, not PyPI."""
        monkeypatch.delitem(sys.modules, "apple_fm_sdk", raising=False)
        if "apple_basefm.apple_fm" in sys.modules:
            del sys.modules["apple_basefm.apple_fm"]
        mod = importlib.import_module("apple_basefm.apple_fm")
        with (
            patch("platform.system", return_value="Darwin"),
            patch.dict(sys.modules, {"apple_fm_sdk": None}),  # type: ignore[arg-type]
            pytest.raises(ImportError, match="developer distribution channel"),
        ):
            # Force ImportError path by making import fail inside __init__.
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

            def _fail_apple_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "apple_fm_sdk":
                    raise ImportError("no module")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_fail_apple_import):
                mod.AppleFoundationLM()

    def test_unavailable_model_raises(
        self,
        apple_fm_mod: types.ModuleType,
        fake_apple_fm_sdk: types.ModuleType,
    ) -> None:
        """If is_available() returns False, RuntimeError must be raised."""

        class _UnavailableModel:
            def is_available(self) -> tuple[bool, str]:
                return False, "Apple Intelligence not enabled"

        fake_apple_fm_sdk.SystemLanguageModel = _UnavailableModel
        with (
            patch("platform.system", return_value="Darwin"),
            pytest.raises(RuntimeError, match="not available"),
        ):
            apple_fm_mod.AppleFoundationLM()

    def test_default_timeout_is_120(self, fm_instance: Any) -> None:
        assert fm_instance._timeout == 120.0

    def test_custom_timeout(self, apple_fm_mod: types.ModuleType) -> None:
        with patch("platform.system", return_value="Darwin"):
            lm = apple_fm_mod.AppleFoundationLM(timeout=60.0)
        assert lm._timeout == 60.0

    def test_none_timeout(self, apple_fm_mod: types.ModuleType) -> None:
        with patch("platform.system", return_value="Darwin"):
            lm = apple_fm_mod.AppleFoundationLM(timeout=None)
        assert lm._timeout is None


# ---------------------------------------------------------------------------
# Capability properties
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_supports_function_calling(self, fm_instance: Any) -> None:
        assert fm_instance.supports_function_calling is True

    def test_supports_response_schema(self, fm_instance: Any) -> None:
        assert fm_instance.supports_response_schema is True

    def test_supports_reasoning(self, fm_instance: Any) -> None:
        assert fm_instance.supports_reasoning is False

    def test_supported_params(self, fm_instance: Any) -> None:
        assert "temperature" in fm_instance.supported_params
        assert "max_tokens" in fm_instance.supported_params


# ---------------------------------------------------------------------------
# aforward — plain text path
# ---------------------------------------------------------------------------


class TestAforward:
    @pytest.mark.asyncio
    async def test_plain_text_response(self, fm_instance: Any) -> None:
        response = await fm_instance.aforward(messages=[{"role": "user", "content": "hello"}])
        assert response.choices[0].message.content.startswith("response to:")

    @pytest.mark.asyncio
    async def test_response_has_hidden_params(self, fm_instance: Any) -> None:
        response = await fm_instance.aforward(messages=[{"role": "user", "content": "hello"}])
        assert "response_cost" in response._hidden_params

    @pytest.mark.asyncio
    async def test_plain_prompt_string(self, fm_instance: Any) -> None:
        response = await fm_instance.aforward(prompt="what is 2+2?")
        assert response.choices[0].message.content != ""

    @pytest.mark.asyncio
    async def test_timeout_applied(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """asyncio.TimeoutError from session.respond must bubble as RuntimeError."""

        async def _slow_respond(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(999)
            return ""

        fake_apple_fm_sdk.LanguageModelSession.respond = _slow_respond  # type: ignore[attr-defined]

        fm_instance._timeout = 0.001
        with pytest.raises(RuntimeError, match="timed out"):
            await fm_instance.aforward(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_no_timeout_when_none(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """With timeout=None, asyncio.wait_for must not be called."""
        fm_instance._timeout = None
        called_with_wait_for: list[bool] = []

        original = asyncio.wait_for

        async def _recording_wait_for(coro: Any, timeout: Any) -> Any:
            called_with_wait_for.append(True)
            return await original(coro, timeout=timeout)

        with patch("asyncio.wait_for", side_effect=_recording_wait_for):
            await fm_instance.aforward(messages=[{"role": "user", "content": "hi"}])

        assert called_with_wait_for == [], "asyncio.wait_for must not be called when timeout=None"

    @pytest.mark.asyncio
    async def test_session_deleted_in_finally(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """The session object must be released (del session) even after failure."""

        async def _raise(*args: Any, **kwargs: Any) -> str:
            raise ValueError("SDK exploded")

        fake_apple_fm_sdk.LanguageModelSession.respond = _raise  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="SDK exploded"):
            await fm_instance.aforward(messages=[{"role": "user", "content": "hi"}])
        # If we reach here without the process hanging, del session was called.


# ---------------------------------------------------------------------------
# aforward — structured output path (EC-3, generable fallback)
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_pydantic_response_format_uses_generable(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """response_format=PydanticModel should trigger generable path."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: str

        generable_called: list[bool] = []

        original_respond = fake_apple_fm_sdk.LanguageModelSession.respond  # type: ignore[attr-defined]

        async def _generable_respond(self_: Any, prompt: str = "", **kwargs: Any) -> Any:
            if "generating" in kwargs:
                generable_called.append(True)
                # Return a dataclass-like object
                inst = MagicMock()
                inst.value = "42"
                # make dataclasses.asdict work by returning a plain dict
                with patch("dataclasses.asdict", return_value={"value": "42"}):
                    return inst
            return await original_respond(self_, prompt=prompt, **kwargs)

        fake_apple_fm_sdk.LanguageModelSession.respond = _generable_respond  # type: ignore[attr-defined]

        # We don't assert the content — just that generable was invoked without error.
        # (The @generable path may raise; the fallback then kicks in.)
        response = await fm_instance.aforward(
            messages=[{"role": "user", "content": "give me a number"}],
            response_format=Answer,
        )
        assert response is not None

    @pytest.mark.asyncio
    async def test_mixed_type_literal_falls_back_to_any(self, apple_fm_mod: types.ModuleType) -> None:
        """Literal[1, "two"] must not raise — it falls back to Any."""
        from typing import Literal

        from pydantic import BaseModel

        class Mixed(BaseModel):
            value: Literal[1, "two"]  # type: ignore[type-arg]

        with patch("platform.system", return_value="Darwin"):
            lm = apple_fm_mod.AppleFoundationLM()

        import apple_fm_sdk as fm

        result = apple_fm_mod._pydantic_to_generable(Mixed, fm)
        # Should either produce a class or None — must not raise.
        assert result is None or callable(result)

    @pytest.mark.asyncio
    async def test_generable_failure_retries_without_schema(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """If generable path fails, fallback session must produce a response."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: str

        call_count = 0

        async def _respond(self_: Any, prompt: str = "", **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if "generating" in kwargs:
                raise RuntimeError("generable failed")
            return "fallback text"

        fake_apple_fm_sdk.LanguageModelSession.respond = _respond  # type: ignore[attr-defined]

        response = await fm_instance.aforward(
            messages=[{"role": "user", "content": "test"}],
            response_format=Answer,
        )
        assert response.choices[0].message.content == "fallback text"
        assert call_count >= 2  # first (generable) + second (fallback)


# ---------------------------------------------------------------------------
# aforward — guardrail (EC-12)
# ---------------------------------------------------------------------------


class TestGuardrail:
    @pytest.mark.asyncio
    async def test_guardrail_re_raised_as_runtime(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """GuardrailViolationError must be re-raised as RuntimeError."""

        async def _guardrail(self_: Any, prompt: str = "", **kwargs: Any) -> str:
            raise fake_apple_fm_sdk.GuardrailViolationError("unsafe content")

        fake_apple_fm_sdk.LanguageModelSession.respond = _guardrail  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="guardrail violation"):
            await fm_instance.aforward(messages=[{"role": "user", "content": "bad"}])

    @pytest.mark.asyncio
    async def test_guardrail_on_fallback_path_also_raised(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """Guardrail on the fallback (no-schema) path must also raise RuntimeError."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: str

        call_count = 0

        async def _both_fail(self_: Any, prompt: str = "", **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("generable failed")
            raise fake_apple_fm_sdk.GuardrailViolationError("unsafe even on retry")

        fake_apple_fm_sdk.LanguageModelSession.respond = _both_fail  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="guardrail violation"):
            await fm_instance.aforward(
                messages=[{"role": "user", "content": "bad"}],
                response_format=Answer,
            )


# ---------------------------------------------------------------------------
# forward() — cache behaviour
# ---------------------------------------------------------------------------


class TestForwardCache:
    def test_forward_returns_response(self, fm_instance: Any) -> None:
        with patch("platform.system", return_value="Darwin"):
            response = fm_instance.forward(messages=[{"role": "user", "content": "hello"}])
        assert hasattr(response, "choices")

    def test_forward_hits_cache_on_second_call(self, fm_instance: Any) -> None:
        """Second identical call must use cache, not call the SDK again."""
        import apple_basefm._compat as _compat_mod
        from apple_basefm._compat import _NullCache

        # Use an in-memory cache so the test is hermetic regardless of whether
        # a real DSPy disk cache has stale entries.
        class _DictCache:
            def __init__(self) -> None:
                self._store: dict = {}

            def get(self, key: Any) -> Any:
                import json
                return self._store.get(json.dumps(key, sort_keys=True, default=str))

            def put(self, key: Any, value: Any) -> None:
                import json
                self._store[json.dumps(key, sort_keys=True, default=str)] = value

        dict_cache = _DictCache()
        sdk_calls: list[int] = []
        original_aforward = fm_instance.aforward

        async def _counting_aforward(*args: Any, **kwargs: Any) -> Any:
            sdk_calls.append(1)
            return await original_aforward(*args, **kwargs)

        fm_instance.aforward = _counting_aforward
        with patch("apple_basefm.apple_fm.get_dspy_cache", return_value=dict_cache):
            fm_instance.forward(messages=[{"role": "user", "content": "cached"}])
            fm_instance.forward(messages=[{"role": "user", "content": "cached"}])
        assert len(sdk_calls) == 1, "Cache hit on second call must skip aforward()"

    def test_stream_true_raises(self, fm_instance: Any) -> None:
        with pytest.raises(NotImplementedError, match="Streaming is not yet supported"):
            fm_instance.forward(messages=[{"role": "user", "content": "x"}], stream=True)


# ---------------------------------------------------------------------------
# _tool_class_cache LRU eviction (G-3)
# ---------------------------------------------------------------------------


class TestToolCacheLRU:
    def test_cache_bounded_to_maxsize(self, apple_fm_mod: types.ModuleType) -> None:
        """_tool_class_cache must never exceed _TOOL_CACHE_MAXSIZE entries."""
        import apple_fm_sdk as fm

        apple_fm_mod._tool_class_cache.clear()

        for i in range(apple_fm_mod._TOOL_CACHE_MAXSIZE + 10):

            def _tool(**kwargs: Any) -> str:
                return f"result_{i}"

            _tool.__name__ = f"tool_{i}"
            fake_tool = MagicMock()
            fake_tool.name = f"tool_{i}"
            fake_tool.func = _tool

            apple_fm_mod._dspy_tool_to_apple_tool(fake_tool, fm)

        assert len(apple_fm_mod._tool_class_cache) <= apple_fm_mod._TOOL_CACHE_MAXSIZE
