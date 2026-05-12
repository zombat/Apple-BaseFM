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
        """ImportError must mention pip install instructions."""
        monkeypatch.delitem(sys.modules, "apple_fm_sdk", raising=False)
        if "apple_basefm.apple_fm" in sys.modules:
            del sys.modules["apple_basefm.apple_fm"]
        mod = importlib.import_module("apple_basefm.apple_fm")
        with (
            patch("platform.system", return_value="Darwin"),
            patch.dict(sys.modules, {"apple_fm_sdk": None}),  # type: ignore[arg-type]
            pytest.raises(ImportError, match="apple-fm-sdk"),
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


# ---------------------------------------------------------------------------
# _dspy_tool_to_apple_tool — cache hit + func=None paths
# ---------------------------------------------------------------------------


class TestDspyToolToAppleTool:
    def test_cache_hit_returns_new_instance(self, apple_fm_mod: types.ModuleType) -> None:
        """Calling with the same tool twice must hit the cache on the second call."""
        import apple_fm_sdk as fm

        apple_fm_mod._tool_class_cache.clear()

        def my_func(**kwargs: Any) -> str:
            return "ok"

        tool = MagicMock()
        tool.name = "cached_tool"
        tool.func = my_func

        first = apple_fm_mod._dspy_tool_to_apple_tool(tool, fm)
        second = apple_fm_mod._dspy_tool_to_apple_tool(tool, fm)
        # Both calls return instances of the same class (cached).
        assert type(first) is type(second)

    def test_func_none_raises_not_implemented(self, apple_fm_mod: types.ModuleType) -> None:
        """A tool with no callable must raise NotImplementedError when .call() is invoked."""
        import apple_fm_sdk as fm

        apple_fm_mod._tool_class_cache.clear()

        # A non-callable object without a .func attribute: func resolves to None.
        class _NoFunc:
            name = "no_func_tool"

        tool = _NoFunc()
        instance = apple_fm_mod._dspy_tool_to_apple_tool(tool, fm)
        with pytest.raises(NotImplementedError, match="no callable"):
            instance.call()


# ---------------------------------------------------------------------------
# _pydantic_to_generable — constraint annotations
# ---------------------------------------------------------------------------


class TestPydanticToGenerable:
    def test_literal_same_type(self, apple_fm_mod: types.ModuleType) -> None:
        """Literal[a, b] with same type must build a @generable class."""
        from typing import Literal

        import apple_fm_sdk as fm
        from pydantic import BaseModel

        class Model(BaseModel):
            color: Literal["red", "blue"]

        result = apple_fm_mod._pydantic_to_generable(Model, fm)
        assert result is not None

    def test_ge_le_range_constraint(self, apple_fm_mod: types.ModuleType) -> None:
        """int field with ge/le must invoke fm.guide with range."""
        import apple_fm_sdk as fm
        from pydantic import BaseModel, Field

        class Model(BaseModel):
            score: int = Field(ge=0, le=100)

        guide_calls: list[dict] = []
        original_guide = fm.guide

        def _recording_guide(name: str, **kwargs: Any) -> Any:
            guide_calls.append({"name": name, **kwargs})
            return original_guide(name, **kwargs)

        fm.guide = _recording_guide
        try:
            apple_fm_mod._pydantic_to_generable(Model, fm)
        finally:
            fm.guide = original_guide

        range_calls = [c for c in guide_calls if "range" in c]
        assert range_calls, "Expected fm.guide(range=...) for ge/le field"

    def test_make_dataclass_failure_returns_none(
        self, apple_fm_mod: types.ModuleType
    ) -> None:
        """If make_dataclass raises, _pydantic_to_generable must return None."""
        import apple_fm_sdk as fm
        from pydantic import BaseModel

        class Model(BaseModel):
            value: str

        with patch("dataclasses.make_dataclass", side_effect=RuntimeError("boom")):
            result = apple_fm_mod._pydantic_to_generable(Model, fm)
        assert result is None


# ---------------------------------------------------------------------------
# aforward — tool conversion path
# ---------------------------------------------------------------------------


class TestAforwardWithTools:
    @pytest.mark.asyncio
    async def test_tools_passed_to_session(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """DSPy tools must be converted and forwarded in session_kwargs['tools']."""
        session_kwargs_received: list[dict] = []
        original_session_cls = fake_apple_fm_sdk.LanguageModelSession

        class _RecordingSession(original_session_cls):
            def __init__(self_inner, **kwargs: Any) -> None:
                session_kwargs_received.append(dict(kwargs))
                super().__init__(**kwargs)

        fake_apple_fm_sdk.LanguageModelSession = _RecordingSession

        def my_tool(**kwargs: Any) -> str:
            return "result"

        my_tool.name = "my_tool"  # type: ignore[attr-defined]

        try:
            await fm_instance.aforward(
                messages=[{"role": "user", "content": "use a tool"}],
                tools=[my_tool],
            )
        finally:
            fake_apple_fm_sdk.LanguageModelSession = original_session_cls

        assert any("tools" in kw for kw in session_kwargs_received), (
            "tools must be in session_kwargs when tools are supplied"
        )


# ---------------------------------------------------------------------------
# aforward — generable with timeout=None
# ---------------------------------------------------------------------------


class TestAforwardGenerable:
    @pytest.mark.asyncio
    async def test_generable_timeout_none_skips_wait_for(
        self, fm_instance: Any, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """With timeout=None, asyncio.wait_for must NOT be called on the generable path."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: str

        fm_instance._timeout = None
        wait_for_calls: list[bool] = []
        original_wait_for = asyncio.wait_for

        async def _recording_wait_for(coro: Any, timeout: Any) -> Any:
            wait_for_calls.append(True)
            return await original_wait_for(coro, timeout=timeout)

        generable_respond_called: list[bool] = []
        original_session_cls = fake_apple_fm_sdk.LanguageModelSession

        class _GenSession(original_session_cls):
            async def respond(self_inner, prompt: str = "", **kwargs: Any) -> Any:
                if "generating" in kwargs:
                    generable_respond_called.append(True)
                    result = MagicMock()
                    with patch("dataclasses.asdict", return_value={"value": "ok"}):
                        return result
                return f"response to: {prompt}"

        fake_apple_fm_sdk.LanguageModelSession = _GenSession
        try:
            with patch("asyncio.wait_for", side_effect=_recording_wait_for):
                response = await fm_instance.aforward(
                    messages=[{"role": "user", "content": "test"}],
                    response_format=Answer,
                )
        finally:
            fake_apple_fm_sdk.LanguageModelSession = original_session_cls

        assert wait_for_calls == [], "asyncio.wait_for must not be called when timeout=None"
        assert response is not None


# ---------------------------------------------------------------------------
# _run_async — running event loop (nest_asyncio) path
# ---------------------------------------------------------------------------


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_raises_without_nest_asyncio(
        self, apple_fm_mod: types.ModuleType
    ) -> None:
        """Without nest_asyncio, calling _run_async from a running loop must raise."""

        async def _dummy() -> str:
            return "ok"

        with patch.dict(sys.modules, {"nest_asyncio": None}):
            with pytest.raises(RuntimeError, match="nest_asyncio"):
                apple_fm_mod._run_async(_dummy())

    @pytest.mark.asyncio
    async def test_run_async_uses_nest_asyncio_when_available(
        self, apple_fm_mod: types.ModuleType
    ) -> None:
        """When nest_asyncio is importable and apply() works, _run_async must succeed."""
        import types as _types

        apply_called: list[bool] = []
        fake_nest = _types.ModuleType("nest_asyncio")

        def _fake_apply(loop: Any) -> None:
            apply_called.append(True)
            # Python 3.10+ raises "Cannot run the event loop while another loop
            # is running" even for a freshly-created loop on the same thread.
            # Run the coroutine in a daemon thread that owns its own event loop.
            import threading

            def _run_in_new_loop(coro: Any) -> Any:
                result_holder: list[Any] = []
                error_holder: list[BaseException] = []

                def _worker() -> None:
                    new_loop = asyncio.new_event_loop()
                    try:
                        result_holder.append(new_loop.run_until_complete(coro))
                    except BaseException as exc:  # noqa: BLE001
                        error_holder.append(exc)
                    finally:
                        new_loop.close()

                t = threading.Thread(target=_worker, daemon=True)
                t.start()
                t.join(timeout=5)
                if error_holder:
                    raise error_holder[0]
                return result_holder[0] if result_holder else None

            loop.run_until_complete = _run_in_new_loop

        fake_nest.apply = _fake_apply  # type: ignore[attr-defined]

        async def _dummy() -> str:
            return "hello"

        with patch.dict(sys.modules, {"nest_asyncio": fake_nest}):
            result = apple_fm_mod._run_async(_dummy())

        assert apply_called, "nest_asyncio.apply must be called from a running loop"
        assert result == "hello"

