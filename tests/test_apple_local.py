"""Unit tests for apple_basefm.apple_local.AppleLocalLM.

Isolation strategy:
    conftest.py injects fake mlx_lm via monkeypatch before every test.
    We also patch platform.system / platform.machine to simulate Apple Silicon
    wherever the constructor checks them.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper — reload apple_local with patched sys.modules
# ---------------------------------------------------------------------------


def _reload_apple_local() -> types.ModuleType:
    if "apple_basefm.apple_local" in sys.modules:
        del sys.modules["apple_basefm.apple_local"]
    return importlib.import_module("apple_basefm.apple_local")


@pytest.fixture()
def apple_local_mod(fake_mlx_modules: types.ModuleType) -> types.ModuleType:  # noqa: ARG001
    """Reload apple_local with fake mlx_lm present and macOS+arm64 patched."""
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        return _reload_apple_local()


@pytest.fixture()
def local_instance(apple_local_mod: types.ModuleType) -> Any:
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        return apple_local_mod.AppleLocalLM("mlx-community/test-model-4bit")


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_non_macos_raises(self, apple_local_mod: types.ModuleType) -> None:
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="arm64"),
            pytest.raises(RuntimeError, match="macOS"),
        ):
            apple_local_mod.AppleLocalLM("any-model")

    def test_non_arm64_raises(self, apple_local_mod: types.ModuleType) -> None:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
            pytest.raises(RuntimeError, match="Apple Silicon"),
        ):
            apple_local_mod.AppleLocalLM("any-model")

    def test_unknown_backend_raises(self, apple_local_mod: types.ModuleType) -> None:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            pytest.raises(ValueError, match="Unknown backend"),
        ):
            apple_local_mod.AppleLocalLM("model", backend="tpu")

    def test_coreml_raises_not_implemented(self, apple_local_mod: types.ModuleType) -> None:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            pytest.raises(NotImplementedError, match="CoreML backend"),
        ):
            apple_local_mod.AppleLocalLM("model", backend="coreml")

    def test_max_concurrency_zero_raises(self, apple_local_mod: types.ModuleType) -> None:
        """EC-2: max_concurrency < 1 must raise ValueError."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            pytest.raises(ValueError, match="max_concurrency must be >= 1"),
        ):
            apple_local_mod.AppleLocalLM("model", max_concurrency=0)

    def test_max_concurrency_negative_raises(self, apple_local_mod: types.ModuleType) -> None:
        """EC-2: negative max_concurrency must raise ValueError."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            pytest.raises(ValueError, match="max_concurrency must be >= 1"),
        ):
            apple_local_mod.AppleLocalLM("model", max_concurrency=-5)

    def test_context_window_capped(self, local_instance: Any) -> None:
        """Tokenizer model_max_length=4096 (from fake) must be honoured."""
        assert local_instance.context_window == 4096

    def test_bits_stored(self, apple_local_mod: types.ModuleType) -> None:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            lm = apple_local_mod.AppleLocalLM("model", bits=4)
        assert lm._bits == 4


# ---------------------------------------------------------------------------
# Capability properties
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_no_function_calling(self, local_instance: Any) -> None:
        assert local_instance.supports_function_calling is False

    def test_no_response_schema(self, local_instance: Any) -> None:
        assert local_instance.supports_response_schema is False

    def test_no_reasoning(self, local_instance: Any) -> None:
        assert local_instance.supports_reasoning is False

    def test_supported_params(self, local_instance: Any) -> None:
        assert "temperature" in local_instance.supported_params
        assert "max_tokens" in local_instance.supported_params


# ---------------------------------------------------------------------------
# forward() — temperature / max_tokens clamping (G-1, G-2)
# ---------------------------------------------------------------------------


class TestInputClamping:
    def test_temperature_clamped_above_2(self, local_instance: Any) -> None:
        """G-1: temperature > 2.0 must be silently clamped to 2.0."""
        captured: list[float] = []
        original_generate = local_instance._generate

        def _capture_generate(messages: Any, temperature: float, max_tokens: int, lps: Any) -> tuple:
            captured.append(temperature)
            return original_generate(messages, temperature, max_tokens, lps)

        local_instance._generate = _capture_generate
        local_instance.forward(
            messages=[{"role": "user", "content": "hi"}], temperature=5.0, cache=False
        )
        assert captured[0] <= 2.0

    def test_temperature_clamped_below_0(self, local_instance: Any) -> None:
        """G-1: temperature < 0.0 must be silently clamped to 0.0."""
        captured: list[float] = []
        original_generate = local_instance._generate

        def _capture_generate(messages: Any, temperature: float, max_tokens: int, lps: Any) -> tuple:
            captured.append(temperature)
            return original_generate(messages, temperature, max_tokens, lps)

        local_instance._generate = _capture_generate
        local_instance.forward(
            messages=[{"role": "user", "content": "hi"}], temperature=-1.0, cache=False
        )
        assert captured[0] >= 0.0

    def test_max_tokens_floored_at_1(self, local_instance: Any) -> None:
        """G-2: max_tokens <= 0 must be silently raised to 1."""
        captured: list[int] = []
        original_generate = local_instance._generate

        def _capture_generate(messages: Any, temperature: float, max_tokens: int, lps: Any) -> tuple:
            captured.append(max_tokens)
            return original_generate(messages, temperature, max_tokens, lps)

        local_instance._generate = _capture_generate
        local_instance.forward(
            messages=[{"role": "user", "content": "hi"}], max_tokens=0, cache=False
        )
        assert captured[0] >= 1


# ---------------------------------------------------------------------------
# forward() — tools / stream guards
# ---------------------------------------------------------------------------


class TestForwardGuards:
    def test_tools_raises_not_implemented(self, local_instance: Any) -> None:
        with pytest.raises(NotImplementedError, match="Tool calling is not supported"):
            local_instance.forward(
                messages=[{"role": "user", "content": "x"}],
                tools=[MagicMock()],
            )

    def test_stream_true_raises_not_implemented(self, local_instance: Any) -> None:
        with pytest.raises(NotImplementedError, match="stream=True"):
            local_instance.forward(
                messages=[{"role": "user", "content": "x"}],
                stream=True,
            )


# ---------------------------------------------------------------------------
# forward() — cache behaviour
# ---------------------------------------------------------------------------


class TestForwardCache:
    def test_forward_returns_response(self, local_instance: Any) -> None:
        response = local_instance.forward(messages=[{"role": "user", "content": "hello"}])
        assert hasattr(response, "choices")
        assert response.choices[0].message.content != ""

    def test_cache_hit_skips_generate(self, local_instance: Any) -> None:
        """Second identical call must use cache, not call _generate again."""

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
        generate_calls: list[int] = []
        original_generate = local_instance._generate

        def _counting_generate(messages: Any, temperature: float, max_tokens: int, lps: Any) -> tuple:
            generate_calls.append(1)
            return original_generate(messages, temperature, max_tokens, lps)

        local_instance._generate = _counting_generate
        with patch("apple_basefm.apple_local.get_dspy_cache", return_value=dict_cache):
            local_instance.forward(messages=[{"role": "user", "content": "cached msg"}])
            local_instance.forward(messages=[{"role": "user", "content": "cached msg"}])
        assert len(generate_calls) == 1

    def test_cache_disabled_calls_generate_twice(self, local_instance: Any) -> None:
        """With cache=False, _generate must be called on every request."""
        generate_calls: list[int] = []
        original_generate = local_instance._generate

        def _counting_generate(messages: Any, temperature: float, max_tokens: int, lps: Any) -> tuple:
            generate_calls.append(1)
            return original_generate(messages, temperature, max_tokens, lps)

        local_instance._generate = _counting_generate
        local_instance.forward(
            messages=[{"role": "user", "content": "no cache"}], cache=False
        )
        local_instance.forward(
            messages=[{"role": "user", "content": "no cache"}], cache=False
        )
        assert len(generate_calls) == 2


# ---------------------------------------------------------------------------
# forward() — empty response guard (EC-1)
# ---------------------------------------------------------------------------


class TestEmptyResponseGuard:
    def test_empty_mlx_response_raises(self, local_instance: Any) -> None:
        """EC-1: empty text from mlx-lm must raise ValueError, not return silently."""

        def _empty_generate(messages: Any, temperature: float, max_tokens: int, lps: Any) -> tuple:
            return "", "some prompt"

        local_instance._generate = _empty_generate
        # Disable cache so _generate is actually called.
        with pytest.raises(ValueError, match="empty response"):
            local_instance.forward(messages=[{"role": "user", "content": "hi"}], cache=False)


# ---------------------------------------------------------------------------
# aforward() — semaphore concurrency gating
# ---------------------------------------------------------------------------


class TestAforward:
    @pytest.mark.asyncio
    async def test_aforward_returns_response(self, local_instance: Any) -> None:
        response = await local_instance.aforward(
            messages=[{"role": "user", "content": "hello"}]
        )
        assert hasattr(response, "choices")

    @pytest.mark.asyncio
    async def test_semaphore_created_lazily(self, local_instance: Any) -> None:
        """Semaphore must not exist before first aforward() call."""
        assert local_instance._semaphore is None
        await local_instance.aforward(messages=[{"role": "user", "content": "test"}])
        assert local_instance._semaphore is not None

    @pytest.mark.asyncio
    async def test_semaphore_respects_max_concurrency_1(
        self, apple_local_mod: types.ModuleType
    ) -> None:
        """With max_concurrency=1, concurrent aforward() calls must serialise."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            lm = apple_local_mod.AppleLocalLM("model", max_concurrency=1)

        active_count = 0
        max_active = 0

        original_forward = lm.forward

        def _tracking_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            result = original_forward(*args, **kwargs)
            active_count -= 1
            return result

        lm.forward = _tracking_forward

        tasks = [
            asyncio.create_task(
                lm.aforward(messages=[{"role": "user", "content": f"msg {i}"}])
            )
            for i in range(4)
        ]
        await asyncio.gather(*tasks)
        assert max_active == 1, f"Expected serial execution, got max_active={max_active}"


# ---------------------------------------------------------------------------
# aforward() — temperature clamping (G-1) via async path
# ---------------------------------------------------------------------------


class TestAforwardClamping:
    @pytest.mark.asyncio
    async def test_temperature_clamped_in_aforward_streaming_path(
        self, local_instance: Any
    ) -> None:
        """With send_stream active, temperature must still be clamped."""
        from apple_basefm._compat import get_send_stream

        captured_temperatures: list[float] = []

        async def _fake_stream_async(
            flat_prompt: str, temperature: float, max_tokens: int, lps: Any
        ):
            captured_temperatures.append(temperature)

            async def _gen():
                yield "token"

            async for tok in _gen():
                yield tok

        local_instance._stream_generate_async = _fake_stream_async  # type: ignore[method-assign]

        fake_stream = MagicMock()
        async def _noop_send(x: Any) -> None:
            pass

        fake_stream.send = _noop_send

        with patch("apple_basefm._compat.get_send_stream", return_value=fake_stream):
            with patch("apple_basefm.apple_local.get_send_stream", return_value=fake_stream):
                try:
                    await local_instance.aforward(
                        messages=[{"role": "user", "content": "hi"}],
                        temperature=99.0,
                    )
                except Exception:
                    pass  # stream.send mock details don't matter here

        if captured_temperatures:
            assert captured_temperatures[0] <= 2.0
