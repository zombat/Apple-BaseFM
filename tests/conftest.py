"""Shared fixtures for apple-basefm unit tests.

All tests run on Linux / non-macOS CI. The fake SDK modules injected here
replace the real apple_fm_sdk and mlx_lm so no Apple hardware is needed.
"""
from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake apple_fm_sdk builder
# ---------------------------------------------------------------------------


def _make_fake_fm_sdk() -> types.ModuleType:
    """Build a synthetic apple_fm_sdk module for unit tests.

    The fake SDK exposes the minimal surface that AppleFoundationLM uses:
    SystemLanguageModel, LanguageModelSession, GenerationOptions, generable,
    guide, and Tool.
    """
    fm = types.ModuleType("apple_fm_sdk")

    # --- SystemLanguageModel ---
    class FakeSystemLanguageModel:
        def is_available(self) -> tuple[bool, str]:
            return True, ""

    fm.SystemLanguageModel = FakeSystemLanguageModel  # type: ignore[attr-defined]

    # --- LanguageModelSession ---
    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def respond(self, prompt: str = "", **kwargs: Any) -> str:
            return f"response to: {prompt}"

    fm.LanguageModelSession = FakeSession  # type: ignore[attr-defined]

    # --- GenerationOptions ---
    class FakeGenerationOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    fm.GenerationOptions = FakeGenerationOptions  # type: ignore[attr-defined]

    # --- @generable decorator ---
    def generable(cls: type) -> type:
        cls._is_generable = True
        return cls

    fm.generable = generable  # type: ignore[attr-defined]

    # --- guide() factory ---
    def guide(name: str, **kwargs: Any) -> Any:
        sentinel = MagicMock()
        sentinel._guide_kwargs = kwargs
        return sentinel

    fm.guide = guide  # type: ignore[attr-defined]

    # --- fm.Tool base class ---
    class Tool:
        def call(self, **kwargs: Any) -> Any:
            raise NotImplementedError

    fm.Tool = Tool  # type: ignore[attr-defined]

    # --- GuardrailViolationError ---
    class GuardrailViolationError(Exception):
        pass

    fm.GuardrailViolationError = GuardrailViolationError  # type: ignore[attr-defined]

    return fm


# ---------------------------------------------------------------------------
# Fake mlx_lm builder
# ---------------------------------------------------------------------------


def _make_fake_mlx_lm() -> types.ModuleType:
    """Build a synthetic mlx_lm module for unit tests."""
    mlx_lm = types.ModuleType("mlx_lm")

    class _FakeSelfAttn:
        head_dim = 64

    class _FakeLayer:
        self_attn = _FakeSelfAttn()

    class FakeModel:
        layers = [_FakeLayer()]

        def __call__(self, x: Any, **kwargs: Any) -> MagicMock:
            """Return a fake logits tensor supporting indexing and mx.argmax."""
            logits = MagicMock()
            # logits[0, -1, :] → another MagicMock (consumed by mx.argmax)
            logits.__getitem__ = MagicMock(return_value=MagicMock())
            return logits

    class FakeTokenizer:
        model_max_length = 4096

        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

        def apply_chat_template(
            self,
            messages: list[dict],
            tokenize: bool = False,
            add_generation_prompt: bool = True,
        ) -> str:
            parts = [m.get("content", "") for m in messages]
            return "\n".join(parts)

    def load(model_path: str) -> tuple[FakeModel, FakeTokenizer]:
        return FakeModel(), FakeTokenizer()

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

    def generate(model: Any, tokenizer: Any, prompt: str = "", **kwargs: Any) -> str:
        return f"generated: {prompt[:20]}"

    def stream_generate(
        model: Any, tokenizer: Any, prompt: str = "", max_tokens: int = 10, **kwargs: Any
    ):
        tokens = ["token1 ", "token2 ", "token3"]
        for t in tokens:
            yield FakeResponse(t)

    mlx_lm.load = load  # type: ignore[attr-defined]
    mlx_lm.generate = generate  # type: ignore[attr-defined]
    mlx_lm.stream_generate = stream_generate  # type: ignore[attr-defined]
    mlx_lm.FakeModel = FakeModel  # type: ignore[attr-defined]
    mlx_lm.FakeTokenizer = FakeTokenizer  # type: ignore[attr-defined]

    # sample_utils submodule
    sample_utils = types.ModuleType("mlx_lm.sample_utils")

    def make_sampler(temp: float = 0.0, **kwargs: Any) -> MagicMock:
        s = MagicMock()
        s.temp = temp
        return s

    sample_utils.make_sampler = make_sampler  # type: ignore[attr-defined]
    mlx_lm.sample_utils = sample_utils  # type: ignore[attr-defined]

    # models.cache submodule — provides QuantizedKVCache for _kv/cache_v2.py
    models_cache = types.ModuleType("mlx_lm.models.cache")

    class FakeQuantizedKVCache:
        """Minimal QuantizedKVCache stub for unit tests."""

        step = 256

        def __init__(self, bits: int = 4, group_size: int = 64) -> None:
            self.bits = bits
            self.group_size = group_size
            self._stored_keys: list[Any] = []
            self._stored_values: list[Any] = []
            self.offset = 0

        def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
            self._stored_keys.append(keys)
            self._stored_values.append(values)
            self.offset += 1
            # Return the most-recent K/V as a stand-in for the full history.
            return keys, values

        @property
        def state(self) -> list[Any]:
            return self._stored_keys + self._stored_values

        @property
        def nbytes(self) -> int:
            return 0

    models_cache.QuantizedKVCache = FakeQuantizedKVCache  # type: ignore[attr-defined]
    models = types.ModuleType("mlx_lm.models")
    models.cache = models_cache  # type: ignore[attr-defined]
    mlx_lm.models = models  # type: ignore[attr-defined]

    return mlx_lm


# ---------------------------------------------------------------------------
# Fake mlx / mlx.core builder
# ---------------------------------------------------------------------------


def _make_fake_mlx_core() -> tuple[types.ModuleType, types.ModuleType]:
    """Build synthetic mlx and mlx.core modules for unit tests.

    Provides the minimal surface used by the raw-forward-pass pattern:
    ``mx.array()``, ``mx.eval()``, and ``mx.argmax()``.
    """
    mlx_mod = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")

    class FakeMXArray:
        def __init__(self, data: Any = None) -> None:
            self._data = data

        def __getitem__(self, idx: Any) -> "FakeMXArray":
            return FakeMXArray()

        def item(self) -> int:
            return 42  # deterministic fake token id

    def array(data: Any) -> FakeMXArray:  # noqa: ANN001
        return FakeMXArray(data)

    def eval(*args: Any) -> None:  # noqa: ANN002
        pass

    def argmax(arr: Any, **kwargs: Any) -> FakeMXArray:
        return FakeMXArray(42)

    core.array = array  # type: ignore[attr-defined]
    core.eval = eval  # type: ignore[attr-defined]
    core.argmax = argmax  # type: ignore[attr-defined]
    mlx_mod.core = core  # type: ignore[attr-defined]

    return mlx_mod, core


# ---------------------------------------------------------------------------
# Autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fake_apple_fm_sdk(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a synthetic apple_fm_sdk into sys.modules for every test."""
    import sys

    fm = _make_fake_fm_sdk()
    monkeypatch.setitem(sys.modules, "apple_fm_sdk", fm)
    return fm


@pytest.fixture(autouse=True)
def fake_mlx_modules(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject synthetic mlx_lm (and submodules) into sys.modules for every test."""
    import sys

    mlx = _make_fake_mlx_lm()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx.sample_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx.models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", mlx.models.cache)
    return mlx


@pytest.fixture(autouse=True)
def fake_mlx_core(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject synthetic mlx and mlx.core into sys.modules for every test."""
    import sys

    mlx_mod, core = _make_fake_mlx_core()
    monkeypatch.setitem(sys.modules, "mlx", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


# ---------------------------------------------------------------------------
# MCP fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mcp_client():
    """Async fixture that yields a connected FastMCP client backed by MCP_SERVER.

    The connection is fully in-process — no subprocess or network socket is
    opened. Use it to call tools directly::

        result = await mcp_client.call_tool("add", {"a": 1, "b": 2})
        assert result.data == 3
    """
    from fastmcp.client import Client

    from tests.tools import MCP_SERVER

    async with Client(MCP_SERVER) as client:
        yield client
