"""Tests for raw mlx-lm forward pass without AppleLocalLM.

Covers the usage pattern from examples/apple_on_device_lm.py (Pattern 21):
loading a model with mlx_lm.load(), performing a raw forward pass to obtain
logits, and running text generation with mlx_lm.generate().

All tests run under the autouse fake_mlx_modules / fake_mlx_core fixtures
injected by conftest.py — no Apple hardware or real MLX installation needed.
"""
from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# mlx_lm.load()
# ---------------------------------------------------------------------------


class TestRawMLXLoad:
    """mlx_lm.load() returns a usable (model, tokenizer) pair."""

    def test_load_returns_model_and_tokenizer(self, fake_mlx_modules: types.ModuleType) -> None:
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        assert model is not None
        assert tokenizer is not None

    def test_tokenizer_encode_returns_list(self, fake_mlx_modules: types.ModuleType) -> None:
        import mlx_lm

        _, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        token_ids = tokenizer.encode("The capital of France is")
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

    def test_tokenizer_encode_non_empty_input(self, fake_mlx_modules: types.ModuleType) -> None:
        import mlx_lm

        _, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        token_ids = tokenizer.encode("hello world foo bar")
        # Fake tokenizer splits on whitespace → 4 tokens
        assert len(token_ids) == 4

    def test_tokenizer_apply_chat_template_returns_string(
        self, fake_mlx_modules: types.ModuleType
    ) -> None:
        import mlx_lm

        _, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        messages = [{"role": "user", "content": "Name three planets."}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(prompt, str)
        assert "Name three planets." in prompt

    def test_tokenizer_apply_chat_template_includes_all_messages(
        self, fake_mlx_modules: types.ModuleType
    ) -> None:
        import mlx_lm

        _, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is MLX?"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "You are helpful." in prompt
        assert "What is MLX?" in prompt


# ---------------------------------------------------------------------------
# Raw forward pass: model(input_ids) → logits
# ---------------------------------------------------------------------------


class TestRawMLXForwardPass:
    """model() forward call returns logits that support indexing."""

    def test_model_callable(
        self, fake_mlx_modules: types.ModuleType, fake_mlx_core: types.ModuleType
    ) -> None:
        import mlx.core as mx
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        token_ids = tokenizer.encode("The capital of France is")
        input_ids = mx.array([token_ids])
        logits = model(input_ids)
        assert logits is not None

    def test_logits_indexable(
        self, fake_mlx_modules: types.ModuleType, fake_mlx_core: types.ModuleType
    ) -> None:
        import mlx.core as mx
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        token_ids = tokenizer.encode("The capital of France is")
        input_ids = mx.array([token_ids])
        logits = model(input_ids)
        # Simulate: next_token_logits = logits[0, -1, :]
        next_token_logits = logits[0, -1, :]
        assert next_token_logits is not None

    def test_argmax_returns_integer_token_id(
        self, fake_mlx_modules: types.ModuleType, fake_mlx_core: types.ModuleType
    ) -> None:
        import mlx.core as mx
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        token_ids = tokenizer.encode("The capital of France is")
        input_ids = mx.array([token_ids])
        logits = model(input_ids)
        mx.eval(logits)
        next_token_logits = logits[0, -1, :]
        next_token_id = mx.argmax(next_token_logits).item()
        assert isinstance(next_token_id, int)

    def test_mx_eval_no_error(
        self, fake_mlx_modules: types.ModuleType, fake_mlx_core: types.ModuleType
    ) -> None:
        import mlx.core as mx

        # mx.eval() is a no-op on fake tensors and must not raise.
        mx.eval(MagicMock())
        mx.eval()

    def test_mx_array_wraps_token_ids(
        self, fake_mlx_modules: types.ModuleType, fake_mlx_core: types.ModuleType
    ) -> None:
        import mlx.core as mx

        token_ids = [1, 2, 3, 4, 5]
        arr = mx.array([token_ids])
        assert arr is not None


# ---------------------------------------------------------------------------
# mlx_lm.generate()
# ---------------------------------------------------------------------------


class TestRawMLXGenerate:
    """mlx_lm.generate() produces a text string from model + tokenizer."""

    def test_generate_returns_string(self, fake_mlx_modules: types.ModuleType) -> None:
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        result = mlx_lm.generate(model, tokenizer, prompt="Hello world", max_tokens=64)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_echoes_prompt_prefix(self, fake_mlx_modules: types.ModuleType) -> None:
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        prompt = "The capital of France is"
        result = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=32)
        # Fake generate returns "generated: <first 20 chars of prompt>"
        assert prompt[:20] in result

    def test_generate_with_chat_template_prompt(
        self, fake_mlx_modules: types.ModuleType
    ) -> None:
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/test-model-4bit")
        messages = [{"role": "user", "content": "Name three planets."}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result = mlx_lm.generate(model, tokenizer, prompt=formatted, max_tokens=64)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Pattern 21 end-to-end (example function smoke test)
# ---------------------------------------------------------------------------


class TestPattern21EndToEnd:
    """Smoke-test the full Pattern 21 flow as demonstrated in the example."""

    def test_full_pattern_no_error(
        self,
        fake_mlx_modules: types.ModuleType,
        fake_mlx_core: types.ModuleType,
        capsys: pytest.CaptureFixture,
    ) -> None:
        import mlx.core as mx
        import mlx_lm

        model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

        # Raw forward pass
        token_ids = tokenizer.encode("The capital of France is")
        input_ids = mx.array([token_ids])
        logits = model(input_ids)
        mx.eval(logits)
        next_token_id = mx.argmax(logits[0, -1, :]).item()

        # Text generation
        messages = [{"role": "user", "content": "Name three planets."}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = mlx_lm.generate(model, tokenizer, prompt=formatted, max_tokens=64)

        assert isinstance(next_token_id, int)
        assert isinstance(text, str)

    def test_missing_mlx_import_handled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """pattern_21_raw_mlx_forward() prints a helpful message when mlx is absent."""
        import sys

        # Remove mlx from sys.modules so the import inside the function fails.
        monkeypatch.delitem(sys.modules, "mlx", raising=False)
        monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
        monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)

        import importlib
        import examples.apple_on_device_lm as ex_mod  # noqa: PLC0415

        importlib.reload(ex_mod)

        # Temporarily make mlx unimportable.
        monkeypatch.setitem(sys.modules, "mlx", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "mlx.core", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "mlx_lm", None)  # type: ignore[arg-type]

        ex_mod.pattern_21_raw_mlx_forward()

        out = capsys.readouterr().out
        assert "not installed" in out or "pip install" in out
