"""Tests for exception-handler branches across apple_basefm.

Rule enforced: no bare ``except Exception:`` without binding the variable.
Each test below corresponds to one exception-handling branch and verifies that:

  * The handler executes (does not crash or silently skip)
  * The expected fallback value / side-effect is produced
  * The original exception is not re-raised unless that is the stated contract

Test naming: ``test_<subject>_<condition>_<expected_outcome>``
"""
from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_apple_local() -> types.ModuleType:
    sys.modules.pop("apple_basefm.apple_local", None)
    return importlib.import_module("apple_basefm.apple_local")


@pytest.fixture()
def apple_local_mod(fake_mlx_modules: types.ModuleType) -> types.ModuleType:  # noqa: ARG001
    """Reload apple_local with fake mlx_lm present, patching macOS+arm64."""
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        return _reload_apple_local()


class _BrokenDspySettings:
    """A fake dspy module whose .settings property always raises."""

    @property
    def settings(self) -> None:  # type: ignore[override]
        raise RuntimeError("simulated dspy.settings failure")


class _BadTool:
    """A fake tool whose .name property always raises."""

    @property
    def name(self) -> str:
        raise RuntimeError("UNIQUE_TOOL_CONV_ERROR")


# ---------------------------------------------------------------------------
# _compat.py — settings shims
# Each shim must return None (not raise) when dspy.settings access blows up.
# ---------------------------------------------------------------------------


class TestCompatShimsReturnNoneOnDspyError:
    """get_send_stream / get_caller_predict / get_usage_tracker must never
    propagate unexpected exceptions — they are called in hot inference paths
    and must degrade gracefully."""

    def test_get_send_stream_returns_none_when_dspy_settings_raises(self) -> None:
        """If dspy.settings raises, get_send_stream() must return None."""
        import apple_basefm._compat as compat

        broken = _BrokenDspySettings()
        with patch.dict(sys.modules, {"dspy": broken}):  # type: ignore[arg-type]
            result = compat.get_send_stream()
        assert result is None

    def test_get_caller_predict_returns_none_when_dspy_settings_raises(self) -> None:
        import apple_basefm._compat as compat

        broken = _BrokenDspySettings()
        with patch.dict(sys.modules, {"dspy": broken}):  # type: ignore[arg-type]
            result = compat.get_caller_predict()
        assert result is None

    def test_get_usage_tracker_returns_none_when_dspy_settings_raises(self) -> None:
        import apple_basefm._compat as compat

        broken = _BrokenDspySettings()
        with patch.dict(sys.modules, {"dspy": broken}):  # type: ignore[arg-type]
            result = compat.get_usage_tracker()
        assert result is None

    def test_get_send_stream_logs_debug_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the exception is caught, a DEBUG message must be emitted."""
        import logging

        import apple_basefm._compat as compat

        broken = _BrokenDspySettings()
        with (
            patch.dict(sys.modules, {"dspy": broken}),  # type: ignore[arg-type]
            caplog.at_level(logging.DEBUG, logger="apple_basefm._compat"),
        ):
            compat.get_send_stream()
        assert "get_send_stream" in caplog.text


# ---------------------------------------------------------------------------
# apple_fm._pydantic_to_generable — fm.guide() failure path
# ---------------------------------------------------------------------------


class TestPydanticToGenerableExceptionPaths:
    """_pydantic_to_generable must not raise when fm.guide() or make_dataclass fails."""

    def _get_fn(self) -> Any:
        import apple_basefm.apple_fm as afm

        return afm._pydantic_to_generable

    def test_guide_failure_does_not_raise(
        self, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """When fm.guide() raises, _pydantic_to_generable must not propagate the error."""
        from pydantic import BaseModel, Field

        fake_apple_fm_sdk.guide = MagicMock(side_effect=RuntimeError("sdk boom"))  # type: ignore[attr-defined]

        class Item(BaseModel):
            name: str = Field(pattern=r"^\w+$")

        result = self._get_fn()(Item, fake_apple_fm_sdk)
        # Must not raise; may return a generable class or None depending on further steps
        assert result is None or callable(result)

    def test_guide_failure_exception_logged_with_error_text(
        self, fake_apple_fm_sdk: types.ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The exc text must appear in the warning log (not silently dropped)."""
        import logging

        from pydantic import BaseModel, Field

        fake_apple_fm_sdk.guide = MagicMock(side_effect=RuntimeError("UNIQUE_GUIDE_ERROR"))  # type: ignore[attr-defined]

        class Item(BaseModel):
            name: str = Field(pattern=r"^\w+$")

        with caplog.at_level(logging.WARNING, logger="apple_basefm.apple_fm"):
            self._get_fn()(Item, fake_apple_fm_sdk)

        assert "UNIQUE_GUIDE_ERROR" in caplog.text

    def test_make_dataclass_failure_returns_none(
        self, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """If @generable compilation raises, _pydantic_to_generable must return None."""
        from pydantic import BaseModel

        fake_apple_fm_sdk.generable = MagicMock(side_effect=TypeError("generable boom"))  # type: ignore[attr-defined]

        class Simple(BaseModel):
            value: str

        result = self._get_fn()(Simple, fake_apple_fm_sdk)
        assert result is None

    def test_make_dataclass_failure_exception_logged(
        self, fake_apple_fm_sdk: types.ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The exc text for @generable failure must appear in the warning log."""
        import logging

        from pydantic import BaseModel

        fake_apple_fm_sdk.generable = MagicMock(side_effect=TypeError("UNIQUE_MAKE_DC_ERROR"))  # type: ignore[attr-defined]

        class Simple(BaseModel):
            value: str

        with caplog.at_level(logging.WARNING, logger="apple_basefm.apple_fm"):
            self._get_fn()(Simple, fake_apple_fm_sdk)

        assert "UNIQUE_MAKE_DC_ERROR" in caplog.text


# ---------------------------------------------------------------------------
# apple_fm._make_generation_options — SDK constructor failure
# ---------------------------------------------------------------------------


class TestMakeGenerationOptionsExceptionPath:
    """_make_generation_options must return None (not raise) if the SDK rejects params."""

    def test_generation_options_failure_returns_none(
        self, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        with patch("platform.system", return_value="Darwin"):
            import apple_basefm.apple_fm as afm

            lm = afm.AppleFoundationLM(temperature=0.5)

        fake_apple_fm_sdk.GenerationOptions = MagicMock(side_effect=ValueError("bad option"))  # type: ignore[attr-defined]
        result = lm._make_generation_options()
        assert result is None

    def test_generation_options_failure_exception_logged(
        self,
        fake_apple_fm_sdk: types.ModuleType,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        with patch("platform.system", return_value="Darwin"):
            import apple_basefm.apple_fm as afm

            lm = afm.AppleFoundationLM(temperature=0.5)

        fake_apple_fm_sdk.GenerationOptions = MagicMock(  # type: ignore[attr-defined]
            side_effect=ValueError("UNIQUE_GEN_OPTS_ERROR")
        )
        with caplog.at_level(logging.DEBUG, logger="apple_basefm.apple_fm"):
            lm._make_generation_options()

        assert "UNIQUE_GEN_OPTS_ERROR" in caplog.text

    def test_max_tokens_passed_when_sdk_accepts_it(
        self, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """When max_tokens is set and SDK accepts it, GenerationOptions gets max_tokens."""
        with patch("platform.system", return_value="Darwin"):
            import apple_basefm.apple_fm as afm

            lm = afm.AppleFoundationLM(temperature=0.5, max_tokens=256)

        call_kwargs: list[dict] = []

        def _capture(**kwargs: Any) -> MagicMock:
            call_kwargs.append(kwargs)
            return MagicMock()

        fake_apple_fm_sdk.GenerationOptions = _capture  # type: ignore[attr-defined]
        result = lm._make_generation_options()
        assert result is not None
        assert call_kwargs[0].get("max_tokens") == 256

    def test_type_error_retries_without_max_tokens(
        self, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """TypeError on first call (SDK doesn't support max_tokens) must retry without it."""
        with patch("platform.system", return_value="Darwin"):
            import apple_basefm.apple_fm as afm

            lm = afm.AppleFoundationLM(temperature=0.5, max_tokens=256)

        call_count = 0

        def _fail_first(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if "max_tokens" in kwargs:
                raise TypeError("unexpected keyword argument 'max_tokens'")
            return MagicMock()

        fake_apple_fm_sdk.GenerationOptions = _fail_first  # type: ignore[attr-defined]
        result = lm._make_generation_options()
        assert result is not None
        assert call_count == 2  # first attempt failed, second succeeded

    def test_type_error_both_attempts_returns_none(
        self, fake_apple_fm_sdk: types.ModuleType
    ) -> None:
        """If both GenerationOptions attempts raise, _make_generation_options returns None."""
        with patch("platform.system", return_value="Darwin"):
            import apple_basefm.apple_fm as afm

            lm = afm.AppleFoundationLM(temperature=0.5, max_tokens=256)

        fake_apple_fm_sdk.GenerationOptions = MagicMock(  # type: ignore[attr-defined]
            side_effect=TypeError("always fails")
        )
        result = lm._make_generation_options()
        assert result is None


# ---------------------------------------------------------------------------
# apple_fm.aforward — tool conversion failure path
# ---------------------------------------------------------------------------


class TestAforwardToolConversionFailure:
    """A tool that fails to convert must be skipped with a warning, not abort aforward."""

    @pytest.mark.asyncio
    async def test_bad_tool_skipped_aforward_still_returns_response(
        self, fake_apple_fm_sdk: types.ModuleType  # noqa: ARG002
    ) -> None:
        with patch("platform.system", return_value="Darwin"):
            import apple_basefm.apple_fm as afm

            lm = afm.AppleFoundationLM()

        response = await lm.aforward(
            messages=[{"role": "user", "content": "hello"}],
            tools=[_BadTool()],
        )
        assert response.choices[0].message.content != ""

    @pytest.mark.asyncio
    async def test_bad_tool_exception_logged(
        self,
        fake_apple_fm_sdk: types.ModuleType,  # noqa: ARG002
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        with patch("platform.system", return_value="Darwin"):
            import apple_basefm.apple_fm as afm

            lm = afm.AppleFoundationLM()

        with caplog.at_level(logging.WARNING, logger="apple_basefm.apple_fm"):
            await lm.aforward(
                messages=[{"role": "user", "content": "hello"}],
                tools=[_BadTool()],
            )
        assert "UNIQUE_TOOL_CONV_ERROR" in caplog.text


# ---------------------------------------------------------------------------
# _mlx._apply_chat_template — tokenizer.apply_chat_template failure
# ---------------------------------------------------------------------------


class TestApplyChatTemplateExceptionPath:
    """When apply_chat_template raises, _apply_chat_template must fall back to concat."""

    def test_apply_chat_template_failure_falls_back_to_concat(self) -> None:
        from apple_basefm._mlx import _apply_chat_template

        class BadTokenizer:
            def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
                raise RuntimeError("template error")

        messages = [{"role": "user", "content": "hello world"}]
        result = _apply_chat_template(BadTokenizer(), messages)
        assert "hello world" in result

    def test_apply_chat_template_failure_exception_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from apple_basefm._mlx import _apply_chat_template

        class BadTokenizer:
            def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
                raise RuntimeError("UNIQUE_TEMPLATE_ERROR")

        with caplog.at_level(logging.DEBUG, logger="apple_basefm._mlx"):
            _apply_chat_template(BadTokenizer(), [{"role": "user", "content": "x"}])

        assert "UNIQUE_TEMPLATE_ERROR" in caplog.text

    def test_apply_chat_template_no_method_falls_back_silently(self) -> None:
        """Tokenizers without apply_chat_template must fall back without raising."""
        from apple_basefm._mlx import _apply_chat_template

        class PlainTokenizer:
            pass

        messages = [{"role": "user", "content": "bare fallback"}]
        result = _apply_chat_template(PlainTokenizer(), messages)
        assert "bare fallback" in result


# ---------------------------------------------------------------------------
# _mlx._build_schema_processor — outlines failure paths
# ---------------------------------------------------------------------------


class TestBuildSchemaProcessorExceptionPaths:
    """_build_schema_processor must return None (not raise) when outlines is absent
    or raises during FSM compilation."""

    def _make_lm(self, mod: types.ModuleType) -> Any:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            return mod.AppleLocalLM("test-model")

    def test_outlines_import_error_returns_none(
        self, apple_local_mod: types.ModuleType
    ) -> None:
        """When outlines is not installed, processor must be None, not an exception."""
        lm = self._make_lm(apple_local_mod)
        schema = {"type": "object", "properties": {"value": {"type": "string"}}}

        with patch.dict(sys.modules, {"outlines": None, "outlines.generator": None}):
            result = lm._build_schema_processor(schema)
        assert result is None

    def test_outlines_compile_error_returns_none(
        self, apple_local_mod: types.ModuleType
    ) -> None:
        """When FSM compilation raises a non-ImportError, processor must be None."""
        lm = self._make_lm(apple_local_mod)
        schema = {"type": "object"}

        # Inject both required modules so imports succeed; make the call raise.
        fake_outlines = types.ModuleType("outlines")
        fake_outlines.MLXLM = MagicMock  # type: ignore[attr-defined]

        fake_gen = types.ModuleType("outlines.generator")

        def _boom(*_: Any, **__: Any) -> Any:
            raise RuntimeError("UNIQUE_OUTLINES_COMPILE_ERROR")

        fake_gen.get_json_schema_logits_processor = _boom  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "outlines": fake_outlines,
                "outlines.generator": fake_gen,
            },
        ):
            result = lm._build_schema_processor(schema)
        assert result is None

    def test_outlines_compile_error_exception_logged(
        self,
        apple_local_mod: types.ModuleType,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        lm = self._make_lm(apple_local_mod)
        schema = {"type": "object"}

        # Both `from outlines import MLXLM` and `from outlines.generator import ...`
        # must succeed so that execution reaches the `get_json_schema_logits_processor`
        # call — only then will a non-ImportError exception hit the except Exception branch.
        fake_outlines = types.ModuleType("outlines")
        fake_outlines.MLXLM = MagicMock  # type: ignore[attr-defined]

        fake_gen = types.ModuleType("outlines.generator")

        def _boom(*_: Any, **__: Any) -> Any:
            raise RuntimeError("UNIQUE_OUTLINES_LOG_CHECK")

        fake_gen.get_json_schema_logits_processor = _boom  # type: ignore[attr-defined]

        with (
            patch.dict(
                sys.modules,
                {
                    "outlines": fake_outlines,
                    "outlines.generator": fake_gen,
                },
            ),
            caplog.at_level(logging.WARNING, logger="apple_basefm._mlx"),
        ):
            lm._build_schema_processor(schema)

        assert "UNIQUE_OUTLINES_LOG_CHECK" in caplog.text
