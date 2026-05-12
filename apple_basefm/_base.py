"""Shared infrastructure for Apple on-device language model adapters.

Provides the helpers and base class shared by apple_fm.AppleFoundationLM
and apple_local.AppleLocalLM. Private implementation detail — only the
concrete adapter classes are part of the public apple_basefm API.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from apple_basefm._compat import BaseLM
from apple_basefm._response import _FMChoice, _FMMessage, _FMResponse, _FMUsage
from apple_basefm._session import _SessionAccumulator, _accumulate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten a list of role/content message dicts into a single prompt string.

    Apple's LanguageModelSession.respond() takes a plain string rather than a
    structured message list. System instructions are included as plain context
    at the top — bracket prefixes such as [System]: trigger Apple's on-device
    content guardrails and must be avoided.

    Args:
        messages: List of {"role": ..., "content": ...} dicts following the
            OpenAI chat format. Multi-modal content lists are supported;
            only text blocks are extracted.

    Returns:
        A single string with all non-empty message contents joined by "\\n\\n".

    Raises:
        ValueError: If all messages are empty and the resulting prompt is blank.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content blocks — extract text parts only.
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if not content:
            continue
        if role == "system":
            # No bracket prefix — "[System]:" triggers Apple's on-device content
            # guardrails (pattern-matched as a jailbreak attempt). System content
            # is included as plain context at the top of the prompt.
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _run_async(coro: Any) -> Any:
    """Execute an async coroutine synchronously, regardless of event-loop state.

    Works in both plain Python scripts (no running event loop) and Jupyter
    notebooks / async frameworks (running event loop). In the latter case,
    nest_asyncio must be installed.

    Args:
        coro: An awaitable coroutine to execute.

    Returns:
        The return value of the coroutine.

    Raises:
        RuntimeError: If called from within a running event loop and
            nest_asyncio is not installed, or if apply() fails.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        try:
            import nest_asyncio

            nest_asyncio.apply(loop)
        except ImportError:
            raise RuntimeError(
                "AppleFoundationLM.forward() was called from within a running event loop "
                "(e.g. a Jupyter notebook). Install nest_asyncio to enable this:\n"
                " pip install nest_asyncio"
            ) from None
        except Exception as exc:
            raise RuntimeError(
                f"AppleFoundationLM.forward() could not patch the running event loop: {exc}\n"
                "Try calling aforward() directly from async code instead."
            ) from exc
        return loop.run_until_complete(coro)

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------


class _AppleBaseLM(BaseLM):
    """Shared base class for Apple on-device language model adapters.

    Provides the _build_response factory method used by both
    AppleFoundationLM and AppleLocalLM. Not intended for direct
    instantiation — use one of the concrete subclasses instead.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.usage = _SessionAccumulator()
        super().__init__(**kwargs)

    def reset_usage(self) -> None:
        """Reset the per-instance lifetime token counter to zero."""
        self.usage = _SessionAccumulator()

    @staticmethod
    def _raise_for_guardrail(exc: Exception) -> None:
        """Re-raise exc as a descriptive RuntimeError if it is a guardrail violation.

        Apple's SDK raises GuardrailViolationError when the on-device content
        filter rejects a prompt. This helper surfaces that as a plain
        RuntimeError with actionable guidance and leaves all other exceptions
        untouched so they propagate normally.

        Args:
            exc: The exception caught from an SDK respond() call.

        Raises:
            RuntimeError: If exc is a guardrail violation.
        """
        if "GuardrailViolation" in type(exc).__name__:
            raise RuntimeError(
                "Apple Foundation Model guardrail violation: the prompt was rejected "
                "by Apple's on-device content filter. Try rephrasing your input.\n"
                f"SDK error: {exc}"
            ) from exc

    def _build_response(self, text: str, usage: _FMUsage | None = None) -> _FMResponse:
        """Wrap a raw text string in an OpenAI-compatible _FMResponse.

        Also updates ``self.usage`` (per-instance lifetime counter) and the
        active ``token_session()`` accumulator, if one is set.

        Args:
            text: The model's generated text.
            usage: Pre-computed token-usage statistics. Pass None to get zeroed
                counters — appropriate when the underlying SDK does not expose
                token counts (e.g. AppleFoundationLM).

        Returns:
            An _FMResponse with a single choice and response_cost=0.0.

        Raises:
            ValueError: If text is empty after any SDK or MLX generation call,
                which would produce a silent no-op in DSPy's output parser.
        """
        if not text:
            raise ValueError(
                "Model returned an empty response. The prompt may have been too long, "
                "or the model may have generated only whitespace. Try rephrasing."
            )
        response = _FMResponse(
            choices=[_FMChoice(message=_FMMessage(content=text))],
            usage=usage or _FMUsage(),
            model=self.model,
            _hidden_params={"response_cost": 0.0},
        )
        try:
            u = response.usage
            p = max(0, int(getattr(u, "prompt_tokens", 0)))
            c = max(0, int(getattr(u, "completion_tokens", 0)))
            self.usage.prompt_tokens += p
            self.usage.completion_tokens += c
            self.usage.total_tokens += p + c
            self.usage.call_count += 1
            _accumulate(u)
        except Exception:
            pass
        return response


__all__ = ["_flatten_messages", "_run_async", "_AppleBaseLM"]
