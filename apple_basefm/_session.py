"""Session-level token accumulation for apple-basefm.

Provides a context manager that transparently tracks cumulative token usage
across all LM calls made within a block — whether standalone or via DSPy.

Usage::

    from apple_basefm import token_session

    lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
    with token_session() as session:
        lm.forward(messages=[{"role": "user", "content": "Hello"}])
        qa = dspy.Predict("question -> answer")
        qa(question="What is 2+2?")

    print(session.prompt_tokens)     # cumulative input tokens
    print(session.completion_tokens) # cumulative output tokens
    print(session.total_tokens)      # prompt + completion
    print(session.call_count)        # number of LM calls

Multiple sessions can be nested — each is isolated and accumulates only the
calls made within its own block::

    with token_session() as outer:
        lm.forward(...)               # counted in outer
        with token_session() as inner:
            lm.forward(...)           # counted in inner only
        # inner exits; outer resumes accumulating
        lm.forward(...)               # counted in outer

To resume accumulation across multiple blocks, pass an existing accumulator::

    acc = _SessionAccumulator()
    with token_session(accumulator=acc):
        lm.forward(...)
    with token_session(accumulator=acc):
        lm.forward(...)
    print(acc.total_tokens)          # sum across both blocks

Notes
-----
- DSPy cache hits are NOT counted — they bypass the generation path entirely.
  This is correct for cost-forecasting: cached calls have no API cost.
- AppleFoundationLM always contributes zero token counts (the on-device SDK
  does not expose token counts). call_count still increments.
- ContextVar propagates into asyncio tasks and asyncio.to_thread() (Python 3.9+)
  but NOT into manually created threading.Thread instances.
- lm.usage is a per-instance lifetime accumulator; it is not reset between
  token_session() blocks. Use lm.reset_usage() to start a fresh baseline.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator


@dataclasses.dataclass
class _SessionAccumulator:
    """Mutable accumulator for token counts across one or more LM calls.

    All fields start at zero and are incremented by each completed LM call
    that passes through _build_response(). DSPy cache hits are not counted.

    Attributes:
        prompt_tokens: Total input tokens consumed.
        completion_tokens: Total output tokens generated.
        total_tokens: Sum of prompt_tokens and completion_tokens.
        call_count: Number of completed LM calls (not counting cache hits).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0


# Module-level ContextVar. None = no active session. Set by token_session().
_current_session: ContextVar[_SessionAccumulator | None] = ContextVar(
    "_current_session", default=None
)


@contextmanager
def token_session(
    accumulator: _SessionAccumulator | None = None,
) -> Generator[_SessionAccumulator, None, None]:
    """Context manager that accumulates token usage across all LM calls in a block.

    Args:
        accumulator: An existing _SessionAccumulator to resume accumulation into.
            If None (default), a fresh accumulator starting at zero is created.
            Pass an existing accumulator to merge multiple session blocks into
            one running total.

    Yields:
        The _SessionAccumulator being written to. Inspect its fields after the
        block exits for a complete picture of token usage.

    Raises:
        TypeError: If accumulator is not None and not a _SessionAccumulator.
    """
    if accumulator is not None and not isinstance(accumulator, _SessionAccumulator):
        raise TypeError(
            f"token_session() accumulator must be a _SessionAccumulator instance, "
            f"got {type(accumulator).__name__!r}"
        )
    acc = accumulator if accumulator is not None else _SessionAccumulator()
    token = _current_session.set(acc)
    try:
        yield acc
    finally:
        _current_session.reset(token)


def _accumulate(usage: Any) -> None:
    """Add token counts from usage to the active session, if one is set.

    Safe to call at all times — returns immediately if no session is active.
    Never raises: any internal error is silently swallowed so that observability
    code cannot break the main LM call path.

    Args:
        usage: An _FMUsage-like object with prompt_tokens and completion_tokens
            attributes. total_tokens is derived from the clamped sum of the two
            fields rather than read from usage directly, to ensure consistency.
    """
    try:
        acc = _current_session.get()
        if acc is None:
            return
        p = max(0, int(getattr(usage, "prompt_tokens", 0)))
        c = max(0, int(getattr(usage, "completion_tokens", 0)))
        acc.prompt_tokens += p
        acc.completion_tokens += c
        acc.total_tokens += p + c
        acc.call_count += 1
    except Exception:
        pass


__all__ = ["_SessionAccumulator", "_accumulate", "token_session"]
