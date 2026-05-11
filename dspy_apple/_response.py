"""Shared response types that satisfy the BaseLM._process_lm_response contract.

_FMUsage    — implements __iter__ so dict(usage) works without a Mapping subclass
_FMMessage  — message wrapper with .content and optional .tool_calls attributes
_FMChoice   — choice wrapper with a .message attribute
_FMResponse — top-level response with .choices, .usage, and ._hidden_params
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any


@dataclasses.dataclass
class _FMUsage:
    """Token-usage statistics for a completion, mirroring OpenAI's Usage object.

    Implements the mapping protocol (__iter__) so that dict(usage) works as
    expected by BaseLM history tracking.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Yield (key, value) pairs so that dict(usage) works as expected by BaseLM."""
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


@dataclasses.dataclass
class _FMMessage:
    """A single message in a completion response, mirroring OpenAI's format."""

    content: str
    tool_calls: list[Any] | None = None


@dataclasses.dataclass
class _FMChoice:
    """One completion choice, mirroring OpenAI's Choice object."""

    message: _FMMessage


@dataclasses.dataclass
class _FMResponse:
    """A completion response object compatible with BaseLM._process_completion.

    Mirrors the subset of OpenAI's ChatCompletion that DSPy accesses.

    _hidden_params must always contain "response_cost" (even 0.0) — BaseLM's
    history aggregator calls entry["cost"] on the value returned by
    dict(response._hidden_params); a missing key raises TypeError.
    On-device inference has no monetary cost; 0.0 is accurate and safe.
    """

    choices: list[_FMChoice]
    usage: _FMUsage
    model: str
    _hidden_params: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"response_cost": 0.0}
    )


__all__ = ["_FMUsage", "_FMMessage", "_FMChoice", "_FMResponse"]
