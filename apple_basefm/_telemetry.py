"""Optional OpenTelemetry span hooks for apple-basefm.

When the ``opentelemetry-api`` package is installed (``pip install
'apple-basefm[otel]'``), every ``forward()`` / ``aforward()`` call is wrapped
in an OpenTelemetry span using the ``gen_ai.*`` semantic conventions
(OpenTelemetry GenAI SIG, 2024).

When ``opentelemetry-api`` is NOT installed, every function in this module is
a no-op. There is zero import-time cost and no ``ImportError`` raised.

Span attributes written
-----------------------
Attribute                     Value
``gen_ai.system``             ``"apple_basefm"``
``gen_ai.operation.name``     ``"chat"``
``gen_ai.request.model``      model identifier string
``gen_ai.request.max_tokens`` max_tokens (if set)
``gen_ai.usage.input_tokens`` prompt_tokens from _FMUsage (written on exit)
``gen_ai.usage.output_tokens`` completion_tokens from _FMUsage (written on exit)
``gen_ai.response.finish_reason`` ``"stop"`` (on-device models always stop)

Usage::

    from apple_basefm._telemetry import forward_span

    with forward_span(model="apple/on-device", max_tokens=512) as span:
        response = await session.respond(prompt)
        record_usage(span, response.usage)

Notes
-----
- The tracer name is ``"apple_basefm"``.
- Spans are created as child spans of the ambient context if one exists.
- On exception the span status is set to ERROR and the exception is recorded.
- This module imports nothing from ``opentelemetry`` at module load time.
  The lazy check occurs once at first use and is cached.
"""
from __future__ import annotations

import contextlib
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Cached result of the first opentelemetry availability check.
# None = not yet checked; True = available; False = unavailable.
_OTEL_AVAILABLE: bool | None = None


def _otel_tracer() -> Any | None:
    """Return a live opentelemetry Tracer, or None if otel is not installed."""
    global _OTEL_AVAILABLE  # noqa: PLW0603

    if _OTEL_AVAILABLE is False:
        return None

    try:
        from opentelemetry import trace  # type: ignore[import]

        _OTEL_AVAILABLE = True
        return trace.get_tracer("apple_basefm")
    except ImportError:
        _OTEL_AVAILABLE = False
        return None


@contextmanager
def forward_span(
    *,
    model: str,
    backend: str = "foundation",
    max_tokens: int | None = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a forward pass in an OpenTelemetry span.

    When ``opentelemetry-api`` is not installed this is a transparent no-op —
    the body executes normally and no span is created.

    Args:
        model: Model identifier string (written to ``gen_ai.request.model``).
        backend: ``"foundation"`` or ``"mlx"`` (informational, not a GenAI
            semantic convention attribute).
        max_tokens: Maximum tokens to generate. Written to
            ``gen_ai.request.max_tokens`` when provided.

    Yields:
        The OpenTelemetry ``Span`` object (or ``None`` when otel is absent).
        Pass the span to :func:`record_usage` after the forward call to write
        token counts.

    Example::

        with forward_span(model="apple/on-device", max_tokens=512) as span:
            response = await session.respond(prompt)
            record_usage(span, response.usage)
    """
    tracer = _otel_tracer()
    if tracer is None:
        yield None
        return

    from opentelemetry import trace  # type: ignore[import]
    from opentelemetry.trace import StatusCode  # type: ignore[import]

    with tracer.start_as_current_span("apple_basefm.chat") as span:
        span.set_attribute("gen_ai.system", "apple_basefm")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.backend", backend)
        if max_tokens is not None:
            span.set_attribute("gen_ai.request.max_tokens", max_tokens)

        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        else:
            span.set_status(StatusCode.OK)


def record_usage(span: Any, usage: Any) -> None:
    """Write token counts from an _FMUsage object to an OpenTelemetry span.

    Safe to call when ``span`` is ``None`` (otel absent) or when ``usage``
    does not have the expected attributes.

    Args:
        span: The span yielded by :func:`forward_span`, or ``None``.
        usage: An ``_FMUsage``-like object with ``prompt_tokens`` and
            ``completion_tokens`` attributes.
    """
    if span is None:
        return
    try:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is not None:
            span.set_attribute("gen_ai.usage.input_tokens", int(prompt))
        if completion is not None:
            span.set_attribute("gen_ai.usage.output_tokens", int(completion))
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            span.set_attribute("gen_ai.usage.total_tokens", int(total))
        span.set_attribute("gen_ai.response.finish_reason", "stop")
    except Exception:
        # Never let telemetry failure propagate into the main call path.
        pass
