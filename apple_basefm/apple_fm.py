"""DSPy adapter for Apple's on-device Foundation Models.

Requires macOS 26+ with Apple Intelligence enabled and the Apple Foundation
Models SDK, installed via Apple's developer distribution channel (not on PyPI).

Usage::

    import dspy
    lm = dspy.AppleFoundationLM()
    dspy.configure(lm=lm)
    result = dspy.Predict("question -> answer")(question="What is DSPy?")
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import platform
from collections import OrderedDict
from typing import Any, Literal, get_args, get_origin

from apple_basefm._base import _AppleBaseLM, _flatten_messages, _run_async
from apple_basefm._compat import get_dspy_cache
from apple_basefm._response import _FMChoice, _FMMessage, _FMResponse, _FMUsage  # noqa: F401
from apple_basefm._telemetry import forward_span, record_usage

logger = logging.getLogger(__name__)


def _pydantic_to_generable(model_cls: type, fm: Any) -> type | None:
    """Dynamically build an Apple @generable dataclass from a Pydantic model.

    Field type mapping:

    * Literal[a, b, c]  → fm.guide(name, anyOf=[a, b, c])
    * int with ge + le  → fm.guide(name, range=(ge, le))
    * str with pattern  → fm.guide(name, regex=pattern)
    * Everything else   → plain annotation with no constraint (fallback)

    Mixed-type Literals (e.g. Literal[1, "two"]) fall back to Any to avoid
    make_dataclass type mismatches.

    Args:
        model_cls: A Pydantic BaseModel subclass.
        fm: The imported apple_fm_sdk module.

    Returns:
        A @generable-decorated dataclass, or None if conversion fails.
    """
    fields: list[tuple[str, type, Any]] = []

    for field_name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        guide_kwargs: dict[str, Any] = {}
        raw_annotation = annotation

        if origin is Literal:
            # Guard against mixed-type Literals (e.g. Literal[1, "two"]).
            # Using type(args[0]) would give int, but args[1] is str — make_dataclass
            # would get a type mismatch. Fall back to Any for safety.
            raw_annotation = (
                type(args[0]) if args and len({type(a) for a in args}) == 1 else Any  # type: ignore[assignment]
            )
            guide_kwargs["anyOf"] = list(args)
        else:
            ge_val: Any = None
            le_val: Any = None
            for meta in getattr(field_info, "metadata", []):
                if hasattr(meta, "ge") and meta.ge is not None:
                    ge_val = meta.ge
                if hasattr(meta, "le") and meta.le is not None:
                    le_val = meta.le
                pattern = getattr(meta, "pattern", None)
                if pattern:
                    guide_kwargs["regex"] = pattern
            if ge_val is not None and le_val is not None:
                guide_kwargs["range"] = (ge_val, le_val)

        if guide_kwargs:
            try:
                default = fm.guide(field_name, **guide_kwargs)
                fields.append((field_name, raw_annotation, default))
                continue
            except Exception as exc:
                logger.warning(
                    "apple_fm: could not create fm.guide for field %r (%s),"
                    " using unconstrained: %s",
                    field_name,
                    guide_kwargs,
                    exc,
                )

        # Unconstrained field — use a sensible default so make_dataclass is happy.
        default_val: Any = "" if raw_annotation is str else None
        fields.append((field_name, raw_annotation, dataclasses.field(default=default_val)))

    try:
        dyn_cls = dataclasses.make_dataclass(model_cls.__name__, fields)
        return fm.generable(dyn_cls)
    except Exception as exc:
        logger.warning("apple_fm: failed to build @generable class from %r: %s", model_cls, exc)
        return None


# Cache of dynamically generated fm.Tool subclasses, keyed by (tool_name, id(func)).
#
# id(func) is stable for the lifetime of a DSPy program object but Python reuses
# memory addresses after GC. We therefore cap the cache at _TOOL_CACHE_MAXSIZE
# entries using simple LRU eviction (OrderedDict.move_to_end). This bounds memory
# and limits the window in which a stale id() collision could serve the wrong class.
#
# Assumption: tools are stable objects (module-level functions or dspy.Predict
# bound tools) for the lifetime of a training/inference session.
_TOOL_CACHE_MAXSIZE = 256
_tool_class_cache: OrderedDict[tuple[str, int], type] = OrderedDict()


def _dspy_tool_to_apple_tool(dspy_tool: Any, fm: Any) -> Any:
    """Wrap a DSPy tool in an Apple fm.Tool subclass.

    Generated subclasses are cached by (tool_name, id(func)) with LRU eviction
    at _TOOL_CACHE_MAXSIZE entries.

    Args:
        dspy_tool: A DSPy tool object with a .name attribute and either
            callable itself or exposing a .func attribute.
        fm: The imported apple_fm_sdk module.

    Returns:
        An instantiated fm.Tool subclass wired to the DSPy tool's callable.
    """
    tool_name = getattr(dspy_tool, "name", type(dspy_tool).__name__)
    func = dspy_tool if callable(dspy_tool) else getattr(dspy_tool, "func", None)

    cache_key = (tool_name, id(func))
    if cache_key in _tool_class_cache:
        _tool_class_cache.move_to_end(cache_key)
        return _tool_class_cache[cache_key]()

    class _WrappedTool(fm.Tool):
        def call(self, **kwargs: Any) -> Any:
            """Delegate the tool call to the underlying DSPy callable."""
            if func is None:
                raise NotImplementedError(f"Tool {tool_name!r} has no callable implementation")
            return func(**kwargs)

    _WrappedTool.__name__ = tool_name

    if len(_tool_class_cache) >= _TOOL_CACHE_MAXSIZE:
        _tool_class_cache.popitem(last=False)  # evict LRU

    _tool_class_cache[cache_key] = _WrappedTool
    return _WrappedTool()


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class AppleFoundationLM(_AppleBaseLM):
    """DSPy language model adapter for Apple's on-device Foundation Models.

    Wraps apple_fm_sdk.SystemLanguageModel + LanguageModelSession in a
    BaseLM subclass. Key features:

    * Native guided generation: when response_format is a Pydantic model,
      the adapter dynamically builds an Apple @generable dataclass and uses
      the model's native constrained decoding instead of injecting a JSON
      schema into the prompt.
    * Tool calling: DSPy tools are converted to fm.Tool subclasses and
      registered on the session.
    * Async bridging: Apple's SDK is async-only; forward() bridges to sync
      via asyncio.run() with nest_asyncio support for notebooks.

    Requirements:
        * macOS 26+ with Apple Intelligence enabled.
        * apple-fm-sdk installed from Apple's developer distribution channel.

    Args:
        model: Identifier string stored in history and cache keys.
        temperature: Passed to GenerationOptions if supported by the SDK.
            None omits the option entirely (model uses its default).
        max_tokens: Maximum tokens for the response. Passed to GenerationOptions;
            falls back gracefully if the SDK version does not support the parameter.
        cache: Whether to enable DSPy's request cache.
        timeout: Maximum seconds to wait for a single session.respond() call.
            Defaults to 120. Pass None to disable.
        **kwargs: Additional keyword arguments forwarded to BaseLM.__init__.

    Raises:
        RuntimeError: If not running on macOS, or if Apple Intelligence is
            unavailable on the current device.
        ImportError: If apple-fm-sdk is not installed.
    """

    # --- Capability properties (queried by DSPy adapters post #9516 refactor) ---

    @property
    def supports_function_calling(self) -> bool:
        return True  # native fm.Tool support

    @property
    def supports_response_schema(self) -> bool:
        return True  # native @generable constrained decoding

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def supported_params(self) -> list[str]:
        return ["temperature", "max_tokens"]

    def __init__(
        self,
        model: str = "apple/on-device",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        timeout: float | None = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens if max_tokens is not None else 1000,
            cache=cache,
            **kwargs,
        )

        if platform.system() != "Darwin":
            raise RuntimeError(
                "AppleFoundationLM requires macOS 26+ with Apple Intelligence enabled. "
                f"Current platform: {platform.system()!r}"
            )

        try:
            import apple_fm_sdk as fm
        except ImportError as exc:
            raise ImportError(
                "apple-fm-sdk is not installed. "
                "Install it with: pip install 'apple-basefm[foundation,apple-fm-sdk,dspy]'\n"
                "Setup guide: https://apple.github.io/python-apple-fm-sdk/getting_started.html"
            ) from exc

        self._fm = fm
        self._apple_model = fm.SystemLanguageModel()

        available, reason = self._apple_model.is_available()
        if not available:
            raise RuntimeError(
                f"Apple Foundation Model is not available on this device: {reason}\n"
                "Ensure Apple Intelligence is enabled in "
                "System Settings \u2192 Apple Intelligence & Siri."
            )

        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        # Apple's on-device model has a fixed context window. The SDK does not expose
        # this value programmatically; 4096 is the documented limit for the initial release.
        self.context_window: int = 4096

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_generation_options(self) -> Any | None:
        """Build a GenerationOptions object if any sampling params are set."""
        fm = self._fm
        opts: dict[str, Any] = {}
        if self._temperature is not None:
            opts["temperature"] = self._temperature
        if self._max_tokens is not None:
            # Pass max_tokens when the SDK supports it. The Apple FM SDK may not
            # expose this kwarg on all releases; fall back gracefully if it raises.
            opts["max_tokens"] = self._max_tokens
        if not opts:
            return None
        try:
            return fm.GenerationOptions(**opts)
        except TypeError:
            # SDK version does not accept max_tokens yet — retry without it.
            opts.pop("max_tokens", None)
            if not opts:
                return None
            try:
                return fm.GenerationOptions(**opts)
            except Exception as exc:
                logger.debug("apple_fm: could not create GenerationOptions(%s): %s", opts, exc)
                return None
        except Exception as exc:
            logger.debug("apple_fm: could not create GenerationOptions(%s): %s", opts, exc)
            return None

    # ------------------------------------------------------------------
    # BaseLM interface
    # ------------------------------------------------------------------

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Synchronous forward pass — checks DSPy cache, then bridges to aforward.

        Args:
            prompt: Plain-text prompt string.
            messages: List of {"role": ..., "content": ...} dicts.
            **kwargs: Generation parameters forwarded to aforward.

        Returns:
            An _FMResponse compatible with BaseLM._process_completion.

        Raises:
            NotImplementedError: If stream=True is passed.
        """
        _cache = get_dspy_cache()
        cache = kwargs.pop("cache", self.cache)

        if kwargs.get("stream"):
            raise NotImplementedError(
                "Streaming is not yet supported for AppleFoundationLM. "
                "Call forward() for a blocking response."
            )

        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        _skip = {"num_retries", "stream", "n"}
        cache_request = {
            "model": self.model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k not in _skip},
        }

        if cache:
            cached = _cache.get(cache_request)
            if cached is not None:
                return cached

        with forward_span(model=self.model, backend="foundation", max_tokens=self._max_tokens):
            response = _run_async(self.aforward(prompt=None, messages=messages, **kwargs))

        if cache:
            _cache.put(cache_request, response)

        return response

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Async forward pass — primary implementation for Apple on-device inference.

        Structured output path (response_format is a Pydantic model):
            Builds an Apple @generable class from the Pydantic schema and calls
            session.respond(generating=...). On failure, recreates the session
            and retries without the schema constraint.

        Plain text path:
            Calls session.respond(prompt=...) directly.

        Args:
            prompt: Plain-text prompt string. Ignored if messages is provided.
            messages: List of {"role": ..., "content": ...} dicts.
            **kwargs: Supports response_format, tools, and standard DSPy-internal
                keys which are stripped before reaching the SDK.

        Returns:
            An _FMResponse with the model's generated text.
        """
        fm = self._fm

        flat_prompt = _flatten_messages(messages) if messages else (prompt or "")

        response_format = kwargs.pop("response_format", None)
        raw_tools = kwargs.pop("tools", [])
        kwargs.pop("num_retries", None)
        kwargs.pop("stream", None)
        kwargs.pop("n", None)
        kwargs.pop("cache", None)

        for _k in ("temperature", "max_tokens"):
            kwargs.pop(_k, None)

        if kwargs:
            logger.warning(
                "apple_fm: ignoring unsupported kwargs %s "
                "(Apple SDK does not accept arbitrary generation parameters)",
                sorted(kwargs),
            )
            kwargs.clear()

        # Structured output: try native guided generation.
        generable_cls: type | None = None

        if response_format is not None:
            try:
                from pydantic import BaseModel as PydanticBaseModel

                if isinstance(response_format, type) and issubclass(
                        response_format, PydanticBaseModel
                ):
                    generable_cls = _pydantic_to_generable(response_format, fm)
            except ImportError:
                pass

            if generable_cls is None:
                logger.warning(
                    "apple_fm: response_format %r could not be mapped to @generable; "
                    "falling back to prompt-based JSON schema (structured output quality may vary)",
                    response_format,
                )

        # Tool conversion.
        apple_tools: list[Any] = []
        for tool in raw_tools:
            try:
                apple_tools.append(_dspy_tool_to_apple_tool(tool, fm))
            except Exception as exc:
                logger.warning("apple_fm: skipping tool %r: %s", tool, exc)

        session_kwargs: dict[str, Any] = {"model": self._apple_model}
        if apple_tools:
            session_kwargs["tools"] = apple_tools

        gen_opts = self._make_generation_options()
        respond_kwargs: dict[str, Any] = {}
        if gen_opts is not None:
            respond_kwargs["options"] = gen_opts

        session = fm.LanguageModelSession(**session_kwargs)
        try:
            if generable_cls is not None:
                try:
                    if self._timeout is not None:
                        result = await asyncio.wait_for(
                            session.respond(
                                prompt=flat_prompt,
                                generating=generable_cls,
                                **respond_kwargs,
                            ),
                            timeout=self._timeout,
                        )
                    else:
                        result = await session.respond(
                            prompt=flat_prompt,
                            generating=generable_cls,
                            **respond_kwargs,
                        )
                    text = json.dumps(dataclasses.asdict(result))
                except (TimeoutError, asyncio.TimeoutError):
                    raise RuntimeError(
                        f"AppleFoundationLM: session.respond() timed out after {self._timeout}s. "
                        "The on-device model may be busy. Try increasing the timeout parameter."
                    ) from None
                except Exception as exc:
                    self._raise_for_guardrail(exc)
                    logger.warning(
                        "apple_fm: native @generable generation failed (%s); "
                        "recreating session and retrying without schema constraint "
                        "(DSPy will handle JSON schema injection via prompt)",
                        exc,
                    )
                    del session
                    session = fm.LanguageModelSession(**session_kwargs)
                    try:
                        if self._timeout is not None:
                            text = await asyncio.wait_for(
                                session.respond(prompt=flat_prompt, **respond_kwargs),
                                timeout=self._timeout,
                            )
                        else:
                            text = await session.respond(prompt=flat_prompt, **respond_kwargs)
                    except (TimeoutError, asyncio.TimeoutError):
                        raise RuntimeError(
                            f"AppleFoundationLM: fallback session.respond() timed out after "
                            f"{self._timeout}s."
                        ) from None
                    except Exception as fallback_exc:
                        self._raise_for_guardrail(fallback_exc)
                        raise
            else:
                try:
                    if self._timeout is not None:
                        text = await asyncio.wait_for(
                            session.respond(prompt=flat_prompt, **respond_kwargs),
                            timeout=self._timeout,
                        )
                    else:
                        text = await session.respond(prompt=flat_prompt, **respond_kwargs)
                except (TimeoutError, asyncio.TimeoutError):
                    raise RuntimeError(
                        f"AppleFoundationLM: session.respond() timed out after {self._timeout}s."
                    ) from None
                except Exception as exc:
                    self._raise_for_guardrail(exc)
                    raise
        finally:
            del session

        response = self._build_response(text)
        return response
