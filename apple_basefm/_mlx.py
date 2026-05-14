"""MLX inference internals for AppleLocalLM.

Provides _MLXMixin and its supporting dataclass / helpers.
These are private implementation details — import AppleLocalLM from
apple_basefm.apple_local for all public usage.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import queue
import threading
from collections.abc import AsyncGenerator
from typing import Any

from apple_basefm._base import _flatten_messages
from apple_basefm._response import _FMResponse, _FMUsage  # noqa: F401 — re-exported for callers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming chunk type
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _LocalStreamChunk:
    """A single token chunk emitted during AppleLocalLM streaming.

    Yielded by dspy.streamify() for each token as the model generates.
    Not a ModelResponseStream instance, so DSPy's StreamListener
    field-extraction is not available — callers receive raw token text via
    chunk.text and accumulate it manually.

    Attributes:
        text: The raw token text for this chunk.
        model: Model identifier, echoed from the AppleLocalLM instance.
        predict_id: id() of the dspy.Predict module that initiated the
            call, or None if called outside a Predict context.
    """

    text: str
    model: str
    predict_id: int | None = None


# ---------------------------------------------------------------------------
# Chat-template helpers
# ---------------------------------------------------------------------------


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Format a message list using the tokenizer's built-in chat template.

    Falls back to simple role-prefixed concatenation if the tokenizer does not
    expose apply_chat_template or if the call raises.

    Args:
        tokenizer: A HuggingFace tokenizer object, or any object that optionally
            exposes an apply_chat_template method.
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        A formatted prompt string ready to pass to mlx_lm.generate().
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.debug("apply_chat_template failed (%s); falling back to concat", exc)

    # Fallback: reuse the same simple flattener as AppleFoundationLM.
    return _flatten_messages(messages)


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------


def _response_format_to_schema(response_format: Any) -> dict[str, Any] | None:
    """Extract a JSON schema dict from a response_format value.

    Returns the Pydantic model's JSON schema if response_format is a
    pydantic.BaseModel subclass. Returns None for plain-dict formats
    (e.g. {"type": "json_object"}), None, or any non-Pydantic value.

    Args:
        response_format: Value from lm_kwargs["response_format"]. May be a
            Pydantic BaseModel subclass, a dict, or None.

    Returns:
        A JSON schema dict, or None if conversion is not possible.
    """
    try:
        from pydantic import BaseModel as PydanticBaseModel

        if isinstance(response_format, type) and issubclass(response_format, PydanticBaseModel):
            return response_format.model_json_schema()
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# MLX mixin
# ---------------------------------------------------------------------------


class _MLXMixin:
    """MLX inference methods mixed into AppleLocalLM.

    The host class __init__ must set these attributes before calling any
    method defined here:

    Attributes:
        _mlx_model: The loaded MLX model object.
        _mlx_tokenizer: The loaded HuggingFace tokenizer.
        _schema_processor_cache: Per-instance cache of compiled outlines FSMs.
        context_window: Maximum sequence length supported by the loaded model.
    """

    _mlx_model: Any
    _mlx_tokenizer: Any
    _schema_processor_cache: dict[str, Any]
    context_window: int

    # Attributes that must be shared (not copied) across deepcopy calls.
    # The executor owns the GPU stream that the model weights were loaded on;
    # the model and tokenizer objects are also bound to that thread/stream.
    _SHARED_ON_DEEPCOPY: frozenset[str] = frozenset(
        {"_mlx_executor", "_mlx_model", "_mlx_tokenizer", "_schema_processor_cache"}
    )

    def __deepcopy__(self, memo: dict) -> "_MLXMixin":
        """Custom deepcopy that shares the MLX executor and model weights.

        ``copy.deepcopy`` is called by ``dspy.LM.copy()`` (e.g. in BestOfN)
        to create a variant with different sampling settings.  The executor
        and loaded MLX objects cannot be pickled / reconstructed, and sharing
        them is correct — all copies should run on the same inference thread
        and use the same weights.
        """
        import copy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self._SHARED_ON_DEEPCOPY:
                object.__setattr__(result, k, v)
            else:
                object.__setattr__(result, k, copy.deepcopy(v, memo))
        return result

    def _load_mlx(self, model_path: str) -> tuple[Any, Any]:
        """Load an MLX model and tokenizer on a dedicated inference thread.

        Creates ``self._mlx_executor`` — a single-thread
        ``ThreadPoolExecutor`` that owns the MLX GPU stream for the lifetime
        of this instance.  All generation (``_generate``, streaming) must
        be dispatched to this executor so that every MLX operation runs on
        the same thread and therefore the same GPU stream index as the
        loaded model weights.

        Args:
            model_path: HuggingFace repo ID or absolute path to a local model directory.

        Returns:
            A (model, tokenizer) tuple as returned by mlx_lm.load().

        Raises:
            ImportError: If mlx-lm is not installed.
        """
        try:
            import mlx_lm
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for AppleLocalLM(backend='mlx'). Install it:\n"
                " pip install mlx-lm"
            ) from exc

        # One persistent worker thread owns the GPU stream for this instance.
        # Model weights are loaded on that thread so all subsequent eval calls
        # use the same stream index and avoid cross-thread stream mismatch.
        self._mlx_executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="apple-basefm-mlx"
            )
        )

        logger.info("AppleLocalLM: loading model %r via mlx-lm…", model_path)
        model, tokenizer = self._mlx_executor.submit(mlx_lm.load, model_path).result()
        logger.info("AppleLocalLM: model loaded")
        return model, tokenizer

    def _build_schema_processor(self, schema: dict[str, Any]) -> Any | None:
        """Build (or return a cached) outlines logits processor for a JSON schema.

        Uses outlines (optional dependency) to compile a finite-state machine
        that constrains mlx_lm.generate() to produce JSON matching schema.
        The compiled processor is cached on the instance so repeated calls
        with the same schema do not re-pay the FSM compilation cost.

        Note: _schema_processor_cache is accessed from both the thread pool
        (via _generate) and potentially async callers. Both results are
        deterministic and idempotent, so a race only wastes one compilation.

        Args:
            schema: A JSON Schema dict (e.g. from PydanticModel.model_json_schema()).

        Returns:
            An outlines logits processor, or None if outlines is not installed
            or if compilation fails.
        """
        cache_key = json.dumps(schema, sort_keys=True)
        if cache_key in self._schema_processor_cache:
            return self._schema_processor_cache[cache_key]

        processor: Any = None
        try:
            from outlines import MLXLM
            from outlines.generator import get_json_schema_logits_processor

            outlines_model = MLXLM(self._mlx_model, self._mlx_tokenizer)
            processor = get_json_schema_logits_processor(None, outlines_model, cache_key)
            logger.debug(
                "apple_local: compiled outlines FSM for schema (key=%d chars)", len(cache_key)
            )
        except ImportError:
            logger.warning(
                "apple_local: outlines is not installed — response_format falls back to "
                "prompt-only mode and may not be honoured by small models. "
                "Install with: pip install 'outlines[mlxlm]'"
            )
        except Exception as exc:
            logger.warning("apple_local: failed to build outlines schema processor: %s", exc)

        self._schema_processor_cache[cache_key] = processor
        return processor

    def _generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        logits_processors: list[Any] | None = None,
        prompt_cache: list[Any] | None = None,
    ) -> tuple[str, str]:
        """Run synchronous (non-streaming) MLX inference.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature forwarded to make_sampler.
            max_tokens: Maximum number of tokens to generate.
            logits_processors: Optional list of logits processors.
            prompt_cache: Per-layer KV cache list, or None for mlx-lm default.

        Returns:
            A (generated_text, flat_prompt) tuple.
        """
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)
        sampler = make_sampler(temp=float(temperature))
        generate_kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "sampler": sampler,
            "verbose": False,
        }
        if logits_processors:
            generate_kwargs["logits_processors"] = logits_processors
        if prompt_cache is not None:
            generate_kwargs["prompt_cache"] = prompt_cache

        model = self._mlx_model
        tokenizer = self._mlx_tokenizer

        def _run() -> str:
            return mlx_lm.generate(
                model,
                tokenizer,
                prompt=flat_prompt,
                **generate_kwargs,
            )

        text: str = self._mlx_executor.submit(_run).result()
        return text, flat_prompt

    async def _stream_generate_async(
        self,
        flat_prompt: str,
        temperature: float,
        max_tokens: int,
        logits_processors: list[Any] | None = None,
        prompt_cache: list[Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Bridge mlx_lm.stream_generate() (sync generator) to an async generator.

        Submits the synchronous MLX generator to ``self._mlx_executor`` (the
        dedicated inference thread) so the GPU stream is always consistent
        with the stream used to load the model weights.  Tokens are forwarded
        to the async consumer via an ``asyncio.Queue``.

        A threading.Event (cancel_event) signals the producer thread to stop
        early if the async consumer is abandoned.

        Args:
            flat_prompt: Pre-formatted prompt string from _apply_chat_template.
            temperature: Sampling temperature forwarded to make_sampler.
            max_tokens: Maximum number of tokens to generate.
            logits_processors: Optional list of logits processors.
            prompt_cache: Per-layer KV cache list, or None for mlx-lm default.

        Yields:
            Individual token strings as they are produced by the model.
        """
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancel_event = threading.Event()
        sampler = make_sampler(temp=float(temperature))
        stream_kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "sampler": sampler,
        }
        if logits_processors:
            stream_kwargs["logits_processors"] = logits_processors
        if prompt_cache is not None:
            stream_kwargs["prompt_cache"] = prompt_cache

        model = self._mlx_model
        tokenizer = self._mlx_tokenizer

        def _run() -> None:
            """Generate tokens on the dedicated MLX thread and enqueue each one."""
            try:
                for response in mlx_lm.stream_generate(
                    model,
                    tokenizer,
                    prompt=flat_prompt,
                    **stream_kwargs,
                ):
                    if cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(token_queue.put_nowait, response.text)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        self._mlx_executor.submit(_run)
        try:
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                yield token
        finally:
            cancel_event.set()


__all__ = [
    "_LocalStreamChunk",
    "_apply_chat_template",
    "_response_format_to_schema",
    "_MLXMixin",
]
