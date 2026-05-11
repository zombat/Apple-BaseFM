"""DSPy adapter for locally-managed Apple Silicon models.

Supports two inference backends via the backend parameter:

* mlx    — runs any mlx-lm-compatible model (HuggingFace repo or local
            directory) using Apple's MLX framework.
* coreml — reserved for compiled .mlpackage models. Not yet implemented.

The primary use case is as a free, private, offline preprocessing layer in
DSPy pipelines that otherwise use expensive cloud LLMs::

    import dspy
    from apple_basefm import AppleLocalLM

    local_lm = AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
    cloud_lm = dspy.LM("anthropic/claude-sonnet-4-6")

    class PreprocessAndReason(dspy.Module):
        def __init__(self):
            self.extract = dspy.Predict("raw_text -> entities, dates", lm=local_lm)
            self.reason  = dspy.Predict("entities, dates -> verdict", lm=cloud_lm)

        def forward(self, raw_text):
            extracted = self.extract(raw_text=raw_text)
            return self.reason(**extracted)

Requirements (MLX backend):
    * macOS 14+ on Apple Silicon (M1 / M2 / M3 / M4)
    * pip install mlx-lm
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
from typing import Any

from apple_basefm._base import _AppleBaseLM
from apple_basefm._compat import get_caller_predict, get_dspy_cache, get_send_stream
from apple_basefm._mlx import (
    _apply_chat_template,
    _LocalStreamChunk,
    _MLXMixin,
    _response_format_to_schema,
)
from apple_basefm._response import _FMResponse, _FMUsage

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = ("mlx", "coreml")


class AppleLocalLM(_AppleBaseLM, _MLXMixin):
    """DSPy language model adapter for locally-managed Apple Silicon models.

    Wraps mlx-lm inside a BaseLM subclass. Streaming is supported via
    dspy.streamify(): tokens are forwarded as _LocalStreamChunk objects.

    Args:
        model: HuggingFace repo ID or absolute path to a local MLX model directory.
        backend: Inference engine. "mlx" (default). "coreml" raises NotImplementedError.
        bits: Informational quantization hint (4 or 8); does NOT trigger quantization.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate per call.
        cache: Whether to enable DSPy's request cache.
        max_concurrency: Semaphore limit for concurrent aforward() calls (default 1).
        **kwargs: Forwarded to BaseLM.__init__.

    Raises:
        ValueError: If backend is not supported, or max_concurrency < 1.
        NotImplementedError: If backend="coreml" is requested.
        RuntimeError: If not running on macOS on Apple Silicon.
        ImportError: If mlx-lm is not installed.
    """

    # --- Capability properties ---

    @property
    def supports_function_calling(self) -> bool:
        return False  # mlx-lm has no native tool API

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def supported_params(self) -> list[str]:
        return ["temperature", "max_tokens"]

    def __init__(
        self,
        model: str,
        backend: str = "mlx",
        bits: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        max_concurrency: int = 1,
        **kwargs: Any,
    ) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from: {_SUPPORTED_BACKENDS}")

        if backend == "coreml":
            raise NotImplementedError(
                "CoreML backend is not yet implemented. "
                "Contributions welcome — see apple_basefm/apple_local.py.\n"
                "For now, use backend='mlx' or AppleFoundationLM() for the "
                "Apple Intelligence system model."
            )

        if max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be >= 1, got {max_concurrency!r}. "
                "Use max_concurrency=1 (default) for sequential MLX inference."
            )

        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )

        if platform.system() != "Darwin":
            raise RuntimeError(
                f"AppleLocalLM requires macOS on Apple Silicon. "
                f"Current platform: {platform.system()!r}"
            )
        if platform.machine() != "arm64":
            raise RuntimeError(
                f"AppleLocalLM requires Apple Silicon (arm64). "
                f"Current architecture: {platform.machine()!r}"
            )

        self._backend = backend
        self._bits = bits
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_concurrency = max_concurrency
        if max_concurrency > 1:
            logger.warning(
                "AppleLocalLM: max_concurrency=%d — MLX generate() thread-safety "
                "on a single model instance is not guaranteed. "
                "If you observe crashes or hangs, reduce to max_concurrency=1.",
                max_concurrency,
            )
        # Lazily initialised in aforward() to avoid binding to the wrong event loop.
        self._semaphore: asyncio.Semaphore | None = None
        # Cache of compiled outlines logits processors, keyed by JSON-serialised schema.
        self._schema_processor_cache: dict[str, Any] = {}

        if bits is not None:
            logger.info("AppleLocalLM: loading %r (expected %d-bit quantization)", model, bits)

        self._mlx_model, self._mlx_tokenizer = self._load_mlx(model)
        # HuggingFace tokenizers carry model_max_length in their saved config.
        # Some tokenizers set model_max_length to a near-infinite sentinel value
        # (e.g. 1_000_000_000_000_000_019). Cap at 131072 to avoid spurious
        # context-window warnings on every call.
        raw_ctx = getattr(self._mlx_tokenizer, "model_max_length", 4096)
        self.context_window: int = min(raw_ctx, 131_072)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_usage(self, flat_prompt: str, text: str, max_tokens: int) -> _FMUsage:
        """Compute token-usage statistics and warn if the context window is tight."""
        prompt_tokens = len(self._mlx_tokenizer.encode(flat_prompt))
        if prompt_tokens > self.context_window - max_tokens:
            logger.warning(
                "apple_local: prompt (%d tokens) + max_tokens (%d) exceeds context_window (%d); "
                "generation may be truncated",
                prompt_tokens,
                max_tokens,
                self.context_window,
            )
        completion_tokens = len(self._mlx_tokenizer.encode(text))
        return _FMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    # ------------------------------------------------------------------
    # BaseLM interface
    # ------------------------------------------------------------------

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Synchronous forward pass — checks DSPy cache, then runs MLX inference.

        When dspy.settings.send_stream is set (called from dspy.streamify()
        via asyncify), uses mlx_lm.stream_generate() and forwards each token
        to the anyio stream via anyio.from_thread.run(). Otherwise runs
        mlx_lm.generate() in a single blocking call.

        Args:
            prompt: Plain-text prompt string.
            messages: List of {"role": ..., "content": ...} dicts.
            **kwargs: Supports temperature, max_tokens, and cache overrides.

        Returns:
            An _FMResponse compatible with BaseLM._process_completion.

        Raises:
            NotImplementedError: If tools or stream=True is passed.
        """
        _cache = get_dspy_cache()
        cache = kwargs.pop("cache", self.cache)

        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        # Clamp to valid ranges before forwarding to mlx-lm (G-1, G-2).
        temperature = max(0.0, min(2.0, float(kwargs.pop("temperature", self._temperature))))
        max_tokens = max(1, int(kwargs.pop("max_tokens", self._max_tokens)))

        if kwargs.get("tools"):
            raise NotImplementedError(
                "Tool calling is not supported for AppleLocalLM (mlx-lm has no native tool API). "
                "Use AppleFoundationLM for native tool support on macOS 26+."
            )

        if kwargs.get("stream"):
            raise NotImplementedError(
                "AppleLocalLM does not support stream=True in forward(). "
                "Use dspy.streamify() to wrap your module for async streaming."
            )

        response_format = kwargs.pop("response_format", None)
        for _k in ("tools", "num_retries", "stream", "n"):
            kwargs.pop(_k, None)

        schema = _response_format_to_schema(response_format)
        logits_processors: list[Any] | None = None
        if schema is not None:
            proc = self._build_schema_processor(schema)
            if proc is not None:
                logits_processors = [proc]

        if kwargs:
            logger.warning(
                "apple_local: ignoring unsupported kwargs %s "
                "(mlx-lm does not accept arbitrary generation parameters)",
                sorted(kwargs),
            )
            kwargs.clear()

        cache_request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            cache_request["response_schema"] = json.dumps(schema, sort_keys=True)

        send_stream = get_send_stream()

        if cache and send_stream is None:
            cached = _cache.get(cache_request)
            if cached is not None:
                return cached

        if send_stream is not None:
            import mlx_lm
            from anyio.from_thread import run as _anyio_run
            from mlx_lm.sample_utils import make_sampler

            caller_predict = get_caller_predict()
            predict_id = id(caller_predict) if caller_predict else None
            flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)
            sampler = make_sampler(temp=float(temperature))

            stream_kwargs: dict[str, Any] = {
                "max_tokens": int(max_tokens),
                "sampler": sampler,
            }
            if logits_processors:
                stream_kwargs["logits_processors"] = logits_processors

            _chunks: list[str] = []
            for _response in mlx_lm.stream_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=flat_prompt,
                **stream_kwargs,
            ):
                _chunks.append(_response.text)
                _chunk = _LocalStreamChunk(
                    text=_response.text, model=self.model, predict_id=predict_id
                )
                _anyio_run(send_stream.send, _chunk)

            text = "".join(_chunks)
        else:
            text, flat_prompt = self._generate(messages, temperature, max_tokens, logits_processors)

        usage = self._compute_usage(flat_prompt, text, max_tokens)
        response = self._build_response(text, usage=usage)

        if cache:
            _cache.put(cache_request, response)

        return response

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Async forward pass.

        Without streaming: delegates to forward() via thread-pool executor,
        gated by a semaphore (default=1).

        With dspy.settings.send_stream: runs _stream_generate_async() and
        sends each _LocalStreamChunk to the stream.
        """
        send_stream = get_send_stream()

        if send_stream is None:
            if self._semaphore is None:
                # Lazy init avoids binding to the wrong event loop at construction time.
                self._semaphore = asyncio.Semaphore(self._max_concurrency)
            async with self._semaphore:
                return await asyncio.to_thread(
                    self.forward, prompt=prompt, messages=messages, **kwargs
                )

        # Streaming path (direct aforward() callers only).
        _cache = get_dspy_cache()
        cache = kwargs.pop("cache", self.cache)

        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        # Clamp to valid ranges (G-1, G-2).
        temperature = max(0.0, min(2.0, float(kwargs.pop("temperature", self._temperature))))
        max_tokens = max(1, int(kwargs.pop("max_tokens", self._max_tokens)))

        response_format = kwargs.pop("response_format", None)
        for _k in ("tools", "num_retries", "stream", "n"):
            kwargs.pop(_k, None)
        if kwargs:
            logger.warning("apple_local: ignoring unsupported kwargs %s", sorted(kwargs))

        schema = _response_format_to_schema(response_format)
        logits_processors: list[Any] | None = None
        if schema is not None:
            proc = self._build_schema_processor(schema)
            if proc is not None:
                logits_processors = [proc]

        flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)
        caller_predict = get_caller_predict()
        predict_id = id(caller_predict) if caller_predict else None

        full_text_parts: list[str] = []
        async for token_text in self._stream_generate_async(
            flat_prompt, temperature, max_tokens, logits_processors
        ):
            full_text_parts.append(token_text)
            chunk = _LocalStreamChunk(
                text=token_text, model=self.model, predict_id=predict_id
            )
            await send_stream.send(chunk)

        full_text = "".join(full_text_parts)
        usage = self._compute_usage(flat_prompt, full_text, max_tokens)
        response = self._build_response(full_text, usage=usage)

        if cache:
            cache_request: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if schema is not None:
                cache_request["response_schema"] = json.dumps(schema, sort_keys=True)
            _cache.put(cache_request, response)

        return response
