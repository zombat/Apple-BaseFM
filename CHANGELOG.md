# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-11

### Added

- **`AppleFoundationLM`** — DSPy adapter for Apple's on-device Foundation Models
  (macOS 26+, Apple Intelligence system model). Features:
  - Native `@generable` constrained decoding for Pydantic `response_format` models.
  - Native `fm.Tool` wrapping for DSPy tools (no prompt injection needed).
  - `asyncio.wait_for()` timeout on every `session.respond()` call (default 120 s).
  - Guardrail-violation detection and re-raising as `RuntimeError`.
  - LRU-bounded tool-class cache (`_tool_class_cache`, max 256 entries).

- **`AppleLocalLM`** — DSPy adapter for locally-managed Apple Silicon models via
  `mlx-lm`. Features:
  - Any HuggingFace repo ID or local MLX model directory.
  - Structured output via outlines FSM logits processors when `response_format` is set.
  - Async streaming via `_stream_generate_async()` with `threading.Event` cancellation.
  - Configurable concurrency via `max_concurrency` + `asyncio.Semaphore`.
  - `temperature` clamped to `[0.0, 2.0]`; `max_tokens` floored at `1`.

- **`dspy_apple._compat`** — dual-mode shim: uses real `dspy.BaseLM` when DSPy is
  installed; provides a minimal stub when it is not.

- **`dspy_apple._response`** — shared `_FMResponse` / `_FMUsage` / `_FMChoice` /
  `_FMMessage` dataclasses satisfying the `BaseLM` contract including
  `_hidden_params["response_cost"]`.

- **`dspy_apple._base`** — shared base class `_AppleBaseLM` with helpers:
  `_build_response()`, `_flatten_messages()`, `_run_async()`.

- **`dspy_apple._mlx`** — MLX internals: `_MLXMixin`, `_LocalStreamChunk`,
  `_apply_chat_template()`, `_response_format_to_schema()`.

- Optional extras: `[foundation]`, `[mlx]`, `[dspy]`, `[dev]`, `[all]`.

### Security

- `_tool_class_cache` bounded to 256 entries with LRU eviction to prevent unbounded
  memory growth in long-running processes.
- `temperature` and `max_tokens` validated at both `forward()` and `aforward()` entry
  points (not just construction time) to prevent out-of-range values from DSPy
  optimizer sweeps reaching the MLX kernel.
- `_FMResponse._hidden_params` always initialised with `{"response_cost": 0.0}`
  to prevent `KeyError` in DSPy's history aggregator when responses are constructed
  directly.
- `threading.Event` cancel signal in `_stream_generate_async()` prevents GPU thread
  from continuing to run after the async consumer abandons the stream.
