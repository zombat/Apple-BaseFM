# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-05-12

### Added

- **README: HuggingFace mirror guide** — documents `HF_ENDPOINT` usage for networks
  where `huggingface.co` is blocked or slow, covering model downloads, all three
  `apple-basefm` CLI subcommands, shell/Python configuration, and authenticated
  enterprise mirrors.

- **TurboQuant V2 KV cache backend** (`apple_basefm._kv`) — optional KV cache
  compression for `AppleLocalLM`. New `kv_cache` parameter accepts string presets
  (`"turboquant-v2"`, `"turboquant-v2-lean"`) or a custom `KVCacheStrategy` instance.
  - `KVCacheStrategy` — `@runtime_checkable` Protocol; any conforming object is accepted.
  - `TurboQuantV2Cache` — wraps `mlx_lm.models.cache.QuantizedKVCache` (bits ∈ {2, 4, 8},
    configurable `group_size` and `step`). Optional QR rotation distributes outliers
    before quantization for lower error at the same bit width.
  - `make_rotation_matrix(head_dim)` — deterministic QR rotation via `mx.linalg.qr`;
    seeded at 42, lazy-cached on the `TurboQuantV2Cache` instance across `forward()` calls.
  - Achieves ~3.6× KV memory reduction at 4-bit (969 MB → 266 MB at T=8192 context).
    Generation speed is effectively unchanged; benefit is sustained throughput as context
    grows in DSPy optimizer loops.

- **README: Apple Silicon memory guide** — per-RAM-tier model recommendations with and
  without TurboQuant V2; stack comparison (mlx-lm raw vs. TurboQuant V2) covering
  generation speed and memory overhead; practical guidance for DSPy optimizer workloads.

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

- **`apple_basefm._compat`** — dual-mode shim: uses real `dspy.BaseLM` when DSPy is
  installed; provides a minimal stub when it is not.

- **`apple_basefm._response`** — shared `_FMResponse` / `_FMUsage` / `_FMChoice` /
  `_FMMessage` dataclasses satisfying the `BaseLM` contract including
  `_hidden_params["response_cost"]`.

- **`apple_basefm._base`** — shared base class `_AppleBaseLM` with helpers:
  `_build_response()`, `_flatten_messages()`, `_run_async()`.

- **`apple_basefm._mlx`** — MLX internals: `_MLXMixin`, `_LocalStreamChunk`,
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
