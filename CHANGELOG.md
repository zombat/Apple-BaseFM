# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-05-13

### Added

- **`apple_basefm._kv.attention_v2`** — SDPA patch for TurboQuant V2 rotated KV
  cache. `install(R)` patches `mlx.fast.scaled_dot_product_attention` to apply
  `Q @ R` before scoring, so that `(Q·R) @ (K·R)ᵀ = Q @ Kᵀ` exactly.
  `uninstall()` restores the original. `rotated_sdpa_context(R)` wraps both in a
  `try/finally` context manager (the entry point used by `AppleLocalLM`).
  `is_installed()` allows callers to query patch state.

- **`TurboQuantV2Cache.rotation_matrix` property** — exposes the lazily-computed
  rotation matrix so `AppleLocalLM` can retrieve it without `isinstance` checks.

- **`rotated_sdpa_context` auto-installed by `AppleLocalLM`** — both `forward()`
  and `aforward()` retrieve the rotation matrix after `build()` and wrap generation
  calls with `rotated_sdpa_context`. Covers the sync path, the
  `send_stream`-streaming path, and the async streaming path.

- **16 new tests in `tests/test_attention_v2.py`** — install/uninstall lifecycle,
  context manager cleanup on exception, and correctness: `(Q@R) @ (K@R).T == Q @ K.T`
  holds to float32 tolerance.

### Changed

- **`"turboquant-v2"` preset now uses `use_rotation=True`** — QR rotation is
  enabled by default now that `attention_v2` ships. Users who need strict numerical
  equivalence to `mlx-lm --kv-bits 4` should pin `"turboquant-v2-lean"`, which
  permanently stays `use_rotation=False`.

- **`BaseLM` stub (`_compat.py`) now stores `self.cache`** — when DSPy is not
  installed the minimal stub previously discarded the `cache` kwarg, causing
  `AttributeError: 'AppleLocalLM' object has no attribute 'cache'` on every
  `forward()` call. Fixed by persisting `self.cache = kwargs.get("cache", True)`.

- **`use_normalization` deprecation message updated** — no longer references
  "until attention_v2.py ships" since rotation is the shipped outlier-reduction
  path. Normalization remains a future-only no-op.

- **Development Status classifier** updated from `3 - Alpha` to `5 - Production/Stable`.

### Fixed

- **KV test collection crash on Linux** — `import numpy` at module level in
  `tests/test_kv_cache.py` and `tests/fuzz/test_kv_fuzz.py` caused
  `ModuleNotFoundError` at pytest collection time (not a graceful skip). Both files
  now use `pytest.importorskip("numpy")`.

- **Pydantic-gated tests now skip gracefully** — 12 test methods in
  `test_apple_fm.py` and `test_exception_handlers.py` that inline-import `pydantic`
  now use `pytest.importorskip("pydantic")` so they skip cleanly instead of
  crashing when pydantic is not installed.

- **`test_get_send_stream_logs_debug_on_failure` caplog test** — the test now
  patches `compat.DSPY_AVAILABLE = True` so the code path under test is actually
  reached.

---

## [0.3.0] — 2026-05-12

### Added

- **`token_session()` context manager** — accumulates `prompt_tokens`,
  `completion_tokens`, `total_tokens`, and `call_count` across all LM calls inside
  a `with` block, whether made via standalone `lm.forward()` or through DSPy
  `Predict`/`ChainOfThought`. Backed by `contextvars.ContextVar` for thread- and
  async-safety. Supports nesting (each session is isolated) and resuming across
  multiple blocks via an explicit `_SessionAccumulator`. DSPy cache hits are
  intentionally not counted (they have no API cost). Exported from the top-level
  package as `apple_basefm.token_session`.

- **`_SessionAccumulator` dataclass** — mutable token counter with four integer
  fields (`prompt_tokens`, `completion_tokens`, `total_tokens`, `call_count`).
  Yielded by `token_session()` and accepted as the optional `accumulator=` argument
  to resume accumulation across non-contiguous blocks. Exported from the top-level
  package as `apple_basefm._SessionAccumulator`.

- **`lm.usage` per-instance lifetime counter** — every `AppleLocalLM` and
  `AppleFoundationLM` instance now exposes a `usage: _SessionAccumulator` attribute
  that accumulates tokens for the lifetime of the object, independent of any
  `token_session()` block.

- **`lm.reset_usage()`** — resets the per-instance `usage` counter to zero without
  creating a new object.

- **Streaming paths now emit telemetry** — `AppleLocalLM` streaming sync and async
  paths previously silently dropped token counts; they now call `forward_span()` and
  `record_usage()`, bringing them to parity with the non-streaming path.

- **`total_tokens` in structured logs and OTel spans** — `_StructuredFormatter` now
  emits `total_tokens` as an extra field on DEBUG log records; `record_usage()` now
  writes `gen_ai.usage.total_tokens` to OpenTelemetry spans.

- **`apple_basefm/_session.py`** — new module implementing the session accumulation
  system. No new runtime dependencies; uses only stdlib (`dataclasses`, `contextlib`,
  `contextvars`).

### Fixed

- **`_infer_params_b('0B')` returned `0.0` instead of `None`** — the regex matched the
  string `"0B"` and returned `float("0")`, violating the function's contract that any
  returned value is a positive float. Zero-parameter counts now return `None`.

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
