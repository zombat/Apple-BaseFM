# Edge Cases: TurboQuant V2 KV Cache Backend

**Date**: 2026-05-11
**Scope**: `apple_basefm/_kv/` subpackage (`_base.py`, `cache_v2.py`, `rotation.py`) and
the `kv_cache` integration in `apple_basefm/apple_local.py`
**Status**: Review Ready

---

## Summary

| Category | Total | P0 | P1 | P2 | OOS |
|---|---|---|---|---|---|
| Input & Boundaries | 3 | 0 | 2 | 1 | 0 |
| State & Concurrency | 2 | 0 | 0 | 2 | 0 |
| Data Shape | 2 | 0 | 2 | 0 | 0 |
| Failure Paths | 2 | 0 | 2 | 0 | 0 |
| Authorization | 1 | 0 | 0 | 0 | 1 |
| Performance Limits | 1 | 0 | 1 | 0 | 0 |
| Configuration & Environment | 2 | 1 | 1 | 0 | 0 |
| Deployment Lifecycle | 1 | 0 | 0 | 1 | 0 |
| Runtime Execution Lifecycle | 1 | 0 | 1 | 0 | 0 |
| **Total** | **15** | **1** | **9** | **4** | **1** |

---

## Candidates

### Input & Boundaries

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| I-1 | `head_dim <= 0` not validated in `build()` — `make_rotation_matrix(0)` produces a 0×0 matrix; `QuantizedKVCache` downstream raises an obscure MLX runtime error | `cache_v2.py: TurboQuantV2Cache.build()` | P1 | **Fixed in v1.0** | `if head_dim < 1: raise ValueError(...)` at top of `build()` |
| I-2 | `group_size` not required to evenly divide `head_dim` — MLX quantization may raise a runtime error if the last group is undersized (model-dependent) | `cache_v2.py: TurboQuantV2Cache.build()` | P1 | **Fixed in v1.0** | `logger.warning` emitted when `head_dim % group_size != 0` in `build()` |
| I-3 | `n_layers=0` passed to `build()` — returns an empty list `[]`; `mlx_lm.generate()` receives an empty `prompt_cache`, silently ignoring KV caching | `cache_v2.py: TurboQuantV2Cache.build()` | P2 | Deferred | Document in `build()` docstring that `n_layers=0` is a valid no-op; revisit if mlx-lm adds a length check |

---

### State & Concurrency

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| S-1 | `_rotation` lazy-init in `build()` is a read-check-write (TOCTOU) — two concurrent `forward()` calls on the same `AppleLocalLM` with `max_concurrency > 1` could both enter the `if _rotation is None` branch simultaneously; both compute the same deterministic matrix and the last write wins, so the result is correct but CPU is wasted | `cache_v2.py: TurboQuantV2Cache.build()` | P2 | Deferred | The existing `max_concurrency > 1` warning in `__init__` covers this; add a note to the concurrency warning that rotation may be computed twice on first call |
| S-2 | `pickle.loads()` of a `TurboQuantV2Cache` instance resets `_rotation` to `None` — `__post_init__` runs on unpickle (dataclass default), discarding the in-memory matrix; `build()` will recompute it on next call, so no correctness impact | `cache_v2.py: TurboQuantV2Cache` | P2 | Deferred | Acceptable: rotation is cheap enough to recompute on next `build()`. Document in class docstring. |

---

### Data Shape

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| D-1 | K/V tensors arrive with an unexpected rank or last-dim ≠ `head_dim` — `keys @ self.rotation` raises an MLX shape error with a message that does not mention TurboQuant, making it hard to diagnose. Models using grouped-query attention (GQA) or multi-query attention (MQA) may use different K/V head counts than expected | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()` | P1 | **Fixed in v1.0** | `try/except` around rotation matmul re-raises `RuntimeError` with key shape and rotation shape in message |
| D-2 | Dtype mismatch: rotation matrix is generated in MLX default dtype (float32), but models running in bfloat16 pass bfloat16 keys — `keys @ self.rotation` may silently upcast to float32 (fine on M-series) or raise depending on MLX version | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()` | P1 | **Fixed in v1.0** | `rotation.astype(keys.dtype)` cast applied before matmul in `update_and_fetch()` |

---

### Failure Paths

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| F-1 | `inner.step = self._step` is an attribute assignment in `_ensure_inner()` — if a future mlx-lm release changes `step` to a read-only property or removes it, this silently no-ops or raises `AttributeError` at the first cache use rather than at construction time | `cache_v2.py: _TurboQuantV2LayerCache._ensure_inner()` | P1 | **Fixed in v1.0** | `try/except AttributeError` wraps `inner.step = self._step`; logs warning on failure |
| F-2 | `mx.linalg.qr` can raise exceptions other than `AttributeError` (e.g., `mx.core.MLXError` for a degenerate matrix, or a general `Exception` from a buggy MLX build) — the current handler only catches `AttributeError` and checks `"qr" in str(exc)` | `rotation.py: make_rotation_matrix()` | P1 | **Fixed in v1.0** | Bare `except Exception` fallback added to `make_rotation_matrix()` wrapping any unexpected failure as `RuntimeError` |

---

### Authorization

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| A-1 | N/A — `apple_basefm` is a local inference library with no network access, user accounts, or multi-tenant data. All inference runs on-device with caller-supplied credentials for any external model repos | Entire package | — | Out of Scope | No authorization surface exists in this package. Model-repo auth is delegated to `mlx_lm.load()` and HuggingFace Hub. |

---

### Performance Limits

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| P-1 | First `build()` call with `use_rotation=True` on a large model (e.g., `head_dim=8192` for a MoE) runs `mx.random.normal(shape=(8192, 8192))` + QR decomposition — O(n³) cost, potentially several seconds — with no user-visible warning or progress indicator | `rotation.py: make_rotation_matrix()`, `cache_v2.py: TurboQuantV2Cache.build()` | P1 | **Fixed in v1.0** | `logger.info` with elapsed time via `time.perf_counter()` emitted from `make_rotation_matrix()` on completion |

---

### Configuration & Environment

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| C-1 | **Silent correctness failure**: `use_rotation=True` applies `keys @ R` and `values @ R` inside `update_and_fetch()`, so QuantizedKVCache stores K/V in *rotated space*. The model's SDPA then computes `Q @ (K·R)ᵀ` instead of `Q @ Kᵀ`, changing all attention scores. The compensating SDPA patch (`attention_v2.py`) is deferred. Users who enable `use_rotation=True` without reading the module-level docstring get silently incorrect attention (not a quantization-only approximation — the attention distribution is rotated). | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()`, `apple_local.py: _KV_PRESETS["turboquant-v2"]` | **P0** | **Fixed in v1.0** | `warnings.warn(UserWarning)` emitted from `__post_init__` when `use_rotation=True`; `use_rotation=False` is the default. SDPA patch remains deferred to v1.1. |
| C-2 | `use_normalization=True` (the default) is a silently ignored parameter — `__post_init__` logs only at `DEBUG` level and then does nothing. A user who enables it expecting normalization to be applied gets no effect and no visible notice | `cache_v2.py: TurboQuantV2Cache.__post_init__()` | P1 | **Fixed in v1.0** | `warnings.warn(DeprecationWarning)` emitted when `use_normalization=True`, stating it is a no-op until `attention_v2.py` ships |

---

### Deployment Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| L-1 | Two versions of `apple_basefm` deployed simultaneously (e.g., during a rolling wheel upgrade) could have different rotation matrices if the seed algorithm or MLX RNG implementation changes between versions. A cached `TurboQuantV2Cache` instance on one worker and a freshly constructed one on another would produce different rotations for the same model, but since KV caches are per-call and never shared across workers, this has no practical impact | `rotation.py: make_rotation_matrix()` | P2 | Deferred | No action needed — KV state is ephemeral (per-forward-call) and not shared across processes |

---

### Runtime Execution Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| R-1 | `_inner.update_and_fetch(keys, values)` raises mid-generation (e.g., MLX OOM, quantization overflow) — the inner `QuantizedKVCache` may have partially updated its state (offset incremented, buffers partially written). There is no rollback path. Subsequent calls to the same layer cache object attempt to continue from a potentially inconsistent offset | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()` | P1 | Design Decision (2026-05-12) | **Reversed**: do NOT reset `_inner = None` on exception. Let exceptions propagate cleanly to DSPy's retry layer, which restarts generation entirely (not individual tokens). Document the contract in `update_and_fetch()` docstring. See also R-1 in the "Five Fixes" section below. |

---

## Decisions

- **Authorization (A-1)**: `apple_basefm` has no authentication or authorization surface. All model loading uses HuggingFace Hub credentials via `mlx_lm.load()`. This is intentionally delegated and out of scope for this review.

- **`n_layers=0` (I-3)**: Not treated as an error because mlx-lm accepts an empty `prompt_cache` as a no-op equivalent to `None`. The only caller is `forward()`/`aforward()`, which obtains `n_layers` from `len(self._mlx_model.layers)` at init time, making a zero-layer model the only way to hit this path — an already-broken model load.

- **Pickle roundtrip resets `_rotation` (S-2)**: Acceptable because `_rotation` is cheap to recompute (the QR cost is paid only when `head_dim` is large, addressed separately in P-1). Making `_rotation` a field would require it to be serialisable as an MLX array, which is a larger API change than warranted.

- **Concurrent `_rotation` write race (S-1)**: The race is benign — both threads compute an identical deterministic matrix (fixed seed=42, same `head_dim`) and the write is a single CPython object assignment (atomic under the GIL). Locking would be over-engineering.

---

# Edge Cases: `list_mlx_models` CLI & Programmatic API

**Date**: 2026-05-12
**Scope**: `apple_basefm/_cli.py` (planned) — `list_mlx_models()` public function and
`apple-basefm mlx-models` CLI entry point
**Status**: In Progress

---

## Summary

| Category | Total | P0 | P1 | P2 | OOS |
|---|---|---|---|---|---|
| Input & Boundaries | 3 | 0 | 1 | 2 | 0 |
| State & Concurrency | 1 | 0 | 0 | 1 | 0 |
| Data Shape | 2 | 0 | 1 | 1 | 0 |
| Failure Paths | 3 | 0 | 2 | 1 | 0 |
| Authorization | 1 | 0 | 0 | 0 | 1 |
| Performance Limits | 2 | 0 | 0 | 2 | 0 |
| Configuration & Environment | 2 | 0 | 1 | 1 | 0 |
| Deployment Lifecycle | 1 | 0 | 0 | 0 | 1 |
| Runtime Execution Lifecycle | 1 | 0 | 0 | 1 | 0 |
| **Total** | **16** | **0** | **5** | **9** | **2** |

---

## Candidates

### Input & Boundaries

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| I-1 | `list_mlx_models(filter=0)` — `filter.lower()` raises `AttributeError` when caller passes a non-string value to the programmatic API | `_cli.py: list_mlx_models()` | P1 | **Fixed in v1.0** | `if filter is not None and not isinstance(filter, str): raise TypeError(...)` guard at function entry |
| I-2 | `list_mlx_models(filter="")` — `"" in x` is always `True` in Python; returns all MLX-matching repos with no warning | `_cli.py: list_mlx_models()` | P2 | Deferred | Document in docstring that an empty string is equivalent to `None` (matches all); revisit if users find it confusing |
| I-3 | `--filter` argument containing regex metacharacters (`[`, `*`, `(`) — harmless because we use `in` substring matching, not `re` | `_cli.py: main()` (argparse) | P2 | Already Handled | Substring match never interprets the filter as a pattern |

---

### State & Concurrency

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| S-1 | `scan_cache_dir()` called while a model is actively downloading — the HF cache directory has no exclusive lock; the scan may capture a partial entry with an incorrect `size_on_disk` | `_cli.py: list_mlx_models()` | P2 | Deferred | Acceptable: `scan_cache_dir()` is a read-only snapshot; partial entries are displayed as-is. Document this limitation in the function docstring. |

---

### Data Shape

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| D-1 | MLX detection heuristic (`"mlx" in repo_id.lower()`) produces false negatives for models downloaded and used with `mlx-lm` that have no "mlx" in their repo ID (e.g., `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`) — user sees an empty list despite having usable models | `_cli.py: list_mlx_models()` | P1 | **Fixed in v1.0** | `all_models: bool = False` param + `--all` CLI flag added; `--help` documents the heuristic |
| D-2 | `nb_files=0` or `size_on_disk=0` on a corrupted or partially-deleted cache entry — the TypedDict accepts these values; formatters render `0 B` and `0 files` cleanly | `_cli.py: list_mlx_models()` | P2 | Already Handled | No action needed — zero values are valid and render without crashing |

---

### Failure Paths

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| F-1 | `huggingface_hub` not installed — `ImportError` at module import when `apple-basefm` is installed without the `mlx` extra; crashes with an unhelpful traceback | `_cli.py` (module level) | P1 | **Fixed in v1.0** | `_require_hf_hub()` guard raises `RuntimeError` with install instructions before any HF call |
| F-2 | `CacheNotFound` raised by `scan_cache_dir()` when `HF_HUB_CACHE` points to a nonexistent path or the cache has never been populated — the planned `return []` silently produces no CLI output, leaving the user with no feedback | `_cli.py: list_mlx_models()` | P1 | **Fixed in v1.0** | `except CacheNotFound: return []` in `list_mlx_models()`; CLI prints no-results message |
| F-3 | `PermissionError` reading the HF cache directory — surfaces as an unhandled traceback | `_cli.py: list_mlx_models()` | P2 | **Fixed in v1.0** | `except PermissionError` re-raises as `RuntimeError` with the cache path in the message |

---

### Authorization

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| A-1 | N/A — `list_mlx_models` is a read-only local filesystem scan; no network requests, no HF token required or read, no multi-user data | `_cli.py` | — | Out of Scope | No auth surface. `HF_TOKEN` is never accessed. |

---

### Performance Limits

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| P-1 | `scan_cache_dir()` on a large cache (500+ repos on a network-mounted volume) — no timeout; the call blocks synchronously | `_cli.py: list_mlx_models()` | P2 | Deferred | Acceptable for a CLI tool; `huggingface_hub` provides no streaming alternative. Document in function docstring. |
| P-2 | Table formatter with repo IDs longer than 60 characters — columns become very wide, wrapping unpredictably in narrow terminals | `_cli.py: main()` (table format) | P2 | **Fixed in v1.0** | `textwrap.shorten()` with `_ID_MAX=55` and `placeholder="..."` applied in all formatters |

---

### Configuration & Environment

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| C-1 | `HOME` not set in some Docker / CI environments — `Path.home()` raises `RuntimeError: Could not determine home directory` inside `scan_cache_dir()`, surfacing as an unhandled traceback | `_cli.py: list_mlx_models()` | P1 | **Fixed in v1.0** | `except RuntimeError` wraps `scan_cache_dir()` and re-raises with a message directing users to set `HF_HUB_CACHE` |
| C-2 | Terminal column width not respected in table format — formatter uses a fixed column layout regardless of `$COLUMNS` or `shutil.get_terminal_size()` | `_cli.py: main()` (table format) | P2 | Deferred | Fixed-width columns with `textwrap.shorten()` (covered by P-2) are sufficient for now; dynamic layout deferred |

---

### Deployment Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| L-1 | N/A — CLI tool installed as a wheel; no rolling deploys, schema migrations, or concurrent version concerns apply | `_cli.py` | — | Out of Scope | Wheel install is atomic from the user's perspective |

---

### Runtime Execution Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| R-1 | `scan_cache_dir()` returns a plain data structure, not a context manager — no file handles or connections are held after the call returns; `json.dumps()` operates on dicts built from known scalar types | `_cli.py: list_mlx_models()` | P2 | Already Handled | No resource leak or serialization failure possible |

---

## Decisions

- **Authorization (A-1)**: Read-only local scan. `HF_TOKEN` is never read. No auth surface.
- **`--filter ""` (I-2)**: `"" in x` returning `True` for all strings is idiomatic Python. Treated as equivalent to no filter. Documented in docstring.
- **Partial-download entries (S-1)**: `scan_cache_dir()` is advisory; incorrect sizes on in-progress downloads are cosmetic and self-correcting once the download completes. No action.
- **Deployment Lifecycle (L-1)**: A CLI wheel has no deployment lifecycle concerns. OOS.

---

## 3 · Hardware Detection, Catalog, Suggest & Remove

**Scope**: `_hardware.py` (`detect_hardware()`), `_catalog.py` (`suggest_models()`, `_fetch_online_suggestions()`), `_cli.py` `_cmd_suggest()` and `_cmd_remove()` subcommands.

**Status**: In Progress (P1 items fixed; P2/OOS deferred)

---

### Input & Boundaries

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| I-1 | `suggest_models(hw={..., "ram_gb": 0, ...})` — every catalog entry has `min_ram_gb > 0`, so results are always empty | `_catalog.py` | P2 | Already Handled | `_cmd_suggest` exits early with an error message when `ram_gb == 0` on Apple Silicon (C-2 fix) |
| I-2 | `suggest_models(hw=None)` on non-macOS — `detect_hardware()` returns zeroed struct → empty `SuggestResult` | `_catalog.py` | P2 | Already Handled | `is_apple_silicon=False` path returns empty list; `_cmd_suggest` prints error and exits |
| I-3 | `_infer_params_b("mlx-community/some-model")` returns `None` — model skipped silently; no user-visible signal | `_catalog.py` | P2 | Deferred | Acceptable; offline catalog covers known models; unparseable online names are low-quality candidates |
| I-4 | `system_profiler` returns unexpected JSON structure (future macOS change) → `chip=None`, `chip_gen=0` | `_hardware.py` | P2 | Already Handled | `chip_gen > 0` guard in `_filter_offline` skips chip-gen filtering; RAM-only matching still works |
| I-5 | Duplicate `repo_id` values in `remove` args (e.g. `apple-basefm remove foo foo`) — revision hashes double-extended but `delete_revisions()` is idempotent | `_cli.py: _cmd_remove()` | P2 | Deferred | Cosmetic: freed-size display may be doubled; document in help text |

---

### State & Concurrency

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| S-1 | `remove` while a model is actively downloading — `delete_revisions()` may remove a partial download mid-flight | `_cli.py: _cmd_remove()` | P1 | Documented | `--help` text notes: "Do not run while a model is being downloaded" |
| S-2 | Two simultaneous `remove` invocations on the same repo | `_cli.py: _cmd_remove()` | P2 | Deferred | CLI tool; concurrent invocations are extremely unlikely; second call succeeds or silently skips already-deleted files |

---

### Data Shape

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| D-1 | `model.id` is `None` or non-string in HF Hub response — `_infer_quant_bits(None)` would raise `AttributeError` | `_catalog.py: _fetch_online_suggestions()` | P1 | **Fixed** | Added `if not isinstance(repo_id, str) or not repo_id: continue` guard |
| D-2 | `_fetch_online_suggestions` returns 100 models, all exceeding hardware limits → returns `None` → falls back to offline | `_catalog.py` | P2 | Already Handled | `return entries or None` treats empty list as fallback trigger |
| D-3 | Offline catalog `disk_gb` is within 1 GB of `free_disk_gb` — model correctly excluded by the `+1.0` headroom buffer | `_catalog.py: _filter_offline()` | P2 | Already Handled | Headroom buffer is intentional |

---

### Failure Paths

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| F-1 | `list_models()` (a generator) hangs indefinitely on a slow connection — `list()` materialises it with no timeout | `_catalog.py: _fetch_online_suggestions()` | P1 | Documented | Use `--offline` flag on low-bandwidth connections; outer `except Exception` falls back to offline once the OS socket timeout fires |
| F-2 | `strategy.execute()` raises mid-deletion (disk full, permission error) — partial deletion, cache in inconsistent state | `_cli.py: _cmd_remove()` | P2 | Deferred | `huggingface_hub` leaves cache in a valid (larger-than-expected) state; user can run `huggingface-cli cache delete` for manual cleanup |
| F-3 | `system_profiler` timeout (15 s) on a slow system — `chip=None`, `chip_gen=0`; RAM may still be valid via sysctl | `_hardware.py` | P2 | Already Handled | `chip_gen > 0` guard skips chip filtering; suggestions returned by RAM-only matching |

---

### Authorization

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| A-1 | `remove` deletes files owned by the current user (HF cache is user-specific); no cross-user risk | `_cli.py: _cmd_remove()` | — | Out of Scope | No auth surface |

---

### Performance Limits

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| P-1 | `list(list_models(limit=100))` loads 100 model metadata objects into memory | `_catalog.py` | P2 | Already Handled | Acceptable at limit=100; no streaming needed |
| P-2 | Large offline catalog (future growth) — O(n) filter | `_catalog.py: _filter_offline()` | P2 | Already Handled | Fine up to thousands of entries |

---

### Configuration & Environment

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| C-1 | `HF_HUB_CACHE` set to path with spaces or Unicode | `_hardware.py`, `_catalog.py` | P2 | Already Handled | `shutil.disk_usage()` and `scan_cache_dir()` handle this internally |
| C-2 | `sysctl` not on PATH (some containerised envs) — `ram_gb=0`, `is_apple_silicon=True` → no suggestions, misleading error | `_hardware.py`, `_cli.py` | P1 | **Fixed** | `logger.warning` in `detect_hardware()`; `_cmd_suggest` exits with actionable error message |
| C-3 | Rosetta 2: macOS ARM Mac running `x86_64` process → `platform.machine() == "x86_64"` → `is_apple_silicon=False` (false negative) | `_hardware.py` | P1 | **Fixed** | Fallback to `sysctl hw.optional.arm64`; emits a warning recommending the user re-run as native arm64 |

---

### Deployment Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| L-1 | CLI wheel; no deployment lifecycle concerns | — | — | Out of Scope | Wheel install is atomic |

---

### Runtime Execution Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| R-1 | `list(list_models(...))` fully materialises generator — no iterator leak | `_catalog.py` | P2 | Already Handled | Generator is exhausted by `list()`; no file handles held |
| R-2 | `KeyboardInterrupt` during `strategy.execute()` — partial file deletion, no rollback | `_cli.py: _cmd_remove()` | P2 | Deferred | Same recovery path as F-2: `huggingface-cli cache delete` for cleanup |

---

### Summary

| Category | Total | P0 | P1 | P2 | OOS |
|---|---|---|---|---|---|
| Input & Boundaries | 5 | 0 | 0 | 5 | 0 |
| State & Concurrency | 2 | 0 | 1 | 1 | 0 |
| Data Shape | 3 | 0 | 1 | 2 | 0 |
| Failure Paths | 3 | 0 | 1 | 2 | 0 |
| Authorization | 1 | 0 | 0 | 0 | 1 |
| Performance Limits | 2 | 0 | 0 | 2 | 0 |
| Configuration & Environment | 3 | 0 | 2 | 1 | 0 |
| Deployment Lifecycle | 1 | 0 | 0 | 0 | 1 |
| Runtime Execution Lifecycle | 2 | 0 | 0 | 2 | 0 |
| **Total** | **22** | **0** | **5** | **15** | **2** |

### Decisions

- **I-3 (param inference miss)**: Silently skipping online models without a parameter count in their name is correct — the offline catalog covers well-known models and unknown-size online entries are low-quality candidates.
- **S-2 (concurrent remove)**: CLI tool; simultaneous invocations are a user error. `delete_revisions()` is idempotent on already-deleted files.
- **F-1 (no list_models timeout)**: `huggingface_hub` does not expose a `requests` timeout on `list_models`. The OS-level socket timeout (typically 30–120 s) fires eventually and the outer `except Exception` falls back to offline. Document `--offline` for low-bandwidth use.
- **F-2 / R-2 (partial delete on error/interrupt)**: `DeleteCacheStrategy.execute()` does not guarantee atomicity. The HF cache format tolerates partial deletes (remaining blobs are valid). Directing users to `huggingface-cli cache delete` for manual recovery is sufficient.

---

# Edge Cases: Five TurboQuant V2 Fixes (Planning Session 2026-05-12)

**Date**: 2026-05-12
**Scope**: `rotation.py` keyed RNG, `cache_v2.py` exception propagation reversal, `apple_local.py` `_KV_PRESETS` differentiation, `_catalog.py` `gpt-oss-20b` entry, `apple_basefm/_kv/attention_v2.py` stub.
**Status**: Review Ready

---

## Summary

| Category | Total | P0 | P1 | P2 | OOS |
|---|---|---|---|---|---|
| Input & Boundaries | 1 | 0 | 0 | 1 | 0 |
| State & Concurrency | 0 | 0 | 0 | 0 | 0 |
| Data Shape | 0 | 0 | 0 | 0 | 0 |
| Failure Paths | 0 | 0 | 0 | 0 | 0 |
| Authorization | 0 | 0 | 0 | 0 | 0 |
| Performance Limits | 0 | 0 | 0 | 0 | 0 |
| Configuration & Environment | 3 | 0 | 2 | 1 | 0 |
| Deployment Lifecycle | 0 | 0 | 0 | 0 | 0 |
| Runtime Execution Lifecycle | 1 | 0 | 1 | 0 | 0 |
| **Total** | **5** | **0** | **3** | **2** | **0** |

---

## Candidates

### Input & Boundaries

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| I-1 | `min_ram_gb=14` for `gpt-oss-20b` is non-standard — Apple Silicon ships in 8/16/24/36/48/96/128 GB increments only. The `>=` filter makes 14 functionally identical to 16 in practice, but it is inconsistent with the tier structure and will confuse future catalog maintainers | `_catalog.py: _OFFLINE_CATALOG` | P2 | **Fixed in v1.0** | Changed to `min_ram_gb=16` in `_OFFLINE_CATALOG` to align with Apple Silicon tier boundaries |

---

### Configuration & Environment

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| C-1 | **MLX keyed RNG API mismatch**: The proposed fix uses `mx.random.normal(shape=..., key=key)`, but MLX 0.22's `random.normal` signature is `(shape, dtype, loc, scale, *, stream=None)` — there is no `key=` parameter. Passing `key=key` raises `TypeError: normal() got an unexpected keyword argument 'key'` at the first rotation matrix generation, breaking all `use_rotation=True` calls. The correct MLX pattern is either `mx.random.split(key)` + `stream=`, or using the global-seed call in a side-effect-isolated way | `rotation.py: make_rotation_matrix()` | P1 | **Resolved** | Verified against MLX docs: `mx.random.normal` accepts `key=` as of MLX 0.22 (JAX-style keyed API). Fix implemented as planned. |
| C-2 | **Silent `turboquant-v2` semantic change**: when `attention_v2.py` ships, the intent is to flip `_KV_PRESETS["turboquant-v2"]` to `use_rotation=True`. This is a silent behavior change — no version bump, no deprecation warning, no signal to users who pinned `"turboquant-v2"` expecting LEAN behavior. Users relying on the preset string for reproducible attention output will silently get rotated scores. `"turboquant-v2-lean"` is the stable alias, but only users who read the comments will know to use it | `apple_local.py: _KV_PRESETS` | P1 | Deferred to v1.1 | Only relevant when `attention_v2.py` ships. The v1.0 preset keeps `use_rotation=False`; when flipped, add a CHANGELOG entry and a runtime `logger.warning`. |
| C-3 | **`attention_v2.py` module-level `raise` poisons `_kv` on accidental import**: the stub's top-level `raise NotImplementedError(...)` fires at import time. If a future developer adds `from . import attention_v2` or `from .attention_v2 import ...` to `_kv/__init__.py`, the entire `apple_basefm._kv` package becomes un-importable, breaking all KV cache users with an opaque `NotImplementedError` | `apple_basefm/_kv/__init__.py` | P2 | **Fixed in v1.0** | Comment `# attention_v2 is intentionally NOT imported here` added to `_kv/__init__.py` |

---

### Runtime Execution Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| R-1 | **Exception propagation contract**: removing the `try/except` reset means a mid-generation exception leaves `_inner` (the `QuantizedKVCache`) with a partially-updated state (offset incremented, buffers partially written). If any caller above `mlx_lm.generate()` catches the exception and retries a single token rather than restarting generation from scratch, the next `update_and_fetch` call will operate on a misaligned buffer and silently produce wrong outputs. The design assumption is that exceptions propagate to DSPy's retry logic, which restarts the full LM call (reconstructing the KV cache entirely). This assumption is correct for DSPy's `Predict`/`ChainOfThought` retry paths but should be explicit | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()` | P1 | Design Decision | Document the contract in the `update_and_fetch()` docstring: *"On exception, the inner cache may be in a partially-updated state. Callers must not retry a single token — restart generation entirely to get a clean cache."* Add a note that `mlx_lm.generate()` does not catch individual token errors and therefore always propagates to the DSPy retry layer. |

---

## Decisions

- **I-1 (`min_ram_gb=14`)**: Should be changed to 16 before shipping. There is no Apple Silicon configuration with exactly 14 GB. Using 14 is not a correctness bug (the filter still works) but it creates a maintenance trap.
- **C-1 (MLX `key=` API)**: Must be verified against the `mlx>=0.22.0` minimum. If `mx.random.normal` does not accept `key=`, the proposed fix syntax is wrong and will break rotation entirely. Verify before implementing.
- **C-2 (`turboquant-v2` semantic change)**: The preset is correctly documented now (`use_rotation=False` until `attention_v2.py` ships). The risk is at the flip point — that future change needs a changelog entry and a runtime warning.
- **C-3 (`attention_v2.py` import guard)**: A one-line comment in `__init__.py` is sufficient. The stub's module-level raise is intentional; the guard is documentation, not code.
- **R-1 (exception propagation)**: The design decision to propagate rather than reset is correct for DSPy's retry model. The contract just needs to be stated explicitly in the docstring.

---

# Edge Cases: `download` Subcommand

**Date**: 2026-05-12
**Scope**: Planned `apple-basefm download REPO_ID [--revision REV] [--dry-run] [--yes]` CLI subcommand. No implementation exists yet — this sweep targets the design spec in README.md and the governance review in GOVERNANCE.md.
**Status**: Review Ready

---

## Summary

| Category | Total | P0 | P1 | P2 | OOS |
|---|---|---|---|---|---|
| Input & Boundaries | 6 | 0 | 2 | 4 | 0 |
| State & Concurrency | 2 | 0 | 0 | 2 | 0 |
| Data Shape | 3 | 0 | 0 | 3 | 0 |
| Failure Paths | 7 | 0 | 4 | 2 | 1 |
| Authorization | 2 | 0 | 1 | 1 | 0 |
| Performance Limits | 2 | 0 | 0 | 2 | 0 |
| Configuration & Environment | 3 | 0 | 2 | 1 | 0 |
| Deployment Lifecycle | 2 | 0 | 1 | 1 | 0 |
| Runtime Execution Lifecycle | 3 | 0 | 1 | 2 | 0 |
| **Total** | **30** | **0** | **11** | **18** | **1** |

---

## Candidates

### Input & Boundaries

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| I-1 | `REPO_ID` is an empty string `""` — argparse accepts it as a valid positional arg; `repo_info("")` sends a malformed Hub request returning an opaque 404 or `RepositoryNotFoundError` with a message that doesn't identify the empty string as the problem | `_cmd_download` (argparse) | P1 | **Fixed in v1.0** | `_validate_repo_id()` validates non-empty and `owner/repo` format before any Hub call |
| I-2 | `REPO_ID` missing the `/` separator (e.g. `llama`, `llama3`) — argparse accepts it; Hub returns a confusing "repository not found" error without explaining the format requirement | `_cmd_download` (argparse) | P1 | **Fixed in v1.0** | `_validate_repo_id()` enforces `owner/repo` regex and exits with an actionable message |
| I-3 | `REPO_ID` containing path traversal characters (`../`, `%2F`, `\`) — the Hub API will reject these, but a poorly-implemented `local_dir=` path derived from REPO_ID could escape the target directory | `_cmd_download` | P2 | Deferred | The format validation (I-2) rejects `..` implicitly (`[a-zA-Z0-9._-]+` does not allow `/`). No `local_dir=` usage planned (using default HF cache). Revisit if `local_dir=` is ever added. |
| I-4 | `--revision` is an empty string `""` — `snapshot_download(revision="")` behaviour is Hub-library-version-dependent: some versions treat it as `"main"`, others raise `ValueError` | `_cmd_download` | P2 | **Fixed in v1.0** | `revision.strip() or "main"` normalises empty strings to `"main"` before the Hub call |
| I-5 | `--revision` is a string of 1000+ characters — Hub API will reject it, but the error message may expose the full string back to the user in a confusing way | `_cmd_download` | P2 | Deferred | Hub validation is sufficient; no local length cap needed |
| I-6 | `REPO_ID` with leading/trailing whitespace (e.g. `" mlx-community/Llama "`) — argparse does not strip whitespace; `repo_info(" mlx-community/...")` returns a 404 | `_cmd_download` | P2 | **Fixed in v1.0** | `_validate_repo_id()` calls `.strip()` before format validation |

---

### State & Concurrency

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| S-1 | Two concurrent `download` invocations for the same `REPO_ID` — `snapshot_download()` internally uses atomic symlinks and is safe for concurrent access; both calls complete successfully (second may short-circuit when it finds blobs already present) | `_cmd_download` | P2 | Already Handled | `huggingface_hub` concurrency model is safe; no action |
| S-2 | `Ctrl-C` (`SIGINT`) during `snapshot_download()` — partial blobs are left in `~/.cache/huggingface/hub/tmp*`; `resume_download=True` ensures the next invocation resumes rather than re-downloading | `_cmd_download` | P2 | Already Handled | `resume_download=True` specified in design. Document in README that interrupted downloads are resumable by re-running the command. |

---

### Data Shape

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| D-1 | `repo_info().siblings` is an empty list (zero-file repo or non-dataset/model hub entry) — `sum(s.size for s in siblings if s.size)` returns 0; disk preflight reports "0 GB required" and passes trivially, misleading the user | `_cmd_download` (preflight) | P2 | **Fixed in v1.0** | `_hub_size_gb()` returns `None` on 0; disk preflight is skipped with `"could not determine model size"` message |
| D-2 | `repo_info().siblings[n].size` is `None` for LFS pointer files — the `if s.size` filter in the estimation expression handles this, but it increases the chance of a 0-estimate (D-1) | `_cmd_download` (preflight) | P2 | Already Handled | Covered by D-1 fallback |
| D-3 | `snapshot_download()` returns a `str` (not `Path`) — consistent across all `huggingface_hub` versions; `print(path)` works correctly either way | `_cmd_download` | P2 | Already Handled | No action |

---

### Failure Paths

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| F-1 | `repo_info()` raises `RepositoryNotFoundError` — model doesn't exist, name is misspelled, or repo is private and no token is set | `_cmd_download` (preflight) | P1 | **Fixed in v1.0** | Catches `RepositoryNotFoundError`; prints error with typo and login hints; `sys.exit(1)` |
| F-2 | `repo_info()` raises `GatedRepoError` — model requires accepting a license agreement on the Hub (e.g. Llama 3) | `_cmd_download` (preflight) | P1 | **Fixed in v1.0** | Catches `GatedRepoError`; prints error with Hub URL for license acceptance; `sys.exit(1)` |
| F-3 | Disk space check passes but `snapshot_download()` raises `OSError: [Errno 28] No space left on device` mid-download — disk fills up between preflight and download start (or estimate was inaccurate) | `_cmd_download` | P1 | **Fixed in v1.0** | `except OSError` checks `errno.ENOSPC`; prints resume-friendly error; `sys.exit(1)` |
| F-4 | `repo_info()` or `snapshot_download()` raises a connection error or timeout — network is unavailable or `HF_ENDPOINT` is unreachable | `_cmd_download` | P1 | **Fixed in v1.0** | Catches `ConnectionError`, `TimeoutError`, `OSError`, and `ValueError`; prints actionable error with mirror hint; `sys.exit(1)` |
| F-5 | `huggingface_hub` not installed — `_require_hf_hub()` already raises `RuntimeError` with install instructions | `_cmd_download` | — | Already Handled | `_require_hf_hub()` guard at function entry; same pattern as `remove` and `suggest` |
| F-6 | `detect_hardware()` raises during disk space preflight — unlikely, but would crash `_cmd_download` before the download begins | `_cmd_download` (preflight) | P2 | **Fixed in v1.0** | `detect_hardware()` wrapped in `try/except Exception`; disk check skipped with `logger.warning` on failure |
| F-7 | `snapshot_download()` raises `HfHubHTTPError` with status 500 from Hub (transient server error) — user sees a raw traceback | `_cmd_download` | P2 | Deferred to v1.1 | `HfHubHTTPError` is not yet caught; raw traceback visible on Hub 5xx. Will add handler in v1.1. |

---

### Authorization

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| A-1 | `HF_TOKEN` not set and model is private — Hub returns 404 (not 401) for private repos to unauthenticated users; surfaces as `RepositoryNotFoundError` (same as F-1 above) | `_cmd_download` (preflight) | P1 | **Fixed in v1.0** | Covered by F-1 handler — the hint to run `huggingface-cli login` is the correct recovery action |
| A-2 | `HF_TOKEN` is expired or revoked — Hub returns 401 wrapped in `HfHubHTTPError`; user sees raw exception if not handled | `_cmd_download` | P2 | Deferred to v1.1 | Requires `HfHubHTTPError` handler (see F-7). Will add status-401 specific message in v1.1. |

---

### Performance Limits

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| P-1 | Very large model (e.g. Llama 3.1 405B quantized, ~200 GB) — `snapshot_download()` runs for potentially hours; `huggingface_hub` provides built-in tqdm progress; no action needed from `_cmd_download` | `_cmd_download` | P2 | Already Handled | HF Hub's tqdm covers progress. Document in README that large downloads are resumable. |
| P-2 | `--dry-run` calls `repo_info()` to estimate disk size — this is a live Hub API call that can take 1–3 seconds even on a good connection; user may expect `--dry-run` to be instantaneous | `_cmd_download` | P2 | Deferred | Acceptable: `--dry-run` exists to prevent wasted bandwidth/disk, not to be zero-latency. If only the catalog is checked (without `repo_info()`), the estimate may be unavailable for non-catalog models. Use `repo_info()` and accept the latency. |

---

### Configuration & Environment

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| C-1 | `HF_HUB_CACHE` set to a non-writable path — `snapshot_download()` raises `PermissionError` before downloading any files | `_cmd_download` | P1 | **Fixed in v1.0** | `except PermissionError` in `snapshot_download()` block prints error with cache path and instructions; `sys.exit(1)` |
| C-2 | `HF_ENDPOINT` set to an unreachable or malformed mirror URL — `repo_info()` raises `ConnectionError` or `ValueError`; covered by F-4 handler | `_cmd_download` | P1 | **Fixed in v1.0** | Covered by F-4 (`ValueError` caught alongside `ConnectionError`) |
| C-3 | `HOME` not set (some Docker / CI environments) — `_cache_path_str()` uses `Path.home()` which raises `RuntimeError`; same issue already documented for `list_mlx_models` (Section 2, C-1) | `_cmd_download`, `_cache_path_str()` | P2 | **Fixed in v1.0** | `_cache_path_str()` wraps `Path.home()` in `except Exception` and returns `"(unknown)"` |

---

### Deployment Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| DL-1 | `download` run for an already-cached model at the same revision — `snapshot_download()` is idempotent and returns the existing cache path immediately, but the command prints no output; user is left wondering if the command ran successfully or did nothing | `_cmd_download` | P1 | **Fixed in v1.0** | `scan_cache_dir()` checked before download; prints `"Already cached: {path}"` and returns without re-downloading |
| DL-2 | `snapshot_download()` atomically commits downloaded blobs via symlinks — if the process is killed mid-operation, the partial `blobs/` directory exists but is not yet linked under `snapshots/`; `resume_download=True` recovers on next run | `_cmd_download` | P2 | Already Handled | `resume_download=True` in design spec. Document in README. |

---

### Runtime Execution Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| R-1 | Pre-download confirmation prompt (`"Download X GB? [y/N]"`) — must handle `EOFError` (non-interactive stdin, CI) and `KeyboardInterrupt` (`Ctrl-C`) cleanly, same as `_cmd_remove()` | `_cmd_download` | P1 | **Fixed in v1.0** | `input()` wrapped in `try/except (EOFError, KeyboardInterrupt)`; prints `"\nAborted."` to stderr; `sys.exit(1)` |
| R-2 | `snapshot_download()` is synchronous blocking I/O — if `_cmd_download` is ever wrapped in an async context, it will block the event loop | `_cmd_download` | P2 | Deferred | CLI-only usage; no async context exists. Note in function docstring: "synchronous; not safe to call from an event loop without `asyncio.to_thread()`". |
| R-3 | tqdm progress bar from Hub writes ANSI escape sequences to stderr — when stderr is redirected to a log file, ANSI codes appear in the log | `_cmd_download` | P2 | Already Handled | `huggingface_hub` disables tqdm automatically when not a TTY (`sys.stderr.isatty() == False`). No action. |

---

## Decisions

- **I-2 (format validation)**: `re.fullmatch(r"[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+", ...)` is required as a pre-condition before any Hub API call. The GOVERNANCE review independently flagged this as required. The regex also covers I-3 (path traversal) implicitly since `..` and `/` in the path component are rejected.
- **I-4 (`--revision ""`)**:  Normalise to `None` rather than validating. Passing an empty string to the Hub is ambiguous; `None` gets the default branch deterministically.
- **S-2 (SIGINT resume)**: `resume_download=True` is already in the design spec. Verify it is passed to every `snapshot_download()` call site.
- **D-1 (0 GB estimate)**: Print "could not determine size; skipping disk check" rather than "0 GB required" to avoid misleading users with a trivially-passing preflight that provides no safety.
- **F-1 + A-1 convergence**: `RepositoryNotFoundError` is the Hub's signal for both "doesn't exist" and "private, no token". A single handler with both hints (typo check + login hint) covers both cases without needing to distinguish them.
- **DL-1 (already cached)**: Checking `scan_cache_dir()` before calling `snapshot_download()` adds a small overhead but prevents silent no-ops that users misread as failures. Worth the latency.
