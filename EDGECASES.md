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
| I-1 | `head_dim <= 0` not validated in `build()` — `make_rotation_matrix(0)` produces a 0×0 matrix; `QuantizedKVCache` downstream raises an obscure MLX runtime error | `cache_v2.py: TurboQuantV2Cache.build()` | P1 | Must Handle | Handle in code: add `if head_dim < 1: raise ValueError(...)` at top of `build()` |
| I-2 | `group_size` not required to evenly divide `head_dim` — MLX quantization may raise a runtime error if the last group is undersized (model-dependent) | `cache_v2.py: TurboQuantV2Cache.build()` | P1 | Should Handle | Handle in code: add a `logger.warning` (not ValueError — mlx-lm clamps internally on some versions); add a unit test for the non-divisible case |
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
| D-1 | K/V tensors arrive with an unexpected rank or last-dim ≠ `head_dim` — `keys @ self.rotation` raises an MLX shape error with a message that does not mention TurboQuant, making it hard to diagnose. Models using grouped-query attention (GQA) or multi-query attention (MQA) may use different K/V head counts than expected | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()` | P1 | Should Handle | Handle in code: wrap `keys @ self.rotation` in a `try/except` that re-raises with a message naming the rotation dimension and actual key shape; add a test with a mismatched shape |
| D-2 | Dtype mismatch: rotation matrix is generated in MLX default dtype (float32), but models running in bfloat16 pass bfloat16 keys — `keys @ self.rotation` may silently upcast to float32 (fine on M-series) or raise depending on MLX version | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()` | P1 | Should Handle | Handle in code: in `update_and_fetch()`, cast `self.rotation` to `keys.dtype` before the matmul; add a unit test with a float16 key array |

---

### Failure Paths

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| F-1 | `inner.step = self._step` is an attribute assignment in `_ensure_inner()` — if a future mlx-lm release changes `step` to a read-only property or removes it, this silently no-ops or raises `AttributeError` at the first cache use rather than at construction time | `cache_v2.py: _TurboQuantV2LayerCache._ensure_inner()` | P1 | Should Handle | Handle in code: wrap the `inner.step = self._step` line in `try/except AttributeError` and log a warning; add to the mlx-lm upgrade checklist |
| F-2 | `mx.linalg.qr` can raise exceptions other than `AttributeError` (e.g., `mx.core.MLXError` for a degenerate matrix, or a general `Exception` from a buggy MLX build) — the current handler only catches `AttributeError` and checks `"qr" in str(exc)` | `rotation.py: make_rotation_matrix()` | P1 | Should Handle | Handle in code: add a bare `except Exception as exc` fallback that wraps in `RuntimeError("make_rotation_matrix failed: ...")` with the original exc chained |

---

### Authorization

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| A-1 | N/A — `apple_basefm` is a local inference library with no network access, user accounts, or multi-tenant data. All inference runs on-device with caller-supplied credentials for any external model repos | Entire package | — | Out of Scope | No authorization surface exists in this package. Model-repo auth is delegated to `mlx_lm.load()` and HuggingFace Hub. |

---

### Performance Limits

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| P-1 | First `build()` call with `use_rotation=True` on a large model (e.g., `head_dim=8192` for a MoE) runs `mx.random.normal(shape=(8192, 8192))` + QR decomposition — O(n³) cost, potentially several seconds — with no user-visible warning or progress indicator | `rotation.py: make_rotation_matrix()`, `cache_v2.py: TurboQuantV2Cache.build()` | P1 | Should Handle | Handle in code: add an `INFO`-level log in `build()` before calling `make_rotation_matrix()` naming `head_dim` and noting the one-time cost; add a `logger.info` in `make_rotation_matrix()` on completion with elapsed time via `time.perf_counter()` |

---

### Configuration & Environment

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| C-1 | **Silent correctness failure**: `use_rotation=True` applies `keys @ R` and `values @ R` inside `update_and_fetch()`, so QuantizedKVCache stores K/V in *rotated space*. The model's SDPA then computes `Q @ (K·R)ᵀ` instead of `Q @ Kᵀ`, changing all attention scores. The compensating SDPA patch (`attention_v2.py`) is deferred. Users who enable `use_rotation=True` without reading the module-level docstring get silently incorrect attention (not a quantization-only approximation — the attention distribution is rotated). | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()`, `apple_local.py: _KV_PRESETS["turboquant-v2"]` | **P0** | Must Handle | Handle in code: (1) emit a `logger.warning` from `build()` when `use_rotation=True` stating the SDPA patch is absent and attention equivalence is not guaranteed; (2) update the `turboquant-v2` preset docstring and README to state that `use_rotation=False` (LEAN) is the safe default until `attention_v2.py` ships; (3) consider defaulting `use_rotation` to `False` and making `True` opt-in |
| C-2 | `use_normalization=True` (the default) is a silently ignored parameter — `__post_init__` logs only at `DEBUG` level and then does nothing. A user who enables it expecting normalization to be applied gets no effect and no visible notice | `cache_v2.py: TurboQuantV2Cache.__post_init__()` | P1 | Should Handle | Handle in code: upgrade the `__post_init__` log for `use_normalization=True` from `DEBUG` to `INFO` level, and add `(no-op — reserved for attention_v2.py)` to the message so it appears in default logging config |

---

### Deployment Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| L-1 | Two versions of `apple_basefm` deployed simultaneously (e.g., during a rolling wheel upgrade) could have different rotation matrices if the seed algorithm or MLX RNG implementation changes between versions. A cached `TurboQuantV2Cache` instance on one worker and a freshly constructed one on another would produce different rotations for the same model, but since KV caches are per-call and never shared across workers, this has no practical impact | `rotation.py: make_rotation_matrix()` | P2 | Deferred | No action needed — KV state is ephemeral (per-forward-call) and not shared across processes |

---

### Runtime Execution Lifecycle

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| R-1 | `_inner.update_and_fetch(keys, values)` raises mid-generation (e.g., MLX OOM, quantization overflow) — the inner `QuantizedKVCache` may have partially updated its state (offset incremented, buffers partially written). There is no rollback path. Subsequent calls to the same layer cache object attempt to continue from a potentially inconsistent offset | `cache_v2.py: _TurboQuantV2LayerCache.update_and_fetch()` | P1 | Should Handle | Handle in code: on `update_and_fetch()` exception, set `self._inner = None` (forcing `_ensure_inner()` to recreate a fresh inner cache on retry) rather than retaining the half-updated state; add a test that verifies recovery after a simulated update failure |

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
| I-1 | `list_mlx_models(filter=0)` — `filter.lower()` raises `AttributeError` when caller passes a non-string value to the programmatic API | `_cli.py: list_mlx_models()` | P1 | Must Handle | Add to Stripboard: type-guard `filter` at function entry — `if filter is not None and not isinstance(filter, str): raise TypeError(...)` |
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
| D-1 | MLX detection heuristic (`"mlx" in repo_id.lower()`) produces false negatives for models downloaded and used with `mlx-lm` that have no "mlx" in their repo ID (e.g., `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`) — user sees an empty list despite having usable models | `_cli.py: list_mlx_models()` | P1 | Should Handle | Add to Stripboard: (1) add `--all` flag to list all `model`-type repos regardless of name; (2) document heuristic in `--help` text and function docstring; (3) print an informational note when the filtered list is empty suggesting `--all` |
| D-2 | `nb_files=0` or `size_on_disk=0` on a corrupted or partially-deleted cache entry — the TypedDict accepts these values; formatters render `0 B` and `0 files` cleanly | `_cli.py: list_mlx_models()` | P2 | Already Handled | No action needed — zero values are valid and render without crashing |

---

### Failure Paths

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| F-1 | `huggingface_hub` not installed — `ImportError` at module import when `apple-basefm` is installed without the `mlx` extra; crashes with an unhelpful traceback | `_cli.py` (module level) | P1 | Must Handle | Add to Stripboard: wrap `import huggingface_hub` in `try/except ImportError` and raise `RuntimeError('list_mlx_models requires huggingface_hub: pip install "apple-basefm[mlx]"')` |
| F-2 | `CacheNotFound` raised by `scan_cache_dir()` when `HF_HUB_CACHE` points to a nonexistent path or the cache has never been populated — the planned `return []` silently produces no CLI output, leaving the user with no feedback | `_cli.py: list_mlx_models()` | P1 | Should Handle | Add to Stripboard: catch `CacheNotFound`; return `[]` from the function; in `main()` print an informational message to `stderr` with the resolved cache path |
| F-3 | `PermissionError` reading the HF cache directory — surfaces as an unhandled traceback | `_cli.py: list_mlx_models()` | P2 | Should Handle | Catch `PermissionError` and re-raise as `RuntimeError` with a human-readable message identifying the cache path |

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
| P-2 | Table formatter with repo IDs longer than 60 characters — columns become very wide, wrapping unpredictably in narrow terminals | `_cli.py: main()` (table format) | P2 | Should Handle | Use fixed column widths with `textwrap.shorten()` for truncation; add `...` suffix for truncated names |

---

### Configuration & Environment

| # | Edge Case | Component | Severity | Status | Action |
|---|---|---|---|---|---|
| C-1 | `HOME` not set in some Docker / CI environments — `Path.home()` raises `RuntimeError: Could not determine home directory` inside `scan_cache_dir()`, surfacing as an unhandled traceback | `_cli.py: list_mlx_models()` | P1 | Should Handle | Add to Stripboard: wrap `scan_cache_dir()` call in `try/except (CacheNotFound, RuntimeError)` and surface a clear message explaining that `HF_HUB_CACHE` must be set when `HOME` is unavailable |
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
