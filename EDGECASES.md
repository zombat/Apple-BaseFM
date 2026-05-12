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
