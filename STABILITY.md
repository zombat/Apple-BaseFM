# Stability and Deprecation Policy

## API Stability Classification

Each public symbol is classified as one of:

| Class | Guarantee |
|-------|-----------|
| **Stable** | No breaking changes without a major version bump (v1 → v2). Deprecated APIs remain for one full major version with `DeprecationWarning`. |
| **Beta** | May change at minor version bumps (v0.x → v0.y) with `DeprecationWarning` and a 2-minor-version removal window. |
| **Experimental** | May change or be removed at any time. Clearly documented as experimental. |

## Current Status: v0.x Pre-Release

apple-basefm is currently in **pre-release** (v0.x, Alpha classifier). During
this period:

- The top-level public API (`AppleFoundationLM`, `AppleLocalLM`, and the
  symbols listed in `__all__`) is classified **Beta**.
- Internal implementation details (`_base`, `_compat`, `_mlx`, `_response`,
  `_kv`, `_logging`, `_telemetry`) are **Experimental** — do not import them
  directly in production code.
- The `TurboQuantV2Cache` / `KVCacheStrategy` interface is **Beta**. Preset
  names (`"turboquant-v2"`, `"turboquant-v2-lean"`) are **Stable** from v0.2+.

## v0.x → v1.0 Compatibility Contract

The following guarantees apply at the v1.0 release:

1. All **Beta** symbols become **Stable**.
2. All symbols deprecated in v0.x with a documented removal target of ≤ v1.0
   **will be removed** at v1.0.
3. The `KVCacheStrategy` protocol interface becomes **Stable** — no changes
   without a major version bump.
4. The preset name strings become **Stable**.

## Deprecation Process

### v0.x (pre-release)

1. The symbol is marked with `warnings.warn(msg, DeprecationWarning, stacklevel=2)`.
2. A `CHANGELOG.md` entry is added under the release that introduces the warning.
3. This table is updated with the symbol, the version it was deprecated in, and
   the planned removal version (at least 2 minor versions after the warning).
4. The symbol **must not be removed** in the same release as the warning.

### v1.0+ (stable)

1. Same warning + CHANGELOG process.
2. Deprecated APIs survive until the next major version bump (v1.x → v2.0).
3. A minimum of one full major version is required between the warning and removal.

## Current Deprecations

| Symbol | Deprecated In | Removal Target | Reason |
|--------|--------------|----------------|--------|
| `TurboQuantV2Cache(use_normalization=True)` | v0.2.1 | v0.4 | No-op until `attention_v2.py` ships. Pass `use_normalization=False` to suppress the warning. |

## What Is Explicitly NOT Stable

- `apple_basefm._kv.attention_v2` — raises `NotImplementedError`. Not part
  of any public contract.
- `_KV_PRESETS` dict in `apple_local.py` — internal. Use the string preset
  names via the `kv_cache` parameter.
- The `_FMUsage`, `_FMChoice`, `_FMMessage`, `_FMResponse` types — internal
  response containers. DSPy's `BaseLM` contract is the stability surface.
- The `_tool_class_cache` and `_TOOL_CACHE_MAXSIZE` constants — internal.

## Endorsing This Policy

By importing `apple_basefm` in production code, you accept that:

- Pre-1.0 minor releases may break **Beta** symbols with a deprecation cycle.
- You should pin a minor version range in production (e.g. `apple-basefm>=0.2,<0.3`).
- You should monitor `DeprecationWarning` output in your test suite to catch
  upcoming removals before they affect you.
