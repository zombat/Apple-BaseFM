# Governance Review: apple-basefm

**Date**: 2026-05-12
**Scope**: Full codebase — `apple_basefm/` package, `tests/`, `scripts/`, `.github/skills/` scripts.
**Scanner**: `python3 scripts/guard.py --strict` (full project) and `--dir apple_basefm --strict` (package only)
**Last scan**: 2026-05-12 (post TurboQuant V2 changes)

---

## Scanner Findings

### Full project (`python3 scripts/guard.py --strict`)

```
Scanning /home/noot/dspy-apple …

[HIGH] INJ-1  .github/skills/distribution/scripts/release.py:25
  subprocess.run() called with shell=True. Use a list of arguments and shell=False to prevent command injection.
  > r = subprocess.run(

[HIGH] INJ-1  .github/skills/planning/scripts/recon.py:31
  subprocess.run() called with shell=True. Use a list of arguments and shell=False to prevent command injection.
  > r = subprocess.run(

[HIGH] INJ-1  .github/skills/production/scripts/check.py:34
  subprocess.run() called with shell=True. Use a list of arguments and shell=False to prevent command injection.
  > r = subprocess.run(

[WARNING] JSON-1  tests/test_cli.py:211
  json.loads() called outside a try/except block.
  > parsed = json.loads(out)

[WARNING] JSON-1  tests/test_cli.py:340
  json.loads() called outside a try/except block.
  > parsed = json.loads(out)

[WARNING] JSON-1  tests/test_cli.py:502
  json.loads() called outside a try/except block.
  > parsed = json.loads(out)

Found 6 violation(s): 3 HIGH, 3 WARNING
```

> **Note**: `review/scripts/pr.py` was flagged in the initial scan (shell=True + git-ref
> interpolation — a genuine injection risk). It has been fixed: the `run()` helper
> now uses `shell=False` with list-form arguments; all callers that passed f-strings
> containing git refs have been updated to pass lists.

### Package only (`python3 scripts/guard.py --dir apple_basefm --strict`)

```
Scanning /home/noot/dspy-apple/apple_basefm …
No violations found. ✓
```

### Tests only (`python3 scripts/guard.py --dir tests --strict`)

```
Scanning /home/noot/dspy-apple/tests …
No violations found. ✓
```

---

## Finding Analysis

### INJ-1 — `shell=True` in `.github/skills/` scripts (HIGH)

All four violations are in developer-tooling skill scripts under `.github/skills/`:

| File | Pattern |
|---|---|
| `distribution/scripts/release.py` | `run("git status --porcelain", ...)` — all callers pass hardcoded string literals |
| `planning/scripts/recon.py` | `run("git log ...", ...)` — all callers pass hardcoded string literals |
| `production/scripts/check.py` | `run(f"{check} {cmd}", ...)` — `check` and `cmd` are both hardcoded tool names |
| `review/scripts/pr.py` | `run(f"git diff {base}...HEAD", ...)` — `base` derives from git branch names |

**Risk assessment**:

- `release.py`, `recon.py`, `check.py`: All `run()` call-sites pass hardcoded string literals. No user-controlled input reaches `shell=True`. Risk is pattern-level only (a future caller could accidentally pass user input).
- `pr.py`: `base` is resolved by `git rev-parse --verify {candidate}` where `candidate` is a branch name. Git branch names may contain shell metacharacters if a repository has been crafted maliciously (e.g., a branch named `main; rm -rf /`). This is a real injection risk on developer workstations.

**Status**: These scripts are not distributed with the package (`.github/` is excluded from the wheel). They are agent-invoked during development only. The `pr.py` finding is a **genuine P1** — see action below.

**Action**:
- `pr.py` (`review/scripts/pr.py`): Replace `run(f"git diff {base}...HEAD")` with `subprocess.run(["git", "diff", f"{base}...HEAD"], shell=False)`. Also replace the `run(f"git rev-parse --verify {candidate}")` call. See [§ Manual Review — Injection](#injection) for the fix.
- `release.py`, `recon.py`, `check.py`: Low immediate risk (hardcoded callers only), but `shell=True` should be replaced with list-form calls to prevent accidental future injection. Track as P2.

---

## Manual Review

### Schema Enforcement

| Check | Status | Notes |
|---|---|---|
| All CLI entry points validate against a declared schema | ✅ Pass | `argparse` with `choices=` constrains `--format`; `list_mlx_models()` checks `isinstance(filter, str)` |
| `list_mlx_models(filter=...)` type-checks its only user-facing parameter | ✅ Pass | Raises `TypeError` for non-str filter |
| `TurboQuantV2Cache.__post_init__()` validates `bits`, `group_size`, `step` | ✅ Pass | ValueError raised with clear message for out-of-range values |
| `AppleLocalLM.__init__()` validates `backend`, `max_concurrency`, `kv_cache` | ✅ Pass | ValueError / TypeError raised at construction time |
| HF Hub API response: `model.id` type-guarded before string operations | ✅ Pass | `isinstance(repo_id, str)` guard added in `_catalog.py` |
| All exit points (CLI output) serialised through formatters, not raw objects | ✅ Pass | `_fmt_mlx_json`, `_fmt_suggest_json` use `json.dumps([dict(m) for m in ...])` |
| Size limits on CLI string fields | ⚠️ Note | `--filter` and `repo_id` have no max-length constraint; for a local CLI tool reading only from the local HF cache this is acceptable — no remote amplification is possible |

### Injection

| Check | Status | Notes |
|---|---|---|
| No SQL string interpolation with user data | ✅ Pass | No SQL in codebase |
| No `subprocess` calls with `shell=True` in package | ✅ Pass | All subprocess calls in `_hardware.py` use `shell=False` with argument lists |
| No user-controlled strings passed to template engines | ✅ Pass | No template engine usage |
| No unsafe deep-merge of user objects | ✅ Pass | Python codebase; no object merge pattern |
| `.github/skills/review/scripts/pr.py` — `git diff {base}...HEAD` with `shell=True` | ❌ **FAIL** | `base` may contain shell metacharacters from a crafted branch name. **Action: rewrite as list-form subprocess call** |

**Fix for `pr.py`**: Replace the `run()` helper's `shell=True` with a list-form wrapper:

```python
# Before (vulnerable pattern):
def run(cmd, cwd=None, timeout=30):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)

# After (safe):
def run(cmd, cwd=None, timeout=30):
    # cmd must be a list; shell=False prevents metacharacter injection
    args = cmd if isinstance(cmd, list) else cmd.split()
    r = subprocess.run(args, shell=False, capture_output=True, text=True, cwd=cwd, timeout=timeout)
```

And update callers that pass format-strings with git refs to use list form:

```python
# Before:
code, diff, _ = run(f"git diff {base}...HEAD", cwd=cwd)

# After:
code, diff, _ = run(["git", "diff", f"{base}...HEAD"], cwd=cwd)
```

### Deserialization

| Check | Status | Notes |
|---|---|---|
| No `pickle.loads()` on untrusted data | ✅ Pass | No pickle usage anywhere in codebase |
| `yaml.safe_load()` used everywhere | ✅ Pass | No `yaml.load()` calls; yaml is not used in the package |
| XML parsed with entity expansion disabled | ✅ Pass | No XML parsing in codebase |
| No `eval()` or `exec()` on user input | ✅ Pass | No eval/exec calls in package or tests |
| JSON parse errors handled before accessing fields | ✅ Pass | `json.loads()` in `_hardware.py` is wrapped in `try/except json.JSONDecodeError`; `json.loads()` in `_catalog.py`'s `read_json` helper catches all exceptions |
| `json.loads()` in `tests/test_cli.py` (lines 211, 340, 502) | ✅ Accepted false positive | These parse the CLI's own serialized output to assert correctness. An uncaught `JSONDecodeError` here is the *correct* failure mode — it means the formatter produced invalid JSON. Wrapping in try/except would suppress a real test failure. Scanner rule JSON-1 does not apply to test assertions on controlled output. |

### Sensitive Data

| Check | Status | Notes |
|---|---|---|
| No secrets in source code | ✅ Pass | No hardcoded credentials; scanner found no SECRET-1 violations |
| No PII written to logs | ✅ Pass | Only model names, repo IDs, and hardware metrics are logged |
| No sensitive fields in output schemas | ✅ Pass | `MLXModelInfo`, `ModelEntry`, `SuggestResult`, `HardwareInfo` contain only non-sensitive ML model metadata |
| Stack traces and internal errors not returned to callers | ✅ Pass | CLI prints structured error messages to stderr; never exposes tracebacks to stdout |
| `input()` prompt in `_cmd_remove()` | ✅ Pass | Used only to collect y/N confirmation; no secret handling |

### Format Normalization

| Check | Status | Notes |
|---|---|---|
| UTF-8 encoding declared on all file reads | ✅ Pass | `encoding="utf-8"` explicit on all file reads (`scripts/guard.py`, `_catalog.py`, `_hardware.py` json parse path) |
| CLI output encoding | ✅ Pass | Python `print()` to stdout; terminal encoding handled by the platform |
| `last_modified` field in `MLXModelInfo` | ⚠️ Note | Stored as Unix float timestamp, not ISO 8601. Acceptable — this is a display/sort value, not an interchange format. JSON output consumers receive a float. |
| Content-Type verification | N/A | No file uploads or HTTP endpoints in this package |

---

## Boundary Map

```
External Data Source          Entry Point                     Guard
─────────────────────         ──────────────────────          ──────────────────────────────
CLI: --filter arg             list_mlx_models(filter=)        isinstance(filter, str) check
CLI: --format arg             argparse choices=               argparse rejects non-choices
CLI: repo_id args (remove)    _cmd_remove()                   dict key lookup (exact match)
HF Hub: model.id              _fetch_online_suggestions()     isinstance(repo_id, str) guard
HF Hub: JSON response         list_models() generator         getattr(..., None) for all fields
system_profiler JSON          _chip_name_from_system_profiler json.loads() in try/except
sysctl output                 _run_sysctl()                   int(raw) in try/except ValueError
TurboQuantV2Cache params      __post_init__()                 ValueError for out-of-range bits/group_size/step
AppleLocalLM params           __init__()                      ValueError for backend/max_concurrency; TypeError for kv_cache
```

---

## Summary

| Area | Distributed Package (`apple_basefm/`) | Tests (`tests/`) | Tooling (`.github/skills/`) |
|---|---|---|---|
| Schema Enforcement | ✅ Clean | N/A | N/A |
| Injection | ✅ Clean | ✅ Clean | ❌ INJ-1 in `pr.py` (P1), INJ-1 in 3 others (P2) |
| Deserialization | ✅ Clean | ✅ Clean (JSON-1 × 3 = accepted false positives) | ✅ Clean |
| Sensitive Data | ✅ Clean | ✅ Clean | ✅ Clean |
| Format Normalization | ✅ Clean | N/A | ✅ Clean |

**The distributed `apple_basefm` package has no governance violations.**

The one genuine security finding (`INJ-1` in `review/scripts/pr.py`) affects only the developer tooling scripts and is blocked from the wheel release. It should be remediated before the skill scripts are used in automated CI pipelines.

---

## Actions Required

| Priority | ID | File | Action |
|---|---|---|---|
| ✅ Fixed | INJ-1 | `.github/skills/review/scripts/pr.py` | `shell=False` + list-form calls; all f-string git-ref callers updated |
| P2 | INJ-1 | `.github/skills/distribution/scripts/release.py` | Replace `shell=True` with list-form calls |
| P2 | INJ-1 | `.github/skills/planning/scripts/recon.py` | Replace `shell=True` with list-form calls |
| P2 | INJ-1 | `.github/skills/production/scripts/check.py` | Replace `shell=True` with list-form calls |
| ✅ Accepted | JSON-1 | `tests/test_cli.py:211,340,502` | False positive — test assertions on controlled CLI output. `JSONDecodeError` is the correct failure signal; suppressing it would hide regressions. No action. |

---

## Pre-Implementation Review: `download` Subcommand

**Date**: 2026-05-12
**Scope**: Planned `apple-basefm download REPO_ID [--revision REV] [--dry-run] [--yes]` CLI subcommand.
**Status**: Design review — no implementation exists yet. Findings are requirements for implementors.

### New Trust Boundaries

```
External Data Source              Entry Point                      Guard Required
─────────────────────────         ──────────────────────────       ──────────────────────────────────────────
CLI: REPO_ID positional arg       _cmd_download(args.repo_id)      Validate format before passing to Hub APIs
CLI: --revision arg               args.revision                    Pass as opaque string; Hub validates; no shell use
Hub API: repo_info() response     repo_info(repo_id)               getattr with None fallback for all fields
Hub API: Hub metadata (disk size) repo_info().siblings[].size      int() in try/except; fall back to catalog value
Local filesystem: cache path      snapshot_download() return value Print verbatim; do not execute or log as command
```

### Schema Enforcement

| Check | Required | Notes |
|---|---|---|
| `REPO_ID` format validated before Hub call | ✅ Required | Must match `owner/repo-name` pattern. A REPO_ID like `../../../../etc/passwd` or one containing shell metacharacters must be rejected before it reaches `snapshot_download()` or `repo_info()`. Recommended: `re.fullmatch(r"[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+", repo_id)`. |
| `--revision` treated as opaque string | ✅ Required | Pass directly to `snapshot_download(revision=...)`. Hub validates commit hashes and branch names; no local validation needed beyond length limit. Must not be shell-interpolated. |
| Hub `repo_info()` response fields accessed defensively | ✅ Required | Use `getattr(info, field, None)` for all metadata fields; Hub API shape can change across `huggingface_hub` versions. |
| Disk size estimation: catalog value preferred over Hub metadata | ✅ Required | For REPO_IDs matching the offline catalog, use `catalog_entry.disk_gb`. For others, derive from `sum(s.size for s in repo_info().siblings if s.size)` in `try/except`; fall back to `None` (skip preflight) if Hub metadata is unavailable. See `gpt-oss-20b` note in `_catalog.py`. |
| Cache path output sanitised before print | ✅ Required | The path returned by `snapshot_download()` is a local filesystem path. Print it verbatim to stdout. Do not format it into a shell command or log it at DEBUG level (avoids leaking the home directory path in public logs). |

### Injection

| Check | Status | Notes |
|---|---|---|
| `REPO_ID` never shell-interpolated | ✅ Must enforce | `snapshot_download` and `repo_info` accept `repo_id` as a Python string argument — no shell invocation. `shell=False` is irrelevant here (no subprocess), but the pattern must hold if any future wrapper adds subprocess calls. |
| `--revision` never shell-interpolated | ✅ Must enforce | Same as above. |
| No `subprocess.run(shell=True)` in `_cmd_download` | ✅ Must enforce | If progress reporting or any helper is added via subprocess, use list-form only. |
| `REPO_ID` path traversal in any `local_dir` usage | ✅ Must enforce | If `snapshot_download(local_dir=...)` is used instead of the default cache path, validate that the resolved `local_dir` path does not escape the intended target directory (use `Path.resolve()` and check the prefix). Default cache path (no `local_dir`) is safe — Hub manages it. |

### Sensitive Data

| Check | Status | Notes |
|---|---|---|
| `HF_TOKEN` never logged | ✅ Must enforce | `huggingface_hub` reads `HF_TOKEN` from env internally. `_cmd_download` must not log `os.environ`, the full Hub client config, or any auth headers. Log only: repo ID, revision, estimated size, and result path. |
| Token not echoed in error messages | ✅ Must enforce | If `repo_info()` raises `RepositoryNotFoundError` or `GatedRepoError`, surface only the repo ID and a human-readable action (e.g. "run `huggingface-cli login`") — never the raw exception chain which may contain auth context. |
| Cache path not logged at INFO level in non-verbose mode | ⚠️ Recommended | The local cache path contains the user's home directory. Print to stdout on success (the user chose to see it); do not emit at `logger.info()` which would appear in library consumers' log streams. |

### Deserialization

| Check | Status | Notes |
|---|---|---|
| Hub API response deserialization | ✅ Already handled | `huggingface_hub` deserializes all Hub responses internally. `_cmd_download` only calls typed Hub SDK methods (`repo_info`, `snapshot_download`) and reads typed attributes — no raw JSON parsing. |
| No pickle, yaml.load, or eval on any Hub response | ✅ N/A | No such patterns are required by the download flow. |

### Format Normalization

| Check | Status | Notes |
|---|---|---|
| `REPO_ID` normalized to lowercase before comparison with catalog | ✅ Required | HuggingFace repo IDs are case-insensitive on the Hub but the local catalog uses lowercase. Normalize with `.lower()` before the catalog lookup to avoid a case mismatch causing the preflight to skip the catalog disk_gb and fall through to unreliable Hub metadata. |
| Disk size comparison uses consistent units | ✅ Required | `detect_hardware().free_disk_gb` returns a float in GB. Hub sibling sizes are in bytes. Convert to GB (`bytes / 1e9`) before comparison; use the catalog `disk_gb` float directly when available. |

### Boundary Map (download subcommand)

```
External Data Source          Entry Point                        Guard
─────────────────────         ──────────────────────             ──────────────────────────────────────────
CLI: REPO_ID                  args.repo_id (argparse)            re.fullmatch pattern before any Hub call
CLI: --revision               args.revision (argparse)           Opaque string; Hub validates; length limit only
Hub: repo_info() metadata     repo_info(repo_id)                 getattr + try/except; catalog disk_gb preferred
Hub: snapshot_download()      snapshot_download(repo_id, rev)    No local_dir → default cache is safe
Filesystem: returned path     print(result_path)                 Verbatim stdout only; not logged via logger
Env: HF_TOKEN                 huggingface_hub internals          Never logged; error messages strip auth context
Hardware: free_disk_gb        detect_hardware().free_disk_gb     Compared in GB; catalog value preferred for size
```
