"""CLI and programmatic interface for Apple BaseFM model management.

Subcommands
-----------
``apple-basefm mlx-models``
    List locally cached MLX-compatible HuggingFace models.

``apple-basefm suggest``
    Detect local Apple Silicon hardware and suggest suitable models.
    Queries HuggingFace Hub by default; use ``--offline`` to skip the
    network call.

``apple-basefm remove``
    Remove one or more cached models from the local HuggingFace cache.

Public API::

    from apple_basefm import list_mlx_models, MLXModelInfo

    for model in list_mlx_models(filter="llama"):
        print(model["repo_id"], model["size_on_disk_str"])

Requires ``huggingface_hub>=0.23``::

    pip install "apple-basefm[mlx]"
"""
from __future__ import annotations

import json
import logging
import sys
import textwrap
from typing import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False


def _require_hf_hub() -> None:
    if not _HF_HUB_AVAILABLE:
        raise RuntimeError(
            "This command requires huggingface_hub. "
            'Install it with: pip install "apple-basefm[mlx]"'
        )


def _cache_path_str() -> str:
    """Best-effort human-readable path to the HF hub cache."""
    try:
        import os
        from pathlib import Path

        return os.environ.get(
            "HF_HUB_CACHE",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
    except Exception:  # noqa: BLE001
        return "(unknown)"


# ---------------------------------------------------------------------------
# MLXModelInfo — public type for list_mlx_models()
# ---------------------------------------------------------------------------


class MLXModelInfo(TypedDict):
    """Metadata for a single locally cached model repo."""

    repo_id: str
    """HuggingFace repo identifier."""
    size_on_disk: int
    """Total cached size across all revisions, in bytes."""
    size_on_disk_str: str
    """Human-readable size string, e.g. ``"1.8G"``."""
    nb_files: int
    """Total number of cached files across all revisions."""
    last_modified: float
    """Unix timestamp of the most recently modified cached file."""


# ---------------------------------------------------------------------------
# list_mlx_models — public programmatic API
# ---------------------------------------------------------------------------


def list_mlx_models(
    filter: str | None = None,  # noqa: A002
    all_models: bool = False,
) -> list[MLXModelInfo]:
    """Return locally cached HuggingFace model repos that are MLX-compatible.

    Args:
        filter: Case-insensitive substring matched against ``repo_id``.
            An empty string matches everything (equivalent to ``None``).
            Must be ``str`` or ``None``; any other type raises ``TypeError``.
        all_models: When ``False`` (default), only repos whose ``repo_id``
            contains ``"mlx"`` are returned. Set to ``True`` to include
            every ``model``-type repo.

    Returns:
        List of :class:`MLXModelInfo` dicts sorted by ``repo_id``. Empty if
        the cache does not exist, is unreadable, or has no matching repos.

    Raises:
        RuntimeError: If ``huggingface_hub`` is not installed or the cache
            cannot be read.
        TypeError: If ``filter`` is not ``str`` or ``None``.
    """
    _require_hf_hub()

    if filter is not None and not isinstance(filter, str):
        raise TypeError(
            f"list_mlx_models() filter must be str or None, got {type(filter).__name__!r}"
        )

    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        logger.debug("HuggingFace cache not found; returning empty list")
        return []
    except PermissionError as exc:
        raise RuntimeError(
            f"Permission denied reading HuggingFace cache: {exc}. "
            "Check directory permissions or set HF_HUB_CACHE to an accessible path."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not scan HuggingFace cache: {exc}. "
            "Set the HF_HUB_CACHE environment variable to an explicit cache path."
        ) from exc

    filter_lower = filter.lower() if filter else None
    results: list[MLXModelInfo] = []

    for repo in cache_info.repos:
        if repo.repo_type != "model":
            continue
        if not all_models and "mlx" not in repo.repo_id.lower():
            continue
        if filter_lower and filter_lower not in repo.repo_id.lower():
            continue
        results.append(
            MLXModelInfo(
                repo_id=repo.repo_id,
                size_on_disk=repo.size_on_disk,
                size_on_disk_str=repo.size_on_disk_str,
                nb_files=repo.nb_files,
                last_modified=repo.last_modified,
            )
        )

    results.sort(key=lambda m: m["repo_id"])
    return results


# ---------------------------------------------------------------------------
# mlx-models formatters
# ---------------------------------------------------------------------------

_ID_MAX = 55


def _fmt_mlx_text(results: list[MLXModelInfo]) -> str:
    lines = []
    for m in results:
        repo_id = textwrap.shorten(m["repo_id"], width=_ID_MAX, placeholder="...")
        lines.append(f"{repo_id:<{_ID_MAX}}  {m['size_on_disk_str']:>8}  {m['nb_files']} files")
    return "\n".join(lines)


def _fmt_mlx_table(results: list[MLXModelInfo]) -> str:
    col_id, col_size, col_files = _ID_MAX, 8, 7
    header = f"{'REPO ID':<{col_id}}  {'SIZE':>{col_size}}  {'FILES':>{col_files}}"
    sep = f"{'─'*col_id}  {'─'*col_size}  {'─'*col_files}"
    lines = [header, sep]
    for m in results:
        repo_id = textwrap.shorten(m["repo_id"], width=col_id, placeholder="...")
        lines.append(
            f"{repo_id:<{col_id}}  {m['size_on_disk_str']:>{col_size}}  {m['nb_files']:>{col_files}}"
        )
    return "\n".join(lines)


def _fmt_mlx_json(results: list[MLXModelInfo]) -> str:
    return json.dumps([dict(m) for m in results], indent=2)


_MLX_FORMATTERS = {"text": _fmt_mlx_text, "table": _fmt_mlx_table, "json": _fmt_mlx_json}


# ---------------------------------------------------------------------------
# suggest formatters
# ---------------------------------------------------------------------------


def _fmt_suggest_header(result: dict) -> str:
    hw = result["hardware"]
    chip = hw["chip"] or ("Apple Silicon" if hw["is_apple_silicon"] else "Unknown")
    ram = f"{hw['ram_gb']} GB RAM" if hw["ram_gb"] else "RAM unknown"
    disk = f"{hw['free_disk_gb']} GB free" if hw["free_disk_gb"] else "disk unknown"
    source_label = (
        "HuggingFace Hub (live)" if result["source"] == "online"
        else "built-in catalog (offline)"
    )
    return (
        f"Hardware:  {chip}  |  {ram}  |  {disk}\n"
        f"Source:    {source_label}"
    )


def _fmt_suggest_table(result: dict) -> str:
    models = result["models"]
    source = result["source"]
    show_dl = source == "online"

    col_id = _ID_MAX
    col_quant, col_size, col_ram, col_dl, col_note = 5, 6, 7, 9, 28

    if show_dl:
        header = (
            f"{'REPO ID':<{col_id}}  {'QUANT':>{col_quant}}  {'~SIZE':>{col_size}}"
            f"  {'MIN RAM':>{col_ram}}  {'DOWNLOADS':>{col_dl}}  NOTE"
        )
        sep = (
            f"{'─'*col_id}  {'─'*col_quant}  {'─'*col_size}"
            f"  {'─'*col_ram}  {'─'*col_dl}  {'─'*col_note}"
        )
    else:
        header = (
            f"{'REPO ID':<{col_id}}  {'QUANT':>{col_quant}}  {'~SIZE':>{col_size}}"
            f"  {'MIN RAM':>{col_ram}}  NOTE"
        )
        sep = (
            f"{'─'*col_id}  {'─'*col_quant}  {'─'*col_size}"
            f"  {'─'*col_ram}  {'─'*col_note}"
        )

    lines = [_fmt_suggest_header(result), "", header, sep]
    for m in models:
        repo_id = textwrap.shorten(m["repo_id"], width=col_id, placeholder="...")
        size_str = f"~{m['disk_gb']:.1f}G"
        ram_str = f"{m['min_ram_gb']}GB"
        note = textwrap.shorten(m["note"], width=col_note, placeholder="...")
        quant_str = f"{m['quant_bits']}bit"
        if show_dl:
            dl_str = f"{m['downloads']:,}" if m["downloads"] is not None else "-"
            lines.append(
                f"{repo_id:<{col_id}}  {quant_str:>{col_quant}}  {size_str:>{col_size}}"
                f"  {ram_str:>{col_ram}}  {dl_str:>{col_dl}}  {note}"
            )
        else:
            lines.append(
                f"{repo_id:<{col_id}}  {quant_str:>{col_quant}}  {size_str:>{col_size}}"
                f"  {ram_str:>{col_ram}}  {note}"
            )
    return "\n".join(lines)


def _fmt_suggest_json(result: dict) -> str:
    return json.dumps(
        {
            "hardware": dict(result["hardware"]),
            "source": result["source"],
            "suggestions": [dict(m) for m in result["models"]],
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_mlx_models(args: object) -> None:
    try:
        results = list_mlx_models(filter=args.filter, all_models=args.all_models)  # type: ignore[attr-defined]
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        hint = "" if args.all_models else " (try --all to include non-MLX-named repos)"  # type: ignore[attr-defined]
        print(f"No models found in cache: {_cache_path_str()}{hint}", file=sys.stderr)
        return

    print(_MLX_FORMATTERS[args.format](results))  # type: ignore[attr-defined]


def _cmd_suggest(args: object) -> None:
    try:
        from apple_basefm._catalog import suggest_models
        from apple_basefm._hardware import detect_hardware
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    hw = detect_hardware()

    if not hw["is_apple_silicon"]:
        print("error: suggest requires Apple Silicon macOS (arm64).", file=sys.stderr)
        sys.exit(1)

    if hw["ram_gb"] == 0:
        print(
            "error: Apple Silicon detected but unified memory size could not be "
            "determined.\n"
            "       Ensure 'sysctl' is on PATH, or use '--offline' to skip hardware "
            "detection.",
            file=sys.stderr,
        )
        sys.exit(1)

    result = suggest_models(hw=hw, online=not args.offline)  # type: ignore[attr-defined]

    if not result["models"]:
        print("No matching models found for your hardware profile.", file=sys.stderr)
        print(_fmt_suggest_header(result), file=sys.stderr)
        return

    if args.format == "json":  # type: ignore[attr-defined]
        print(_fmt_suggest_json(result))
    else:
        print(_fmt_suggest_table(result))


def _cmd_remove(args: object) -> None:
    _require_hf_hub()

    repo_ids: list[str] = args.repo_id  # type: ignore[attr-defined]
    yes: bool = args.yes  # type: ignore[attr-defined]

    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        print(f"error: HuggingFace cache not found at {_cache_path_str()}", file=sys.stderr)
        sys.exit(1)
    except (PermissionError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    cached = {r.repo_id: r for r in cache_info.repos}

    missing = [rid for rid in repo_ids if rid not in cached]
    if missing:
        for rid in missing:
            print(f"error: model not found in cache: {rid}", file=sys.stderr)
        sys.exit(1)

    all_hashes: list[str] = []
    print()
    for rid in repo_ids:
        repo = cached[rid]
        all_hashes.extend(rev.commit_hash for rev in repo.revisions)
        print(f"  {rid}  ({repo.size_on_disk_str}, {repo.nb_files} files)")

    strategy = cache_info.delete_revisions(*all_hashes)
    print(f"\nThis will free {strategy.expected_freed_size_str}.")

    if not yes:
        try:
            answer = input("Continue? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            sys.exit(1)
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    strategy.execute()
    noun = "model" if len(repo_ids) == 1 else "models"
    print(f"Removed {len(repo_ids)} {noun}. Freed {strategy.expected_freed_size_str}.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``apple-basefm`` CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="apple-basefm",
        description="Apple BaseFM — Apple Silicon model management tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── mlx-models ──────────────────────────────────────────────────────────
    p_list = sub.add_parser(
        "mlx-models",
        help="List locally cached MLX-compatible models.",
        description=(
            "List locally cached HuggingFace model repos that are MLX-compatible.\n\n"
            "By default, only repos whose name contains 'mlx' are shown. Use\n"
            "--all to include every model-type repo (useful for non-MLX-named\n"
            "repos run with mlx-lm)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_list.add_argument("--filter", metavar="TERM", default=None,
                        help="Case-insensitive substring filter on repo IDs.")
    p_list.add_argument("--all", dest="all_models", action="store_true", default=False,
                        help="Include all model-type repos, not just MLX-named ones.")
    p_list.add_argument("--format", choices=["text", "table", "json"], default="text",
                        help="Output format (default: text).")

    # ── suggest ─────────────────────────────────────────────────────────────
    p_suggest = sub.add_parser(
        "suggest",
        help="Suggest MLX models suited to your Apple Silicon hardware.",
        description=(
            "Detects your chip (M1/M2/M3/M4), unified memory, and free disk\n"
            "space, then recommends mlx-community models that fit.\n\n"
            "Queries HuggingFace Hub by default for the latest popular models.\n"
            "Use --offline to skip the network call and use the built-in catalog."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_suggest.add_argument("--offline", action="store_true", default=False,
                           help="Use the built-in catalog; skip the HuggingFace Hub query.")
    p_suggest.add_argument("--format", choices=["table", "json"], default="table",
                           help="Output format (default: table).")

    # ── remove ──────────────────────────────────────────────────────────────
    p_remove = sub.add_parser(
        "remove",
        help="Remove cached model(s) from the local HuggingFace cache.",
        description=(
            "Deletes all locally cached revisions of the specified model(s),\n"
            "freeing disk space. Shows what will be freed and prompts for\n"
            "confirmation unless --yes is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_remove.add_argument(
        "repo_id",
        nargs="+",
        metavar="REPO_ID",
        help='Repo ID(s) to remove, e.g. mlx-community/Llama-3.2-3B-Instruct-4bit',
    )
    p_remove.add_argument("--yes", "-y", action="store_true", default=False,
                          help="Skip the confirmation prompt.")

    args = parser.parse_args(argv)

    if args.command == "mlx-models":
        _cmd_mlx_models(args)
    elif args.command == "suggest":
        _cmd_suggest(args)
    elif args.command == "remove":
        _cmd_remove(args)
