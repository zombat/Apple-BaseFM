"""Curated MLX model catalog and hardware-aware model suggestion.

Provides :func:`suggest_models`, which cross-references a hardware profile
from :func:`~apple_basefm._hardware.detect_hardware` against a curated list
of well-known ``mlx-community`` models and optionally the live HuggingFace Hub.

Online mode (default)
    Queries the HF Hub for the most-downloaded ``mlx-community`` text-generation
    models, infers RAM/disk requirements from model name heuristics, and filters
    by the detected hardware. Falls back to the offline catalog automatically if
    the network is unavailable or any error occurs.

Offline mode
    Uses a hand-curated catalog of well-known models with verified size and RAM
    requirements. No network access required.

Public API::

    from apple_basefm import suggest_models, detect_hardware

    # Auto-detects hardware and queries HF Hub
    result = suggest_models()
    for m in result["models"]:
        print(m["repo_id"], m["min_ram_gb"], "GB RAM required")

    # Force offline; supply pre-detected hardware
    hw = detect_hardware()
    result = suggest_models(hw=hw, online=False)
    print("Source:", result["source"])  # "offline"

Note on online size estimates
    When ``source == "online"``, ``disk_gb`` and ``min_ram_gb`` are inferred
    from the model name (parameter count × bits-per-weight). These are
    approximations; the offline catalog contains verified values.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from apple_basefm._hardware import HardwareInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ModelEntry(TypedDict):
    """Metadata for a single suggested model."""

    repo_id: str
    """HuggingFace repo identifier, e.g. ``"mlx-community/Llama-3.2-3B-Instruct-4bit"``."""
    quant_bits: int
    """Quantization level in bits (e.g. 4, 8, 16)."""
    disk_gb: float
    """Approximate disk space required to download, in gigabytes."""
    min_ram_gb: int
    """Minimum unified memory needed for inference at moderate context length."""
    min_chip_gen: int
    """Minimum Apple Silicon generation required (1 = M1, 2 = M2, …)."""
    note: str
    """One-line description or rationale for the suggestion."""
    downloads: int | None
    """HF Hub all-time download count. ``None`` for offline-catalog entries."""
    likes: int | None
    """HF Hub like count. ``None`` for offline-catalog entries."""


class SuggestResult(TypedDict):
    """Result returned by :func:`suggest_models`."""

    models: list[ModelEntry]
    """Filtered, hardware-appropriate model suggestions sorted by ``min_ram_gb`` descending."""
    source: Literal["online", "offline"]
    """Whether suggestions came from the live HF Hub (``"online"``) or the built-in
    curated catalog (``"offline"``)."""
    hardware: HardwareInfo
    """The hardware profile used for filtering."""


# ---------------------------------------------------------------------------
# Curated offline catalog
# Disk sizes and min_ram_gb values are verified approximations.
# Rule of thumb: min_ram_gb ≈ disk_gb + 2 GB KV/activation overhead.
# ---------------------------------------------------------------------------

def _entry(
    repo_id: str,
    quant_bits: int,
    disk_gb: float,
    min_ram_gb: int,
    min_chip_gen: int,
    note: str,
) -> ModelEntry:
    return ModelEntry(
        repo_id=repo_id,
        quant_bits=quant_bits,
        disk_gb=disk_gb,
        min_ram_gb=min_ram_gb,
        min_chip_gen=min_chip_gen,
        note=note,
        downloads=None,
        likes=None,
    )


_OFFLINE_CATALOG: list[ModelEntry] = [
    # ── 4 GB RAM tier ────────────────────────────────────────────────────────
    _entry("mlx-community/Llama-3.2-1B-Instruct-4bit",      4,  0.6,  2,  1, "Tiny 1B — fastest on any M-series chip"),
    _entry("mlx-community/Llama-3.2-3B-Instruct-4bit",      4,  1.8,  4,  1, "Solid 3B all-rounder"),
    _entry("mlx-community/Phi-3.5-mini-instruct-4bit",       4,  2.2,  4,  1, "Microsoft 3.8B, strong reasoning"),
    _entry("mlx-community/Qwen2.5-3B-Instruct-4bit",         4,  1.8,  4,  1, "Alibaba 3B, multilingual"),
    # ── 8 GB RAM tier ────────────────────────────────────────────────────────
    _entry("mlx-community/Mistral-7B-Instruct-v0.3-4bit",   4,  3.9,  6,  1, "Classic 7B instruction model"),
    _entry("mlx-community/Llama-3.1-8B-Instruct-4bit",      4,  4.4,  6,  1, "Meta 8B — strong coding and chat"),
    _entry("mlx-community/Qwen2.5-7B-Instruct-4bit",        4,  4.3,  6,  1, "Alibaba 7B, excellent multilingual"),
    _entry("mlx-community/gemma-3-9b-it-4bit",              4,  5.5,  8,  1, "Google 9B, multimodal-ready"),
    # ── 16 GB RAM tier ───────────────────────────────────────────────────────
    _entry("mlx-community/Mistral-Nemo-Instruct-2407-4bit", 4,  7.0, 10,  1, "Mistral 12B, long context"),
    _entry("mlx-community/Phi-4-4bit",                      4,  8.4, 10,  1, "Microsoft 14B, top reasoning"),
    _entry("mlx-community/Qwen2.5-14B-Instruct-4bit",       4,  8.4, 12,  1, "Alibaba 14B"),
    # NOTE: gpt-oss-20b — disk_gb=11.0 is the MLX Q4 weight footprint. min_ram_gb=16
    # reflects real-world Apple Silicon needs (weights + KV cache + activation buffers);
    # the 12 GB OpenAI quotes is weights-only and causes paging on M4 Air at 16 GB.
    # When the download subcommand lands: use catalog disk_gb (11.0) for the preflight
    # disk-space check rather than inferring from Hub metadata — the Hub repo size for
    # gpt-oss variants varies by revision and can be misleading.
    # The mlx-community repo is the correct target for AppleLocalLM; MXFP4 native
    # weights live on a separate repo with different performance characteristics.
    _entry("mlx-community/gpt-oss-20b-4bit",                4, 11.0, 16,  1, "OpenAI 20B MoE, o3-mini quality, reasoning effort configurable"),
    # ── 32 GB RAM tier ───────────────────────────────────────────────────────
    _entry("mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit", 4, 13.0, 16, 1, "Mistral 24B, strong all-rounder"),
    _entry("mlx-community/Qwen2.5-32B-Instruct-4bit",       4, 18.0, 24,  1, "Alibaba 32B, near-frontier quality"),
    # ── 64 GB RAM tier ───────────────────────────────────────────────────────
    _entry("mlx-community/Llama-3.3-70B-Instruct-4bit",     4, 38.0, 40,  1, "Meta 70B, near-frontier quality"),
    _entry("mlx-community/Qwen2.5-72B-Instruct-4bit",       4, 41.0, 48,  1, "Alibaba 72B, excellent reasoning"),
]


# ---------------------------------------------------------------------------
# Name-based inference helpers (for online results)
# ---------------------------------------------------------------------------


def _infer_params_b(repo_id: str) -> float | None:
    """Infer parameter count in billions from a repo_id string."""
    match = re.search(r"(\d+(?:\.\d+)?)B", repo_id, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        return value if value > 0 else None
    return None


def _infer_quant_bits(repo_id: str) -> int:
    """Infer quantisation level in bits from a repo_id string. Defaults to 4."""
    lower = repo_id.lower()
    for bits, patterns in [
        (3,  ["3bit", "q3"]),
        (4,  ["4bit", "q4"]),
        (6,  ["6bit", "q6"]),
        (8,  ["8bit", "q8"]),
        (16, ["fp16", "bf16", "fp32"]),
    ]:
        if any(p in lower for p in patterns):
            return bits
    return 4  # most mlx-community models are 4-bit


def _infer_disk_gb(params_b: float, quant_bits: int) -> float:
    """Estimate download size: params × bytes-per-weight × 1.1 overhead."""
    bytes_per_param = quant_bits / 8
    raw = params_b * 1e9 * bytes_per_param / (1024**3)
    return round(raw * 1.1, 1)


def _infer_min_ram_gb(params_b: float, quant_bits: int) -> int:
    """Estimate minimum RAM: model weights + KV/activation overhead."""
    model_gb = _infer_disk_gb(params_b, quant_bits)
    overhead_gb = max(2.0, model_gb * 0.15)
    return max(2, int(model_gb + overhead_gb) + 1)


# ---------------------------------------------------------------------------
# Hardware filtering
# ---------------------------------------------------------------------------


def _filter_offline(hw: HardwareInfo) -> list[ModelEntry]:
    results = []
    for entry in _OFFLINE_CATALOG:
        if entry["min_ram_gb"] > hw["ram_gb"]:
            continue
        # Only apply disk filter when we have a real reading (>0)
        if hw["free_disk_gb"] > 0 and (entry["disk_gb"] + 1.0) > hw["free_disk_gb"]:
            continue
        if hw["chip_gen"] > 0 and entry["min_chip_gen"] > hw["chip_gen"]:
            continue
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Online fetch
# ---------------------------------------------------------------------------


def _fetch_online_suggestions(hw: HardwareInfo) -> list[ModelEntry] | None:
    """Query HF Hub for mlx-community models, filtered by *hw*.

    Returns a list of :class:`ModelEntry` on success, or ``None`` if the
    network is unavailable or any error occurs (caller should fall back to
    the offline catalog).

    Size estimates are name-based heuristics — see module docstring.
    """
    try:
        from huggingface_hub import list_models
    except ImportError:
        logger.debug("huggingface_hub unavailable; cannot fetch online suggestions")
        return None

    try:
        raw_models = list(
            list_models(
                author="mlx-community",
                sort="downloads",
                direction=-1,
                limit=100,
            )
        )
    except Exception as exc:  # network down, timeout, API error — all fall back
        logger.debug("HF Hub fetch failed (%s); falling back to offline catalog", exc)
        return None

    entries: list[ModelEntry] = []
    for model in raw_models:
        repo_id = model.id
        # Guard: model.id should always be a non-empty string, but the
        # huggingface_hub API does not enforce this at the type level.
        if not isinstance(repo_id, str) or not repo_id:
            logger.debug("Skipping model with non-string id: %r", repo_id)
            continue
        quant_bits = _infer_quant_bits(repo_id)
        params_b = _infer_params_b(repo_id)

        if params_b is None:
            # Cannot estimate hardware requirements without a parameter count
            continue

        disk_gb = _infer_disk_gb(params_b, quant_bits)
        min_ram_gb = _infer_min_ram_gb(params_b, quant_bits)

        if min_ram_gb > hw["ram_gb"]:
            continue
        if hw["free_disk_gb"] > 0 and (disk_gb + 1.0) > hw["free_disk_gb"]:
            continue

        entries.append(
            ModelEntry(
                repo_id=repo_id,
                quant_bits=quant_bits,
                disk_gb=disk_gb,
                min_ram_gb=min_ram_gb,
                min_chip_gen=1,
                note=f"~{params_b:.0f}B params, {quant_bits}-bit",
                downloads=getattr(model, "downloads", None),
                likes=getattr(model, "likes", None),
            )
        )

    return entries or None  # treat empty list as "no useful results" → fall back


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def suggest_models(
    hw: HardwareInfo | None = None,
    online: bool = True,
) -> SuggestResult:
    """Return MLX model suggestions appropriate for the local hardware.

    Args:
        hw: Hardware profile from :func:`~apple_basefm._hardware.detect_hardware`.
            When ``None``, hardware is detected automatically.
        online: When ``True`` (default), query the HuggingFace Hub for the
            latest ``mlx-community`` models before falling back to the
            built-in offline catalog. Pass ``False`` to skip the network call.

    Returns:
        A :class:`SuggestResult` with ``models`` (sorted best-first by
        ``min_ram_gb`` descending), ``source`` (``"online"`` or
        ``"offline"``), and ``hardware``.
    """
    if hw is None:
        from apple_basefm._hardware import detect_hardware

        hw = detect_hardware()

    if not hw["is_apple_silicon"]:
        logger.debug("Not Apple Silicon — no model suggestions available")
        return SuggestResult(models=[], source="offline", hardware=hw)

    source: Literal["online", "offline"] = "offline"
    models: list[ModelEntry] | None = None

    if online:
        models = _fetch_online_suggestions(hw)
        if models is not None:
            source = "online"

    if models is None:
        models = _filter_offline(hw)
        source = "offline"

    # Sort best-fit first: largest model that still fits (min_ram_gb desc),
    # then by repo_id for stable ordering within a tier.
    models.sort(key=lambda m: (-m["min_ram_gb"], m["repo_id"]))

    return SuggestResult(models=models, source=source, hardware=hw)
