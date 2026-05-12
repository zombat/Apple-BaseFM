"""Apple Silicon hardware detection for model recommendation.

Provides :func:`detect_hardware`, which reads chip generation, unified
memory size, and available disk space from the local macOS system without
any third-party dependencies.

On non-macOS platforms (Linux, Windows, CI) the function returns a
:class:`HardwareInfo` with ``is_apple_silicon=False`` and zeroed numeric
fields — callers should check ``is_apple_silicon`` before using the values
for filtering.

All subprocess calls use ``shell=False`` and pass arguments as a list to
prevent command injection. No user-supplied data is ever passed to a
subprocess.
"""
from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
from typing import TypedDict

logger = logging.getLogger(__name__)


class HardwareInfo(TypedDict):
    """Hardware profile of the local Mac.

    Attributes:
        chip: Human-readable chip name, e.g. ``"Apple M4 Pro"``.
            ``None`` if detection failed or the platform is not macOS.
        chip_gen: Integer generation number (1 = M1, 2 = M2, 3 = M3, 4 = M4).
            ``0`` if unknown or non-Apple-Silicon.
        ram_gb: Unified memory in whole gigabytes (floor division).
        free_disk_gb: Free space on the root volume in whole gigabytes.
        is_apple_silicon: ``True`` only when running on Apple Silicon macOS.
    """

    chip: str | None
    chip_gen: int
    ram_gb: int
    free_disk_gb: int
    is_apple_silicon: bool


def _unknown() -> HardwareInfo:
    return HardwareInfo(
        chip=None,
        chip_gen=0,
        ram_gb=0,
        free_disk_gb=0,
        is_apple_silicon=False,
    )


def _run_sysctl(key: str) -> str | None:
    """Return the string value of a sysctl key, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("sysctl %s failed: %s", key, exc)
    return None


def _chip_name_from_system_profiler() -> str | None:
    """Return the chip name via ``system_profiler SPHardwareDataType -json``.

    Example returned value: ``"Apple M4 Pro"``.
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        # Path: SPHardwareDataType[0].chip_type
        hw_items = data.get("SPHardwareDataType", [])
        if hw_items:
            return hw_items[0].get("chip_type") or hw_items[0].get("cpu_type")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.debug("system_profiler chip detection failed: %s", exc)
    return None


def _parse_chip_gen(chip_name: str | None) -> int:
    """Extract the M-series generation integer from a chip name string.

    Args:
        chip_name: e.g. ``"Apple M4 Pro"``, ``"Apple M2"``, ``"Apple M1 Ultra"``.

    Returns:
        Generation integer (1–4+), or ``0`` if the generation cannot be parsed.
    """
    if not chip_name:
        return 0
    import re

    match = re.search(r"\bM(\d+)\b", chip_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def _ram_gb_from_sysctl() -> int:
    """Return unified memory in GB from ``hw.memsize``."""
    raw = _run_sysctl("hw.memsize")
    if raw:
        try:
            return int(raw) // (1024 ** 3)
        except ValueError:
            pass
    return 0


def _free_disk_gb(path: str = "/") -> int:
    """Return free disk space on *path* in whole gigabytes."""
    try:
        usage = shutil.disk_usage(path)
        return int(usage.free) // (1024 ** 3)
    except OSError as exc:
        logger.debug("disk_usage(%s) failed: %s", path, exc)
        return 0


def detect_hardware() -> HardwareInfo:
    """Detect Apple Silicon hardware profile of the current machine.

    Returns a :class:`HardwareInfo` dict. On non-macOS platforms all numeric
    fields are ``0`` and ``is_apple_silicon`` is ``False``.

    The function never raises — any detection failure is logged at DEBUG level
    and results in zeroed / ``None`` fields.

    Returns:
        A :class:`HardwareInfo` with chip name, generation, RAM, and free disk.
    """
    if platform.system() != "Darwin":
        logger.debug("Not macOS — hardware detection skipped")
        return _unknown()

    # Detect chip via system_profiler (most accurate), fallback to hw.model
    chip = _chip_name_from_system_profiler()
    if not chip:
        # hw.model returns identifiers like "MacBookPro18,2"; not human-readable
        # but chip_gen can still be estimated via hw.optional.arm.FEAT_SME (M3+)
        logger.debug("system_profiler chip detection failed; falling back to sysctl")
        model = _run_sysctl("hw.model")
        # We can't reliably derive the chip name from the model identifier alone
        # without a lookup table — leave chip as None, gen as 0
        chip = None
        _ = model  # retained for future lookup table if added

    chip_gen = _parse_chip_gen(chip)
    ram_gb = _ram_gb_from_sysctl()
    free_disk_gb = _free_disk_gb()

    # Confirm Apple Silicon by checking the CPU architecture.
    # On Apple Silicon Macs running under Rosetta 2, platform.machine()
    # reports "x86_64".  Falling back to hw.optional.arm64 (returns "1" on
    # all Apple Silicon Macs regardless of the process architecture) lets us
    # correctly detect the hardware and emit a useful warning.
    is_apple_silicon = platform.machine() == "arm64"
    if not is_apple_silicon:
        arm64_flag = _run_sysctl("hw.optional.arm64")
        if arm64_flag == "1":
            logger.warning(
                "Running under Rosetta 2 (x86_64 process on Apple Silicon). "
                "Re-run as a native arm64 process for accurate model suggestions."
            )
            is_apple_silicon = True

    if is_apple_silicon and ram_gb == 0:
        logger.warning(
            "Apple Silicon detected but unified memory size could not be "
            "determined (sysctl hw.memsize failed). Model suggestions will be "
            "empty. Ensure sysctl is on PATH or set HF_HUB_CACHE explicitly."
        )

    info = HardwareInfo(
        chip=chip,
        chip_gen=chip_gen,
        ram_gb=ram_gb,
        free_disk_gb=free_disk_gb,
        is_apple_silicon=is_apple_silicon,
    )
    logger.debug("Detected hardware: %s", info)
    return info
