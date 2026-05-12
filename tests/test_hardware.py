"""Unit tests for apple_basefm._hardware.

All subprocess calls are mocked — no real sysctl/system_profiler is ever
invoked.  Tests pass on Linux CI as well as macOS.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import apple_basefm._hardware as hw_mod
from apple_basefm._hardware import (
    HardwareInfo,
    _free_disk_gb,
    _parse_chip_gen,
    _ram_gb_from_sysctl,
    _run_sysctl,
    detect_hardware,
)


# ---------------------------------------------------------------------------
# _parse_chip_gen
# ---------------------------------------------------------------------------


class TestParseChipGen:
    def test_m4_pro(self) -> None:
        assert _parse_chip_gen("Apple M4 Pro") == 4

    def test_m1(self) -> None:
        assert _parse_chip_gen("Apple M1") == 1

    def test_m2_ultra(self) -> None:
        assert _parse_chip_gen("Apple M2 Ultra") == 2

    def test_m3_max(self) -> None:
        assert _parse_chip_gen("Apple M3 Max") == 3

    def test_none_returns_zero(self) -> None:
        assert _parse_chip_gen(None) == 0

    def test_empty_string_returns_zero(self) -> None:
        assert _parse_chip_gen("") == 0

    def test_intel_returns_zero(self) -> None:
        assert _parse_chip_gen("Intel Core i9") == 0

    def test_no_generation_number_returns_zero(self) -> None:
        assert _parse_chip_gen("Apple Silicon") == 0

    def test_case_insensitive(self) -> None:
        assert _parse_chip_gen("apple m4") == 4

    def test_large_generation(self) -> None:
        assert _parse_chip_gen("Apple M10 Ultra") == 10


# ---------------------------------------------------------------------------
# _run_sysctl
# ---------------------------------------------------------------------------


def _make_sysctl_result(stdout: str, returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


class TestRunSysctl:
    def test_returns_stripped_value(self) -> None:
        with patch("subprocess.run", return_value=_make_sysctl_result("  42  \n")) as mock_run:
            val = _run_sysctl("hw.memsize")
        assert val == "42"
        mock_run.assert_called_once_with(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_non_zero_returncode_returns_none(self) -> None:
        with patch("subprocess.run", return_value=_make_sysctl_result("", returncode=1)):
            val = _run_sysctl("hw.memsize")
        assert val is None

    def test_file_not_found_returns_none(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError("sysctl not found")):
            val = _run_sysctl("hw.memsize")
        assert val is None

    def test_timeout_returns_none(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sysctl", timeout=5)):
            val = _run_sysctl("hw.memsize")
        assert val is None

    def test_os_error_returns_none(self) -> None:
        with patch("subprocess.run", side_effect=OSError("unexpected")):
            val = _run_sysctl("hw.memsize")
        assert val is None


# ---------------------------------------------------------------------------
# _ram_gb_from_sysctl
# ---------------------------------------------------------------------------


class TestRamGbFromSysctl:
    def test_8gb(self) -> None:
        # 8 * 1024^3 = 8589934592
        with patch.object(hw_mod, "_run_sysctl", return_value="8589934592"):
            assert _ram_gb_from_sysctl() == 8

    def test_16gb(self) -> None:
        with patch.object(hw_mod, "_run_sysctl", return_value="17179869184"):
            assert _ram_gb_from_sysctl() == 16

    def test_none_returns_zero(self) -> None:
        with patch.object(hw_mod, "_run_sysctl", return_value=None):
            assert _ram_gb_from_sysctl() == 0

    def test_invalid_value_returns_zero(self) -> None:
        with patch.object(hw_mod, "_run_sysctl", return_value="not-a-number"):
            assert _ram_gb_from_sysctl() == 0


# ---------------------------------------------------------------------------
# _free_disk_gb
# ---------------------------------------------------------------------------


class TestFreeDiskGb:
    def test_returns_floor_gb(self) -> None:
        usage = MagicMock()
        usage.free = 100 * (1024 ** 3)  # exactly 100 GB
        with patch("shutil.disk_usage", return_value=usage):
            assert _free_disk_gb() == 100

    def test_floors_partial_gb(self) -> None:
        usage = MagicMock()
        usage.free = int(15.9 * (1024 ** 3))
        with patch("shutil.disk_usage", return_value=usage):
            assert _free_disk_gb() == 15

    def test_os_error_returns_zero(self) -> None:
        with patch("shutil.disk_usage", side_effect=OSError("no disk")):
            assert _free_disk_gb() == 0


# ---------------------------------------------------------------------------
# detect_hardware
# ---------------------------------------------------------------------------


def _sp_json_output(chip_type: str) -> str:
    return json.dumps({"SPHardwareDataType": [{"chip_type": chip_type}]})


def _make_sp_result(chip: str) -> MagicMock:
    result = MagicMock()
    result.returncode = 0
    result.stdout = _sp_json_output(chip)
    return result


def _make_sysctl_run(memsize: str = "8589934592") -> Any:
    """Return a subprocess.run side_effect that routes commands."""

    def _run(args: list[str], **kwargs: Any) -> MagicMock:
        cmd = args[0]
        if cmd == "system_profiler":
            return _make_sp_result("Apple M2 Pro")
        # sysctl: dispatch on key
        key = args[2] if len(args) > 2 else ""
        mock = MagicMock()
        mock.returncode = 0
        if key == "hw.memsize":
            mock.stdout = memsize
        elif key == "hw.optional.arm64":
            mock.stdout = "0"
        else:
            mock.stdout = ""
        return mock

    return _run


class TestDetectHardware:
    def test_non_darwin_returns_unknown(self) -> None:
        with patch("platform.system", return_value="Linux"):
            result = detect_hardware()
        assert result["is_apple_silicon"] is False
        assert result["chip"] is None
        assert result["chip_gen"] == 0
        assert result["ram_gb"] == 0

    def test_non_darwin_windows_returns_unknown(self) -> None:
        with patch("platform.system", return_value="Windows"):
            result = detect_hardware()
        assert result["is_apple_silicon"] is False

    def test_darwin_arm64_detects_chip(self) -> None:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("subprocess.run", side_effect=_make_sysctl_run()),
            patch("shutil.disk_usage", return_value=MagicMock(free=50 * 1024 ** 3)),
        ):
            result = detect_hardware()
        assert result["is_apple_silicon"] is True
        assert result["chip"] == "Apple M2 Pro"
        assert result["chip_gen"] == 2
        assert result["ram_gb"] == 8
        assert result["free_disk_gb"] == 50

    def test_darwin_arm64_correct_types(self) -> None:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("subprocess.run", side_effect=_make_sysctl_run()),
            patch("shutil.disk_usage", return_value=MagicMock(free=50 * 1024 ** 3)),
        ):
            result = detect_hardware()
        assert isinstance(result["chip"], str)
        assert isinstance(result["chip_gen"], int)
        assert isinstance(result["ram_gb"], int)
        assert isinstance(result["free_disk_gb"], int)
        assert isinstance(result["is_apple_silicon"], bool)

    def test_rosetta2_fallback(self) -> None:
        """x86_64 process on Apple Silicon triggers sysctl hw.optional.arm64 fallback."""

        def _run(args: list[str], **kwargs: Any) -> MagicMock:
            cmd = args[0]
            mock = MagicMock()
            mock.returncode = 0
            if cmd == "system_profiler":
                mock.stdout = _sp_json_output("Apple M1")
            elif args[2] == "hw.memsize":
                mock.stdout = "8589934592"
            elif args[2] == "hw.optional.arm64":
                mock.stdout = "1"
            else:
                mock.stdout = ""
            return mock

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
            patch("subprocess.run", side_effect=_run),
            patch("shutil.disk_usage", return_value=MagicMock(free=30 * 1024 ** 3)),
        ):
            result = detect_hardware()
        assert result["is_apple_silicon"] is True
        assert result["chip_gen"] == 1

    def test_x86_non_apple_silicon(self) -> None:
        """x86_64 with hw.optional.arm64 = 0 is not Apple Silicon."""

        def _run(args: list[str], **kwargs: Any) -> MagicMock:
            cmd = args[0]
            mock = MagicMock()
            mock.returncode = 0
            if cmd == "system_profiler":
                mock.stdout = "{}"  # no chip
            elif args[2] == "hw.memsize":
                mock.stdout = "8589934592"
            elif args[2] == "hw.optional.arm64":
                mock.stdout = "0"
            else:
                mock.stdout = ""
            return mock

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
            patch("subprocess.run", side_effect=_run),
            patch("shutil.disk_usage", return_value=MagicMock(free=30 * 1024 ** 3)),
        ):
            result = detect_hardware()
        assert result["is_apple_silicon"] is False

    def test_system_profiler_timeout_falls_back(self) -> None:
        """system_profiler timeout leaves chip=None but does not crash."""

        def _run(args: list[str], **kwargs: Any) -> MagicMock:
            if args[0] == "system_profiler":
                raise subprocess.TimeoutExpired(cmd="system_profiler", timeout=15)
            mock = MagicMock()
            mock.returncode = 0
            key = args[2] if len(args) > 2 else ""
            mock.stdout = "8589934592" if key == "hw.memsize" else "0"
            return mock

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("subprocess.run", side_effect=_run),
            patch("shutil.disk_usage", return_value=MagicMock(free=20 * 1024 ** 3)),
        ):
            result = detect_hardware()
        assert result["chip"] is None
        assert result["chip_gen"] == 0
        assert result["ram_gb"] == 8
        assert result["is_apple_silicon"] is True

    def test_system_profiler_bad_json_falls_back(self) -> None:
        """Malformed JSON from system_profiler leaves chip=None."""

        def _run(args: list[str], **kwargs: Any) -> MagicMock:
            mock = MagicMock()
            mock.returncode = 0
            if args[0] == "system_profiler":
                mock.stdout = "not-json"
            else:
                key = args[2] if len(args) > 2 else ""
                mock.stdout = "8589934592" if key == "hw.memsize" else "0"
            return mock

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("subprocess.run", side_effect=_run),
            patch("shutil.disk_usage", return_value=MagicMock(free=20 * 1024 ** 3)),
        ):
            result = detect_hardware()
        assert result["chip"] is None
        assert result["is_apple_silicon"] is True

    def test_ram_zero_on_apple_silicon_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """detect_hardware warns when is_apple_silicon=True but ram_gb=0."""

        def _run(args: list[str], **kwargs: Any) -> MagicMock:
            mock = MagicMock()
            mock.returncode = 1  # sysctl fails
            mock.stdout = ""
            if args[0] == "system_profiler":
                mock.returncode = 0
                mock.stdout = _sp_json_output("Apple M2")
            return mock

        import logging

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("subprocess.run", side_effect=_run),
            patch("shutil.disk_usage", return_value=MagicMock(free=0)),
            caplog.at_level(logging.WARNING, logger="apple_basefm._hardware"),
        ):
            result = detect_hardware()
        assert result["ram_gb"] == 0
        assert result["is_apple_silicon"] is True
        assert any("unified memory" in r.message for r in caplog.records)

    def test_never_raises_on_os_errors(self) -> None:
        """detect_hardware must not raise on expected OS-level failures."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("subprocess.run", side_effect=OSError("exec failed")),
            patch("shutil.disk_usage", side_effect=OSError("no disk")),
        ):
            result = detect_hardware()
        # Should return a valid (zero-ish) HardwareInfo without raising
        assert isinstance(result, dict)
        assert "is_apple_silicon" in result
        # is_apple_silicon from platform.machine() = arm64, chip=None (sp failed)
        assert result["is_apple_silicon"] is True
        assert result["chip"] is None
        assert result["ram_gb"] == 0
        assert result["free_disk_gb"] == 0
