"""Unit tests for apple_basefm._cli: download subcommand.

Isolation strategy: all huggingface_hub calls (repo_info, snapshot_download,
scan_cache_dir), hardware detection, and catalog lookups are patched.
No real network calls are made and no real disk writes occur.

Coverage targets (from README.md § CLI: download and EDGECASES.md):
  - REPO_ID format validation (empty, missing slash, path traversal, whitespace)
  - --revision normalisation (empty string → "main")
  - --dry-run flag (prints info, no download)
  - --yes flag (skips confirmation prompt)
  - Preflight: repo exists (RepositoryNotFoundError, GatedRepoError)
  - Preflight: already cached (scan_cache_dir hit)
  - Preflight: disk space check (insufficient, catalog vs Hub metadata, unit conversion)
  - Confirmation prompt: y / n / EOFError / KeyboardInterrupt
  - snapshot_download called with resume_download=True and correct args
  - OSError ENOSPC → resume hint
  - ConnectionError/TimeoutError → mirror hint
  - PermissionError → HF_HUB_CACHE hint
  - HF_TOKEN never logged
  - Final cache path printed on success
"""
from __future__ import annotations

import errno
import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_repo_info(repo_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit") -> MagicMock:
    info = MagicMock()
    info.id = repo_id
    info.siblings = []
    return info


class _FakeRepoNotFound(Exception):
    pass


class _FakeGatedRepo(Exception):
    pass


def _make_fake_hf_hub(
    repo_info_return: Any = None,
    repo_info_side_effect: Any = None,
    snapshot_return: str = "/cache/models--mlx-community--Llama-3.2-3B-Instruct-4bit",
    snapshot_side_effect: Any = None,
    scan_repos: list | None = None,
) -> types.ModuleType:
    fake = types.ModuleType("huggingface_hub")
    if repo_info_side_effect:
        fake.repo_info = MagicMock(side_effect=repo_info_side_effect)
    else:
        fake.repo_info = MagicMock(
            return_value=repo_info_return or _make_fake_repo_info()
        )
    if snapshot_side_effect:
        fake.snapshot_download = MagicMock(side_effect=snapshot_side_effect)
    else:
        fake.snapshot_download = MagicMock(return_value=snapshot_return)

    if scan_repos is not None:
        cache = MagicMock()
        cache.repos = scan_repos
        fake.scan_cache_dir = MagicMock(return_value=cache)
    else:
        empty_cache = MagicMock()
        empty_cache.repos = []
        fake.scan_cache_dir = MagicMock(return_value=empty_cache)

    # Error classes as real exception subclasses
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.CacheNotFound = type("CacheNotFound", (Exception,), {})
    fake_errors.RepositoryNotFoundError = _FakeRepoNotFound
    fake_errors.GatedRepoError = _FakeGatedRepo
    fake.errors = fake_errors
    return fake


def _make_fake_hw(free_disk_gb: float = 100.0) -> types.ModuleType:
    hw_mod = types.ModuleType("apple_basefm._hardware")
    hw = MagicMock()
    hw.free_disk_gb = free_disk_gb
    hw_mod.detect_hardware = MagicMock(return_value=hw)
    # HardwareInfo is a TypedDict; expose a minimal stub so package-level imports don't fail
    hw_mod.HardwareInfo = dict  # type: ignore[attr-defined]
    return hw_mod


def _reload_cli(fake_hf: types.ModuleType, fake_hw: types.ModuleType | None = None) -> types.ModuleType:
    sys.modules["huggingface_hub"] = fake_hf
    sys.modules["huggingface_hub.errors"] = fake_hf.errors
    if fake_hw is not None:
        sys.modules["apple_basefm._hardware"] = fake_hw
    sys.modules.pop("apple_basefm._cli", None)
    return importlib.import_module("apple_basefm._cli")


def _args(
    repo_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    revision: str = "main",
    dry_run: bool = False,
    yes: bool = True,
) -> MagicMock:
    a = MagicMock()
    a.repo_id = repo_id
    a.revision = revision
    a.dry_run = dry_run
    a.yes = yes
    return a


# ===========================================================================
# 1. Input validation — REPO_ID
# ===========================================================================


class TestRepoIdValidation:
    def test_empty_repo_id_exits(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args(repo_id=""))
        assert exc.value.code != 0
        fake_hf.repo_info.assert_not_called()

    def test_whitespace_only_repo_id_exits(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args(repo_id="   "))
        assert exc.value.code != 0
        fake_hf.repo_info.assert_not_called()

    def test_missing_slash_exits(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args(repo_id="llama"))
        assert exc.value.code != 0
        fake_hf.repo_info.assert_not_called()

    def test_path_traversal_exits(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args(repo_id="../../etc/passwd"))
        assert exc.value.code != 0
        fake_hf.repo_info.assert_not_called()

    def test_null_byte_exits(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args(repo_id="mlx-community/model\x00evil"))
        assert exc.value.code != 0
        fake_hf.repo_info.assert_not_called()

    def test_newline_in_repo_id_exits(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args(repo_id="mlx-community/model\ninjected"))
        assert exc.value.code != 0

    def test_valid_repo_id_proceeds(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit"))
        fake_hf.repo_info.assert_called_once()

    def test_repo_id_with_dots_and_dashes_valid(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(repo_id="mlx-community/gpt-oss-20b-4bit"))
        fake_hf.repo_info.assert_called_once()

    def test_error_message_mentions_format(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit):
            cli._cmd_download(_args(repo_id="justname"))
        err = capsys.readouterr().err
        assert "REPO_ID" in err or "owner" in err or "/" in err

    def test_validate_repo_id_helper_raises_on_empty(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(ValueError, match="REPO_ID"):
            cli._validate_repo_id("")

    def test_validate_repo_id_helper_raises_on_no_slash(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(ValueError, match="REPO_ID"):
            cli._validate_repo_id("llama")

    def test_validate_repo_id_helper_accepts_valid(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        result = cli._validate_repo_id("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert result == "mlx-community/Llama-3.2-3B-Instruct-4bit"


# ===========================================================================
# 2. --revision normalisation
# ===========================================================================


class TestRevisionNormalisation:
    def test_empty_revision_becomes_main(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(revision=""))
        _, kwargs = fake_hf.snapshot_download.call_args
        assert kwargs.get("revision") == "main"

    def test_whitespace_revision_becomes_main(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(revision="   "))
        _, kwargs = fake_hf.snapshot_download.call_args
        assert kwargs.get("revision") == "main"

    def test_explicit_revision_passed_through(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(revision="a1b2c3d"))
        _, kwargs = fake_hf.snapshot_download.call_args
        assert kwargs.get("revision") == "a1b2c3d"

    def test_default_revision_is_main(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(revision="main"))
        _, kwargs = fake_hf.snapshot_download.call_args
        assert kwargs.get("revision") == "main"


# ===========================================================================
# 3. Preflight: repo existence
# ===========================================================================


class TestRepoExistencePreflight:
    def test_repo_not_found_exits_with_message(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=_FakeRepoNotFound("not found"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args())
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert "mlx-community/Llama-3.2-3B-Instruct-4bit" in err

    def test_repo_not_found_includes_login_hint(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=_FakeRepoNotFound("not found"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit):
            cli._cmd_download(_args())
        err = capsys.readouterr().err.lower()
        assert "login" in err or "private" in err or "typo" in err

    def test_gated_repo_exits_with_message(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=_FakeGatedRepo("gated"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args())
        assert exc.value.code != 0
        err = capsys.readouterr().err.lower()
        assert "license" in err or "accept" in err or "gated" in err

    def test_gated_repo_includes_hub_url(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=_FakeGatedRepo("gated"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit):
            cli._cmd_download(_args())
        err = capsys.readouterr().err
        assert "huggingface.co" in err

    def test_snapshot_not_called_when_repo_info_fails(self) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=_FakeRepoNotFound("not found"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit):
            cli._cmd_download(_args())
        fake_hf.snapshot_download.assert_not_called()

    def test_connection_error_exits_with_mirror_hint(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=ConnectionError("unreachable"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args())
        assert exc.value.code != 0
        err = capsys.readouterr().err.lower()
        assert "mirror" in err or "hf_endpoint" in err or "network" in err

    def test_value_error_from_bad_endpoint_exits_with_hint(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=ValueError("bad endpoint"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit):
            cli._cmd_download(_args())
        err = capsys.readouterr().err.lower()
        assert "endpoint" in err or "hf_endpoint" in err

    def test_hf_token_not_in_error_output(self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        token = "hf_supersecret99999"
        monkeypatch.setenv("HF_TOKEN", token)
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=_FakeRepoNotFound("not found"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit):
            cli._cmd_download(_args())
        captured = capsys.readouterr()
        assert token not in captured.out
        assert token not in captured.err


# ===========================================================================
# 4. Preflight: already cached
# ===========================================================================


class TestAlreadyCachedPreflight:
    def test_already_cached_prints_path_and_skips_download(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        cached_path = "/cache/models--mlx-community--Llama-3.2-3B-Instruct-4bit"
        cached_repo = MagicMock()
        cached_repo.repo_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"
        cached_repo.repo_path = cached_path
        fake_hf = _make_fake_hf_hub(scan_repos=[cached_repo])
        cli = _reload_cli(fake_hf)
        cli._cmd_download(_args())
        fake_hf.snapshot_download.assert_not_called()
        out = capsys.readouterr().out
        assert cached_path in out

    def test_already_cached_case_insensitive_match(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        cached_repo = MagicMock()
        cached_repo.repo_id = "MLX-Community/Llama-3.2-3B-Instruct-4bit"
        cached_repo.repo_path = "/cache/model"
        fake_hf = _make_fake_hf_hub(scan_repos=[cached_repo])
        cli = _reload_cli(fake_hf)
        cli._cmd_download(_args(repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit"))
        fake_hf.snapshot_download.assert_not_called()

    def test_different_cached_model_does_not_block_download(self) -> None:
        other_repo = MagicMock()
        other_repo.repo_id = "mlx-community/Mistral-7B-4bit"
        other_repo.repo_path = "/cache/other"
        fake_hf = _make_fake_hf_hub(scan_repos=[other_repo])
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args())
        fake_hf.snapshot_download.assert_called_once()

    def test_scan_cache_dir_error_is_non_fatal(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hf.scan_cache_dir = MagicMock(side_effect=RuntimeError("scan error"))
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args())
        fake_hf.snapshot_download.assert_called_once()


# ===========================================================================
# 5. Preflight: disk space
# ===========================================================================


class TestDiskSpacePreflight:
    def test_insufficient_disk_exits_without_download(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw(free_disk_gb=0.5)
        cli = _reload_cli(fake_hf, fake_hw)
        # patch catalog to return 2.0 GB requirement
        with patch.object(cli, "_catalog_disk_gb", return_value=2.0):
            with pytest.raises(SystemExit) as exc:
                cli._cmd_download(_args())
        assert exc.value.code != 0
        fake_hf.snapshot_download.assert_not_called()

    def test_sufficient_disk_proceeds(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw(free_disk_gb=100.0)
        cli = _reload_cli(fake_hf, fake_hw)
        with patch.object(cli, "_catalog_disk_gb", return_value=4.0):
            cli._cmd_download(_args())
        fake_hf.snapshot_download.assert_called_once()

    def test_insufficient_disk_error_mentions_size(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw(free_disk_gb=0.5)
        cli = _reload_cli(fake_hf, fake_hw)
        with patch.object(cli, "_catalog_disk_gb", return_value=2.0):
            with pytest.raises(SystemExit):
                cli._cmd_download(_args())
        err = capsys.readouterr().err.lower()
        assert "disk" in err or "space" in err

    def test_catalog_disk_gb_preferred_over_hub_metadata(self) -> None:
        # Hub siblings report 0 bytes; catalog returns 4.0 GB — both ok, just verify
        # catalog is consulted
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw(free_disk_gb=100.0)
        cli = _reload_cli(fake_hf, fake_hw)
        catalog_calls: list[str] = []

        def _recording(repo_id: str) -> float | None:
            catalog_calls.append(repo_id)
            return 4.0

        with patch.object(cli, "_catalog_disk_gb", side_effect=_recording):
            cli._cmd_download(_args())
        assert catalog_calls, "catalog must be consulted during disk preflight"

    def test_zero_disk_estimate_skips_preflight_gracefully(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw(free_disk_gb=1.0)
        cli = _reload_cli(fake_hf, fake_hw)
        with patch.object(cli, "_catalog_disk_gb", return_value=0.0):
            cli._cmd_download(_args())
        fake_hf.snapshot_download.assert_called_once()

    def test_hub_size_gb_helper_converts_bytes_to_gb(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        info = MagicMock()
        s = MagicMock()
        s.size = 5_000_000_000  # 5 GB in bytes (using 1e9 denominator)
        info.siblings = [s]
        result = cli._hub_size_gb(info)
        assert result is not None
        assert 4.9 < result < 5.1

    def test_hub_size_gb_none_siblings_returns_none(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        info = MagicMock()
        info.siblings = []
        assert cli._hub_size_gb(info) is None

    def test_hub_size_gb_none_size_fields_returns_none(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        info = MagicMock()
        s = MagicMock()
        s.size = None
        info.siblings = [s]
        assert cli._hub_size_gb(info) is None

    def test_hardware_detection_failure_skips_disk_check(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        fake_hw.detect_hardware = MagicMock(side_effect=RuntimeError("no hw"))
        cli = _reload_cli(fake_hf, fake_hw)
        with patch.object(cli, "_catalog_disk_gb", return_value=4.0):
            cli._cmd_download(_args())
        fake_hf.snapshot_download.assert_called_once()


# ===========================================================================
# 6. --dry-run flag
# ===========================================================================


class TestDryRun:
    def test_dry_run_does_not_call_snapshot_download(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(dry_run=True))
        fake_hf.snapshot_download.assert_not_called()

    def test_dry_run_prints_repo_id(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(dry_run=True))
        out = capsys.readouterr().out
        assert "mlx-community/Llama-3.2-3B-Instruct-4bit" in out

    def test_dry_run_prints_gb_unit(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch.object(cli, "_catalog_disk_gb", return_value=4.2):
            cli._cmd_download(_args(dry_run=True))
        out = capsys.readouterr().out
        assert "GB" in out

    def test_dry_run_prints_revision(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(dry_run=True, revision="a1b2c3d"))
        out = capsys.readouterr().out
        assert "a1b2c3d" in out

    def test_dry_run_missing_repo_still_exits(self) -> None:
        fake_hf = _make_fake_hf_hub(repo_info_side_effect=_FakeRepoNotFound("nope"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args(dry_run=True))
        assert exc.value.code != 0


# ===========================================================================
# 7. --yes flag / confirmation prompt
# ===========================================================================


class TestYesFlag:
    def test_yes_skips_input(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch("builtins.input") as mock_input:
            cli._cmd_download(_args(yes=True))
        mock_input.assert_not_called()

    def test_without_yes_calls_input(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch("builtins.input", return_value="y") as mock_input:
            cli._cmd_download(_args(yes=False))
        mock_input.assert_called_once()

    def test_user_types_n_aborts(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch("builtins.input", return_value="n"):
            cli._cmd_download(_args(yes=False))
        fake_hf.snapshot_download.assert_not_called()

    def test_user_types_y_proceeds(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch("builtins.input", return_value="y"):
            cli._cmd_download(_args(yes=False))
        fake_hf.snapshot_download.assert_called_once()

    def test_eoferror_on_input_exits_cleanly(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch("builtins.input", side_effect=EOFError):
            try:
                cli._cmd_download(_args(yes=False))
            except SystemExit as exc:
                assert exc.code != 0
            except EOFError:
                pytest.fail("EOFError must be caught internally, not propagated")
        fake_hf.snapshot_download.assert_not_called()

    def test_keyboard_interrupt_on_input_exits_cleanly(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            try:
                cli._cmd_download(_args(yes=False))
            except SystemExit as exc:
                assert exc.code != 0
            except KeyboardInterrupt:
                pytest.fail("KeyboardInterrupt must be caught internally, not propagated")
        fake_hf.snapshot_download.assert_not_called()


# ===========================================================================
# 8. snapshot_download call contract
# ===========================================================================


class TestSnapshotDownloadContract:
    def test_resume_download_true(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args())
        _, kwargs = fake_hf.snapshot_download.call_args
        assert kwargs.get("resume_download") is True

    def test_repo_id_passed_correctly(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit"))
        args_call, kwargs = fake_hf.snapshot_download.call_args
        passed_id = args_call[0] if args_call else kwargs.get("repo_id")
        assert "Llama-3.2-3B-Instruct-4bit" in passed_id or "llama-3.2-3b" in passed_id.lower()

    def test_revision_passed_correctly(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args(revision="abc1234"))
        _, kwargs = fake_hf.snapshot_download.call_args
        assert kwargs.get("revision") == "abc1234"

    def test_cache_path_printed_on_success(self, capsys: pytest.CaptureFixture) -> None:
        expected_path = "/Users/user/.cache/huggingface/hub/models--mlx"
        fake_hf = _make_fake_hf_hub(snapshot_return=expected_path)
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args())
        out = capsys.readouterr().out
        assert expected_path in out

    def test_hf_token_not_in_stdout(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = "hf_supersecrettoken12345"
        monkeypatch.setenv("HF_TOKEN", token)
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        cli._cmd_download(_args())
        captured = capsys.readouterr()
        assert token not in captured.out
        assert token not in captured.err


# ===========================================================================
# 9. snapshot_download failure paths
# ===========================================================================


class TestSnapshotDownloadFailurePaths:
    def test_enospc_exits_with_resume_hint(self, capsys: pytest.CaptureFixture) -> None:
        err = OSError(errno.ENOSPC, "No space left on device")
        fake_hf = _make_fake_hf_hub(snapshot_side_effect=err)
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args())
        assert exc.value.code != 0
        out = capsys.readouterr().err.lower()
        assert "resume" in out or "space" in out

    def test_permission_error_exits_with_cache_hint(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(snapshot_side_effect=PermissionError("denied"))
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args())
        assert exc.value.code != 0
        err = capsys.readouterr().err.lower()
        assert "permission" in err or "hf_hub_cache" in err or "cache" in err

    def test_connection_error_exits_with_mirror_hint(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(snapshot_side_effect=ConnectionError("unreachable"))
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args())
        assert exc.value.code != 0
        err = capsys.readouterr().err.lower()
        assert "mirror" in err or "hf_endpoint" in err or "network" in err

    def test_timeout_error_exits_with_mirror_hint(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub(snapshot_side_effect=TimeoutError("timed out"))
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_download(_args())
        assert exc.value.code != 0


# ===========================================================================
# 10. main() integration — download subcommand wired up
# ===========================================================================


class TestMainDownloadIntegration:
    def test_main_download_subcommand_dispatches(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        with patch.object(cli, "_cmd_download") as mock_cmd:
            cli.main(["download", "mlx-community/Llama-3.2-3B-Instruct-4bit", "--yes"])
        mock_cmd.assert_called_once()

    def test_main_download_requires_repo_id(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli.main(["download"])
        assert exc.value.code != 0

    def test_main_download_dry_run_flag_parsed(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        captured_args: list[Any] = []

        def _capture(args: Any) -> None:
            captured_args.append(args)

        with patch.object(cli, "_cmd_download", side_effect=_capture):
            cli.main(["download", "mlx-community/Llama-3.2-3B-Instruct-4bit", "--dry-run"])
        assert captured_args[0].dry_run is True

    def test_main_download_revision_flag_parsed(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        captured_args: list[Any] = []

        def _capture(args: Any) -> None:
            captured_args.append(args)

        with patch.object(cli, "_cmd_download", side_effect=_capture):
            cli.main(["download", "mlx-community/Llama-3.2-3B-Instruct-4bit", "--revision", "abc1234"])
        assert captured_args[0].revision == "abc1234"

    def test_main_download_yes_flag_parsed(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        captured_args: list[Any] = []

        def _capture(args: Any) -> None:
            captured_args.append(args)

        with patch.object(cli, "_cmd_download", side_effect=_capture):
            cli.main(["download", "mlx-community/Llama-3.2-3B-Instruct-4bit", "--yes"])
        assert captured_args[0].yes is True

    def test_main_revision_defaults_to_main(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hw = _make_fake_hw()
        cli = _reload_cli(fake_hf, fake_hw)
        captured_args: list[Any] = []

        def _capture(args: Any) -> None:
            captured_args.append(args)

        with patch.object(cli, "_cmd_download", side_effect=_capture):
            cli.main(["download", "mlx-community/Llama-3.2-3B-Instruct-4bit"])
        assert captured_args[0].revision == "main"
