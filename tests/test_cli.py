"""Unit tests for apple_basefm._cli.

huggingface_hub is mocked throughout — no real HF cache or network calls.
Tests cover list_mlx_models(), all three CLI subcommands, and main().
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build a fake huggingface_hub module
# ---------------------------------------------------------------------------


def _make_fake_repo(
    repo_id: str,
    size_on_disk: int = 1_800_000_000,
    size_on_disk_str: str = "1.8G",
    nb_files: int = 10,
    last_modified: float = 1700000000.0,
    repo_type: str = "model",
) -> MagicMock:
    repo = MagicMock()
    repo.repo_id = repo_id
    repo.repo_type = repo_type
    repo.size_on_disk = size_on_disk
    repo.size_on_disk_str = size_on_disk_str
    repo.nb_files = nb_files
    repo.last_modified = last_modified
    rev = MagicMock()
    rev.commit_hash = "abc123"
    repo.revisions = [rev]
    return repo


def _make_fake_cache(repos: list[MagicMock]) -> MagicMock:
    cache = MagicMock()
    cache.repos = repos

    strategy = MagicMock()
    strategy.expected_freed_size_str = "1.8G"
    strategy.execute = MagicMock()
    cache.delete_revisions = MagicMock(return_value=strategy)
    return cache


class _FakeCacheNotFound(Exception):
    pass


def _make_fake_hf_hub(repos: list[MagicMock] | None = None) -> types.ModuleType:
    fake = types.ModuleType("huggingface_hub")
    cache = _make_fake_cache(repos or [])
    fake.scan_cache_dir = MagicMock(return_value=cache)  # type: ignore[attr-defined]
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.CacheNotFound = _FakeCacheNotFound  # type: ignore[attr-defined]
    fake.errors = fake_errors  # type: ignore[attr-defined]
    return fake


def _reload_cli(fake_hf: types.ModuleType) -> types.ModuleType:
    """Reload _cli with a patched huggingface_hub in sys.modules."""
    sys.modules["huggingface_hub"] = fake_hf
    sys.modules["huggingface_hub.errors"] = fake_hf.errors
    sys.modules.pop("apple_basefm._cli", None)
    return importlib.import_module("apple_basefm._cli")


# ---------------------------------------------------------------------------
# list_mlx_models — programmatic API
# ---------------------------------------------------------------------------


class TestListMlxModels:
    def test_returns_empty_list_on_cache_not_found(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hf.scan_cache_dir = MagicMock(side_effect=_FakeCacheNotFound("no cache"))
        cli = _reload_cli(fake_hf)
        result = cli.list_mlx_models()
        assert result == []

    def test_filters_by_mlx_substring_by_default(self) -> None:
        repos = [
            _make_fake_repo("mlx-community/Llama-3.2-3B-4bit"),
            _make_fake_repo("hf-user/bert-base-uncased"),
        ]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        result = cli.list_mlx_models()
        assert len(result) == 1
        assert result[0]["repo_id"] == "mlx-community/Llama-3.2-3B-4bit"

    def test_all_models_bypasses_mlx_filter(self) -> None:
        repos = [
            _make_fake_repo("mlx-community/Llama-3.2-3B-4bit"),
            _make_fake_repo("hf-user/bert-base-uncased"),
        ]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        result = cli.list_mlx_models(all_models=True)
        assert len(result) == 2

    def test_filter_substring_case_insensitive(self) -> None:
        repos = [
            _make_fake_repo("mlx-community/Llama-3.2-3B-Instruct-4bit"),
            _make_fake_repo("mlx-community/Mistral-7B-4bit"),
        ]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        result = cli.list_mlx_models(filter="LLAMA")
        assert len(result) == 1
        assert "Llama" in result[0]["repo_id"]

    def test_filter_non_string_raises_type_error(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(TypeError, match="filter must be str or None"):
            cli.list_mlx_models(filter=123)  # type: ignore[arg-type]

    def test_results_sorted_by_repo_id(self) -> None:
        repos = [
            _make_fake_repo("mlx-community/zzz-model-4bit"),
            _make_fake_repo("mlx-community/aaa-model-4bit"),
        ]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        result = cli.list_mlx_models(all_models=True)
        assert result[0]["repo_id"] < result[1]["repo_id"]

    def test_permission_error_raises_runtime_error(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hf.scan_cache_dir = MagicMock(side_effect=PermissionError("denied"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(RuntimeError, match="Permission denied"):
            cli.list_mlx_models()

    def test_runtime_error_from_scan_is_wrapped(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hf.scan_cache_dir = MagicMock(side_effect=RuntimeError("internal"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(RuntimeError, match="Could not scan"):
            cli.list_mlx_models()

    def test_skips_non_model_repo_types(self) -> None:
        repos = [
            _make_fake_repo("mlx-community/some-dataset", repo_type="dataset"),
            _make_fake_repo("mlx-community/some-space", repo_type="space"),
            _make_fake_repo("mlx-community/actual-model-4bit", repo_type="model"),
        ]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        result = cli.list_mlx_models()
        assert len(result) == 1
        assert result[0]["repo_id"] == "mlx-community/actual-model-4bit"

    def test_returns_correct_field_values(self) -> None:
        repos = [_make_fake_repo(
            "mlx-community/test-4bit",
            size_on_disk=999,
            size_on_disk_str="999B",
            nb_files=5,
            last_modified=12345.0,
        )]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        result = cli.list_mlx_models()
        m = result[0]
        assert m["repo_id"] == "mlx-community/test-4bit"
        assert m["size_on_disk"] == 999
        assert m["size_on_disk_str"] == "999B"
        assert m["nb_files"] == 5
        assert m["last_modified"] == 12345.0

    def test_hf_hub_not_available_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate huggingface_hub absent at import time by reloading with None
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", None)  # type: ignore[arg-type]
        sys.modules.pop("apple_basefm._cli", None)
        cli = importlib.import_module("apple_basefm._cli")
        with pytest.raises(RuntimeError, match="huggingface_hub"):
            cli.list_mlx_models()


# ---------------------------------------------------------------------------
# _cmd_mlx_models — CLI subcommand
# ---------------------------------------------------------------------------


class TestCmdMlxModels:
    def _args(self, fmt: str = "text", filter: str | None = None, all_models: bool = False) -> MagicMock:
        args = MagicMock()
        args.format = fmt
        args.filter = filter
        args.all_models = all_models
        return args

    def test_json_output_is_valid_json(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/some-model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        cli._cmd_mlx_models(self._args(fmt="json"))
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert parsed[0]["repo_id"] == "mlx-community/some-model-4bit"

    def test_table_output_has_header(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/some-model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        cli._cmd_mlx_models(self._args(fmt="table"))
        out = capsys.readouterr().out
        assert "REPO ID" in out
        assert "SIZE" in out

    def test_empty_cache_prints_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub([])  # no repos
        cli = _reload_cli(fake_hf)
        cli._cmd_mlx_models(self._args())
        err = capsys.readouterr().err
        assert "No models found" in err

    def test_runtime_error_exits_1(self) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hf.scan_cache_dir = MagicMock(side_effect=RuntimeError("boom"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_mlx_models(self._args())
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _cmd_suggest — CLI subcommand
# ---------------------------------------------------------------------------


def _hw_apple(ram_gb: int = 16) -> MagicMock:
    hw = MagicMock()
    hw.__getitem__ = lambda self, key: {
        "is_apple_silicon": True,
        "ram_gb": ram_gb,
        "free_disk_gb": 100,
        "chip_gen": 2,
        "chip": "Apple M2 Pro",
    }[key]
    return hw


def _make_fake_suggest_result(source: str = "offline") -> dict:
    from apple_basefm._catalog import ModelEntry, SuggestResult
    from apple_basefm._hardware import HardwareInfo

    hw = HardwareInfo(
        chip="Apple M2 Pro",
        chip_gen=2,
        ram_gb=16,
        free_disk_gb=100,
        is_apple_silicon=True,
    )
    model = ModelEntry(
        repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        quant_bits=4,
        disk_gb=1.8,
        min_ram_gb=4,
        min_chip_gen=1,
        note="Solid 3B all-rounder",
        downloads=None,
        likes=None,
    )
    return SuggestResult(models=[model], source=source, hardware=hw)


class TestCmdSuggest:
    def _args(self, fmt: str = "table", offline: bool = True) -> MagicMock:
        args = MagicMock()
        args.format = fmt
        args.offline = offline
        return args

    def test_non_apple_silicon_exits_1(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        non_silicon_hw = {
            "is_apple_silicon": False,
            "ram_gb": 0,
            "free_disk_gb": 0,
            "chip_gen": 0,
            "chip": None,
        }
        with (
            patch("apple_basefm._hardware.detect_hardware", return_value=non_silicon_hw),
            pytest.raises(SystemExit) as exc,
        ):
            cli._cmd_suggest(self._args())
        assert exc.value.code == 1
        assert "Apple Silicon" in capsys.readouterr().err

    def test_ram_zero_exits_1(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        zero_ram_hw = {
            "is_apple_silicon": True,
            "ram_gb": 0,
            "free_disk_gb": 100,
            "chip_gen": 2,
            "chip": "Apple M2",
        }
        with (
            patch("apple_basefm._hardware.detect_hardware", return_value=zero_ram_hw),
            pytest.raises(SystemExit) as exc,
        ):
            cli._cmd_suggest(self._args())
        assert exc.value.code == 1

    def test_json_format_output(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        fake_result = _make_fake_suggest_result()
        apple_hw = {
            "is_apple_silicon": True,
            "ram_gb": 16,
            "free_disk_gb": 100,
            "chip_gen": 2,
            "chip": "Apple M2 Pro",
        }
        with (
            patch("apple_basefm._hardware.detect_hardware", return_value=apple_hw),
            patch("apple_basefm._catalog.suggest_models", return_value=fake_result),
        ):
            cli._cmd_suggest(self._args(fmt="json"))
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "suggestions" in parsed
        assert "hardware" in parsed

    def test_table_format_output(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        fake_result = _make_fake_suggest_result()
        apple_hw = {
            "is_apple_silicon": True,
            "ram_gb": 16,
            "free_disk_gb": 100,
            "chip_gen": 2,
            "chip": "Apple M2 Pro",
        }
        with (
            patch("apple_basefm._hardware.detect_hardware", return_value=apple_hw),
            patch("apple_basefm._catalog.suggest_models", return_value=fake_result),
        ):
            cli._cmd_suggest(self._args(fmt="table"))
        out = capsys.readouterr().out
        assert "REPO ID" in out or "Hardware" in out


# ---------------------------------------------------------------------------
# _cmd_remove — CLI subcommand
# ---------------------------------------------------------------------------


def _args_remove(repo_ids: list[str], yes: bool = False) -> MagicMock:
    args = MagicMock()
    args.repo_id = repo_ids
    args.yes = yes
    return args


class TestCmdRemove:
    def test_missing_repo_exits_1(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/existing-model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_remove(_args_remove(["mlx-community/does-not-exist"]))
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().err

    def test_yes_flag_skips_prompt(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/model-to-delete-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        cli._cmd_remove(_args_remove(["mlx-community/model-to-delete-4bit"], yes=True))
        # strategy.execute should have been called (strategy is in fake_hf cache)
        out = capsys.readouterr().out
        assert "Removed" in out

    def test_prompt_y_confirms(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        with patch("builtins.input", return_value="y"):
            cli._cmd_remove(_args_remove(["mlx-community/model-4bit"]))
        out = capsys.readouterr().out
        assert "Removed" in out

    def test_prompt_yes_confirms(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        with patch("builtins.input", return_value="yes"):
            cli._cmd_remove(_args_remove(["mlx-community/model-4bit"]))
        out = capsys.readouterr().out
        assert "Removed" in out

    def test_prompt_n_aborts(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        with patch("builtins.input", return_value="n"):
            cli._cmd_remove(_args_remove(["mlx-community/model-4bit"]))
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_prompt_empty_aborts(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        with patch("builtins.input", return_value=""):
            cli._cmd_remove(_args_remove(["mlx-community/model-4bit"]))
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_cache_not_found_exits_1(self, capsys: pytest.CaptureFixture) -> None:
        fake_hf = _make_fake_hf_hub()
        fake_hf.scan_cache_dir = MagicMock(side_effect=_FakeCacheNotFound("no cache"))
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_remove(_args_remove(["mlx-community/anything"]))
        assert exc.value.code == 1

    def test_keyboard_interrupt_during_prompt_exits_1(self) -> None:
        repos = [_make_fake_repo("mlx-community/model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        with (
            patch("builtins.input", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit) as exc,
        ):
            cli._cmd_remove(_args_remove(["mlx-community/model-4bit"]))
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# main() — argument dispatch
# ---------------------------------------------------------------------------


class TestMain:
    def test_mlx_models_subcommand_dispatched(self) -> None:
        fake_hf = _make_fake_hf_hub([])
        cli = _reload_cli(fake_hf)
        with patch.object(cli, "_cmd_mlx_models") as mock_cmd:
            cli.main(["mlx-models"])
        mock_cmd.assert_called_once()

    def test_suggest_subcommand_dispatched(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        apple_hw = {
            "is_apple_silicon": True,
            "ram_gb": 16,
            "free_disk_gb": 100,
            "chip_gen": 2,
            "chip": "Apple M2",
        }
        with (
            patch("apple_basefm._hardware.detect_hardware", return_value=apple_hw),
            patch("apple_basefm._catalog.suggest_models", return_value=_make_fake_suggest_result()),
        ):
            cli.main(["suggest", "--offline"])
        # No exception = correct dispatch

    def test_remove_subcommand_dispatched(self) -> None:
        repos = [_make_fake_repo("mlx-community/model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        with patch.object(cli, "_cmd_remove") as mock_cmd:
            cli.main(["remove", "mlx-community/model-4bit"])
        mock_cmd.assert_called_once()

    def test_no_subcommand_exits_nonzero(self) -> None:
        fake_hf = _make_fake_hf_hub()
        cli = _reload_cli(fake_hf)
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code != 0

    def test_mlx_models_json_format_arg(self, capsys: pytest.CaptureFixture) -> None:
        repos = [_make_fake_repo("mlx-community/model-4bit")]
        fake_hf = _make_fake_hf_hub(repos)
        cli = _reload_cli(fake_hf)
        cli.main(["mlx-models", "--format", "json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
