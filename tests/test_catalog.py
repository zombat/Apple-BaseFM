"""Unit tests for apple_basefm._catalog.

huggingface_hub is mocked for all online tests — no network calls are made.
Tests cover offline filtering, online fetching, helper functions, and edge cases.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apple_basefm._catalog import (
    ModelEntry,
    SuggestResult,
    _fetch_online_suggestions,
    _filter_offline,
    _infer_disk_gb,
    _infer_min_ram_gb,
    _infer_params_b,
    _infer_quant_bits,
    suggest_models,
)
from apple_basefm._hardware import HardwareInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hw(
    ram_gb: int = 16,
    free_disk_gb: int = 100,
    chip_gen: int = 2,
    is_apple_silicon: bool = True,
    chip: str | None = "Apple M2 Pro",
) -> HardwareInfo:
    return HardwareInfo(
        chip=chip,
        chip_gen=chip_gen,
        ram_gb=ram_gb,
        free_disk_gb=free_disk_gb,
        is_apple_silicon=is_apple_silicon,
    )


def _fake_model(repo_id: str, downloads: int = 100, likes: int = 10) -> MagicMock:
    m = MagicMock()
    m.id = repo_id
    m.downloads = downloads
    m.likes = likes
    return m


# ---------------------------------------------------------------------------
# Name-based inference helpers
# ---------------------------------------------------------------------------


class TestInferParamsB:
    def test_3b(self) -> None:
        # In "Llama-3.2-3B", the regex picks up "3B" → 3.0
        assert _infer_params_b("mlx-community/Llama-3.2-3B-Instruct-4bit") == 3.0

    def test_70b(self) -> None:
        assert _infer_params_b("mlx-community/Llama-3.3-70B-Instruct-4bit") == 70.0

    def test_no_b_count_returns_none(self) -> None:
        assert _infer_params_b("mlx-community/some-model-without-params") is None

    def test_7b(self) -> None:
        assert _infer_params_b("mlx-community/Mistral-7B-Instruct") == 7.0


class TestInferQuantBits:
    def test_4bit(self) -> None:
        assert _infer_quant_bits("mlx-community/Llama-3.2-3B-4bit") == 4

    def test_8bit(self) -> None:
        assert _infer_quant_bits("mlx-community/model-8bit") == 8

    def test_fp16(self) -> None:
        assert _infer_quant_bits("mlx-community/model-fp16") == 16

    def test_default_4_when_unknown(self) -> None:
        assert _infer_quant_bits("mlx-community/model-instruct") == 4

    def test_q3(self) -> None:
        assert _infer_quant_bits("mlx-community/model-q3") == 3


class TestInferDiskGb:
    def test_3b_4bit_roughly_correct(self) -> None:
        # 3e9 * 0.5 / 1e9 * 1.1 ≈ 1.65 → rounds to 1.6 or nearby
        result = _infer_disk_gb(3.0, 4)
        assert 1.0 < result < 3.0

    def test_7b_4bit_roughly_correct(self) -> None:
        result = _infer_disk_gb(7.0, 4)
        assert 3.0 < result < 5.0


class TestInferMinRamGb:
    def test_small_model_at_least_2gb(self) -> None:
        assert _infer_min_ram_gb(1.0, 4) >= 2

    def test_7b_4bit_at_least_5gb(self) -> None:
        assert _infer_min_ram_gb(7.0, 4) >= 5


# ---------------------------------------------------------------------------
# _filter_offline
# ---------------------------------------------------------------------------


class TestFilterOffline:
    def test_8gb_ram_excludes_large_models(self) -> None:
        hw = _hw(ram_gb=8, free_disk_gb=100)
        results = _filter_offline(hw)
        assert all(m["min_ram_gb"] <= 8 for m in results)
        assert len(results) > 0

    def test_64gb_ram_includes_all_tiers(self) -> None:
        hw = _hw(ram_gb=64, free_disk_gb=200)
        results = _filter_offline(hw)
        # Should include 70B models
        repo_ids = [m["repo_id"] for m in results]
        assert any("70B" in r or "72B" in r for r in repo_ids)

    def test_0gb_ram_returns_empty(self) -> None:
        hw = _hw(ram_gb=0, free_disk_gb=100)
        assert _filter_offline(hw) == []

    def test_disk_headroom_excludes_large_models(self) -> None:
        # Only 3 GB free — only tiny models should pass
        hw = _hw(ram_gb=64, free_disk_gb=3)
        results = _filter_offline(hw)
        # Every result must fit: disk_gb + 1.0 <= 3
        assert all(m["disk_gb"] + 1.0 <= 3 for m in results)

    def test_disk_free_zero_skips_disk_filter(self) -> None:
        """When free_disk_gb=0 (unknown), disk filtering is disabled."""
        hw_no_disk = _hw(ram_gb=16, free_disk_gb=0)
        hw_with_disk = _hw(ram_gb=16, free_disk_gb=100)
        no_disk_results = _filter_offline(hw_no_disk)
        with_disk_results = _filter_offline(hw_with_disk)
        # Without disk filter, same or more models pass
        assert len(no_disk_results) >= len(with_disk_results)

    def test_chip_gen_filter_respects_min_chip_gen(self) -> None:
        """Entries requiring chip_gen > hw.chip_gen are excluded."""
        # Create a test by temporarily adding a high-gen-only entry to catalog
        import apple_basefm._catalog as cat_mod

        original = cat_mod._OFFLINE_CATALOG[:]
        cat_mod._OFFLINE_CATALOG.append(
            cat_mod._entry(
                "mlx-community/fake-m5-only-4bit", 4, 1.0, 4, 5, "M5 only"
            )
        )
        try:
            hw = _hw(ram_gb=64, free_disk_gb=200, chip_gen=2)
            results = _filter_offline(hw)
            assert not any(m["repo_id"] == "mlx-community/fake-m5-only-4bit" for m in results)
        finally:
            cat_mod._OFFLINE_CATALOG[:] = original

    def test_chip_gen_zero_skips_chip_filter(self) -> None:
        """When chip_gen=0 (unknown), chip generation filter is disabled."""
        hw = _hw(ram_gb=64, free_disk_gb=200, chip_gen=0)
        results = _filter_offline(hw)
        # All entries with min_chip_gen=1 should still pass
        assert len(results) > 0


# ---------------------------------------------------------------------------
# _fetch_online_suggestions
# ---------------------------------------------------------------------------


def _make_fake_hf_hub(models: list[MagicMock]) -> types.ModuleType:
    """Return a fake huggingface_hub module with list_models returning models."""
    fake = types.ModuleType("huggingface_hub")
    fake.list_models = lambda **kwargs: iter(models)  # type: ignore[attr-defined]
    return fake


class TestFetchOnlineSuggestions:
    def test_returns_entries_for_valid_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models = [
            _fake_model("mlx-community/Llama-3.2-3B-Instruct-4bit"),
            _fake_model("mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
        ]
        fake_hf = _make_fake_hf_hub(models)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        hw = _hw(ram_gb=16, free_disk_gb=100)
        import importlib
        import apple_basefm._catalog as cat_mod
        importlib.reload(cat_mod)
        result = cat_mod._fetch_online_suggestions(hw)
        assert result is not None
        assert len(result) >= 1

    def test_skips_model_with_non_string_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        none_id_model = MagicMock()
        none_id_model.id = None
        int_id_model = MagicMock()
        int_id_model.id = 12345
        valid = _fake_model("mlx-community/Llama-3.2-3B-Instruct-4bit")
        fake_hf = _make_fake_hf_hub([none_id_model, int_id_model, valid])
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        import importlib
        import apple_basefm._catalog as cat_mod
        importlib.reload(cat_mod)
        hw = _hw(ram_gb=16, free_disk_gb=100)
        result = cat_mod._fetch_online_suggestions(hw)
        # Only valid model should appear
        assert result is None or all(isinstance(m["repo_id"], str) for m in result)

    def test_skips_model_without_params_in_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        no_params = _fake_model("mlx-community/some-model-without-size")
        fake_hf = _make_fake_hf_hub([no_params])
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        import importlib
        import apple_basefm._catalog as cat_mod
        importlib.reload(cat_mod)
        hw = _hw(ram_gb=16, free_disk_gb=100)
        result = cat_mod._fetch_online_suggestions(hw)
        # Should be None (empty treated as None)
        assert result is None

    def test_returns_none_on_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.list_models = MagicMock(side_effect=ConnectionError("no network"))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        import importlib
        import apple_basefm._catalog as cat_mod
        importlib.reload(cat_mod)
        hw = _hw(ram_gb=16, free_disk_gb=100)
        result = cat_mod._fetch_online_suggestions(hw)
        assert result is None

    def test_returns_none_when_hf_hub_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # type: ignore[arg-type]
        import importlib
        import apple_basefm._catalog as cat_mod
        importlib.reload(cat_mod)
        hw = _hw(ram_gb=16, free_disk_gb=100)
        result = cat_mod._fetch_online_suggestions(hw)
        assert result is None


# ---------------------------------------------------------------------------
# suggest_models
# ---------------------------------------------------------------------------


class TestSuggestModels:
    def test_non_apple_silicon_returns_empty(self) -> None:
        hw = _hw(is_apple_silicon=False)
        result = suggest_models(hw=hw, online=False)
        assert result["models"] == []
        assert result["source"] == "offline"
        assert result["hardware"] is hw

    def test_offline_mode_returns_filtered_catalog(self) -> None:
        hw = _hw(ram_gb=8, free_disk_gb=100)
        result = suggest_models(hw=hw, online=False)
        assert result["source"] == "offline"
        assert all(m["min_ram_gb"] <= 8 for m in result["models"])

    def test_offline_sorted_by_min_ram_desc(self) -> None:
        hw = _hw(ram_gb=64, free_disk_gb=200)
        result = suggest_models(hw=hw, online=False)
        ram_values = [m["min_ram_gb"] for m in result["models"]]
        assert ram_values == sorted(ram_values, reverse=True)

    def test_online_uses_hf_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models = [_fake_model("mlx-community/Llama-3.2-3B-Instruct-4bit")]
        fake_hf = _make_fake_hf_hub(models)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        import importlib
        import apple_basefm._catalog as cat_mod
        importlib.reload(cat_mod)
        hw = _hw(ram_gb=16, free_disk_gb=100)
        result = cat_mod.suggest_models(hw=hw, online=True)
        assert result["source"] == "online"

    def test_online_falls_back_to_offline_on_network_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.list_models = MagicMock(side_effect=ConnectionError("no net"))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        import importlib
        import apple_basefm._catalog as cat_mod
        importlib.reload(cat_mod)
        hw = _hw(ram_gb=16, free_disk_gb=100)
        result = cat_mod.suggest_models(hw=hw, online=True)
        assert result["source"] == "offline"
        assert len(result["models"]) > 0

    def test_hardware_is_auto_detected_when_none(self) -> None:
        """When hw=None, detect_hardware() is called internally."""
        fake_hw = _hw(ram_gb=8, free_disk_gb=50)
        # detect_hardware is imported lazily inside suggest_models — patch at source
        with patch("apple_basefm._hardware.detect_hardware", return_value=fake_hw):
            result = suggest_models(hw=None, online=False)
        assert result["hardware"] is fake_hw

    def test_returns_valid_suggest_result_keys(self) -> None:
        hw = _hw(ram_gb=8, free_disk_gb=100)
        result = suggest_models(hw=hw, online=False)
        assert "models" in result
        assert "source" in result
        assert "hardware" in result

    def test_model_entry_keys_present(self) -> None:
        hw = _hw(ram_gb=16, free_disk_gb=100)
        result = suggest_models(hw=hw, online=False)
        for model in result["models"]:
            for key in ("repo_id", "quant_bits", "disk_gb", "min_ram_gb",
                        "min_chip_gen", "note", "downloads", "likes"):
                assert key in model, f"Missing key {key!r} in ModelEntry"
