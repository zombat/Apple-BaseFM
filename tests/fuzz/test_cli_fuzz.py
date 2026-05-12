"""Property-based fuzz tests for apple_basefm._cli and _hardware helpers.

Uses Hypothesis to verify that:
  - list_mlx_models(filter=arbitrary_str) never raises unhandled exceptions
  - list_mlx_models(filter=non_str) always raises TypeError (never crashes otherwise)
  - _parse_chip_gen(arbitrary_str) never raises
  - _infer_quant_bits(arbitrary_str) always returns a positive int
  - _infer_params_b(arbitrary_str) always returns float or None
  - _infer_disk_gb / _infer_min_ram_gb never raise on valid float input

Run with:
    pytest tests/fuzz/ -m fuzz -v

All tests mock huggingface_hub — no real HF cache or network required.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis import HealthCheck

from apple_basefm._hardware import _parse_chip_gen
from apple_basefm._catalog import (
    _infer_disk_gb,
    _infer_min_ram_gb,
    _infer_params_b,
    _infer_quant_bits,
)

pytestmark = pytest.mark.fuzz


# ---------------------------------------------------------------------------
# Fake huggingface_hub for list_mlx_models
# ---------------------------------------------------------------------------


class _FakeCacheNotFound(Exception):
    pass


def _make_fake_hf_hub_empty() -> types.ModuleType:
    fake = types.ModuleType("huggingface_hub")
    cache = types.SimpleNamespace(repos=[])
    fake.scan_cache_dir = lambda: cache  # type: ignore[attr-defined]
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.CacheNotFound = _FakeCacheNotFound  # type: ignore[attr-defined]
    fake.errors = fake_errors  # type: ignore[attr-defined]
    return fake


@pytest.fixture(autouse=True)
def _patch_hf_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_hf_hub_empty()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake.errors)
    sys.modules.pop("apple_basefm._cli", None)


def _get_cli() -> types.ModuleType:
    sys.modules.pop("apple_basefm._cli", None)
    return importlib.import_module("apple_basefm._cli")


# ---------------------------------------------------------------------------
# Fuzz: list_mlx_models with arbitrary string filters
# ---------------------------------------------------------------------------


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(filter_text=st.text())
def test_list_mlx_models_str_filter_never_raises(filter_text: str) -> None:
    """list_mlx_models(filter=<any string>) must return a list, never crash."""
    cli = _get_cli()
    result = cli.list_mlx_models(filter=filter_text)
    assert isinstance(result, list)


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(bad_filter=st.one_of(st.integers(), st.floats(), st.lists(st.text()), st.booleans()))
def test_list_mlx_models_non_str_filter_raises_type_error(bad_filter: object) -> None:
    """list_mlx_models(filter=<non-str>) must always raise TypeError."""
    cli = _get_cli()
    with pytest.raises(TypeError):
        cli.list_mlx_models(filter=bad_filter)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fuzz: _parse_chip_gen with arbitrary text
# ---------------------------------------------------------------------------


@settings(max_examples=500)
@given(chip_name=st.one_of(st.none(), st.text()))
def test_parse_chip_gen_never_raises(chip_name: str | None) -> None:
    """_parse_chip_gen must never raise on any str or None input."""
    result = _parse_chip_gen(chip_name)
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# Fuzz: _infer_quant_bits with arbitrary repo_id strings
# ---------------------------------------------------------------------------


@settings(max_examples=300)
@given(repo_id=st.text())
def test_infer_quant_bits_always_positive_int(repo_id: str) -> None:
    """_infer_quant_bits must return a positive int for any string."""
    result = _infer_quant_bits(repo_id)
    assert isinstance(result, int)
    assert result > 0


# ---------------------------------------------------------------------------
# Fuzz: _infer_params_b with arbitrary repo_id strings
# ---------------------------------------------------------------------------


@settings(max_examples=300)
@given(repo_id=st.text())
def test_infer_params_b_returns_float_or_none(repo_id: str) -> None:
    """_infer_params_b must return a positive float or None for any string."""
    result = _infer_params_b(repo_id)
    assert result is None or (isinstance(result, float) and result > 0)


# ---------------------------------------------------------------------------
# Fuzz: _infer_disk_gb / _infer_min_ram_gb with valid param ranges
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    params_b=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    quant_bits=st.integers(min_value=1, max_value=32),
)
def test_infer_disk_gb_never_raises(params_b: float, quant_bits: int) -> None:
    """_infer_disk_gb must never raise for valid param ranges."""
    result = _infer_disk_gb(params_b, quant_bits)
    assert isinstance(result, float)
    assert result >= 0


@settings(max_examples=200)
@given(
    params_b=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    quant_bits=st.integers(min_value=1, max_value=32),
)
def test_infer_min_ram_gb_never_raises(params_b: float, quant_bits: int) -> None:
    """_infer_min_ram_gb must never raise for valid param ranges."""
    result = _infer_min_ram_gb(params_b, quant_bits)
    assert isinstance(result, int)
    assert result >= 2
