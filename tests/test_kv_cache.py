"""Unit tests for apple_basefm._kv — TurboQuant KV cache backend.

All tests run on Linux (no MLX hardware required). mlx.core and
mlx_lm.models.cache are mocked at the module level so the lazy-import
pattern in _kv/ is exercised correctly.

conftest.py provides:
  - fake_mlx_modules  (autouse) — mlx_lm + mlx_lm.models.cache in sys.modules
  - fake_mlx_core     (session scoped) — mlx.core and mlx.linalg in sys.modules
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake mlx.core — module-level fixture used by _kv tests
# ---------------------------------------------------------------------------


def _make_fake_mlx_core() -> types.ModuleType:
    """Build a minimal mlx.core stub that satisfies the _kv/ imports.

    Uses numpy for array arithmetic so `@` (matmul) works correctly in
    update_and_fetch() without Apple Silicon.
    """
    mx = types.ModuleType("mlx.core")

    # --- mlx.core.random ---
    mx_random = types.ModuleType("mlx.core.random")
    mx_random.seed = lambda seed: np.random.seed(seed)  # seed numpy rng for determinism
    mx_random.normal = lambda shape=None, **kwargs: np.random.randn(*shape)
    mx.random = mx_random  # type: ignore[attr-defined]

    # --- mlx.core.linalg ---
    mx_linalg = types.ModuleType("mlx.core.linalg")

    def _qr(a, **kwargs):
        Q, R = np.linalg.qr(a)
        return Q, R

    mx_linalg.qr = _qr  # type: ignore[attr-defined]
    mx.linalg = mx_linalg  # type: ignore[attr-defined]

    return mx


@pytest.fixture(autouse=True)
def fake_mlx_core(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a synthetic mlx.core into sys.modules for every test in this file."""
    mx = _make_fake_mlx_core()
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.core.random", mx.random)
    monkeypatch.setitem(sys.modules, "mlx.core.linalg", mx.linalg)
    return mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_kv() -> types.ModuleType:
    """Reload _kv subpackage after sys.modules is patched."""
    for key in list(sys.modules):
        if "apple_basefm._kv" in key:
            del sys.modules[key]
    import apple_basefm._kv as kv  # noqa: PLC0415

    return kv


# ---------------------------------------------------------------------------
# KVCacheStrategy protocol
# ---------------------------------------------------------------------------


class TestKVCacheStrategyProtocol:
    def test_protocol_satisfied_by_conforming_object(self) -> None:
        from apple_basefm._kv import KVCacheStrategy

        class GoodCache:
            def build(self, n_layers: int, head_dim: int) -> list:
                return []

            def describe(self) -> str:
                return "GoodCache()"

        assert isinstance(GoodCache(), KVCacheStrategy)

    def test_protocol_not_satisfied_by_missing_describe(self) -> None:
        from apple_basefm._kv import KVCacheStrategy

        class BadCache:
            def build(self, n_layers: int, head_dim: int) -> list:
                return []

        assert not isinstance(BadCache(), KVCacheStrategy)

    def test_protocol_not_satisfied_by_plain_object(self) -> None:
        from apple_basefm._kv import KVCacheStrategy

        assert not isinstance(object(), KVCacheStrategy)


# ---------------------------------------------------------------------------
# TurboQuantV2Cache — construction validation
# ---------------------------------------------------------------------------


class TestTurboQuantV2CacheValidation:
    def test_default_construction_succeeds(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache()
        assert c.bits == 4
        assert c.group_size == 64
        assert c.use_rotation is True
        assert c.step == 256

    def test_invalid_bits_raises(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="bits must be one of"):
            TurboQuantV2Cache(bits=3)

    def test_invalid_bits_zero_raises(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="bits must be one of"):
            TurboQuantV2Cache(bits=0)

    def test_invalid_bits_seven_raises(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="bits must be one of"):
            TurboQuantV2Cache(bits=7)

    def test_valid_bits_values_accepted(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        for bits in (2, 4, 8):
            assert TurboQuantV2Cache(bits=bits).bits == bits

    def test_group_size_zero_raises(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="group_size must be >= 1"):
            TurboQuantV2Cache(group_size=0)

    def test_group_size_negative_raises(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="group_size must be >= 1"):
            TurboQuantV2Cache(group_size=-1)

    def test_step_zero_raises(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="step must be >= 1"):
            TurboQuantV2Cache(step=0)

    def test_step_negative_raises(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="step must be >= 1"):
            TurboQuantV2Cache(step=-10)


# ---------------------------------------------------------------------------
# TurboQuantV2Cache.describe()
# ---------------------------------------------------------------------------


class TestTurboQuantV2CacheDescribe:
    def test_describe_rotated(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=4, use_rotation=True, group_size=64)
        assert "rotated" in c.describe()
        assert "4" in c.describe()

    def test_describe_lean(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=4, use_rotation=False)
        assert "LEAN" in c.describe()


# ---------------------------------------------------------------------------
# TurboQuantV2Cache.build()
# ---------------------------------------------------------------------------


class TestTurboQuantV2CacheBuild:
    def test_build_returns_correct_length(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=False)
        layers = c.build(n_layers=4, head_dim=64)
        assert len(layers) == 4

    def test_build_zero_layers_returns_empty_list(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=False)
        assert c.build(n_layers=0, head_dim=64) == []

    def test_build_with_rotation_caches_matrix(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=True)
        c.build(n_layers=2, head_dim=64)
        assert c._rotation is not None

    def test_build_rotation_reused_on_second_call(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=True)
        c.build(n_layers=2, head_dim=64)
        rotation_first = c._rotation
        c.build(n_layers=2, head_dim=64)
        # Same object — not recomputed
        assert c._rotation is rotation_first

    def test_build_no_rotation_leaves_rotation_none(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=False)
        c.build(n_layers=2, head_dim=64)
        assert c._rotation is None

    def test_build_each_call_returns_fresh_list(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=False)
        first = c.build(n_layers=2, head_dim=64)
        second = c.build(n_layers=2, head_dim=64)
        assert first is not second
        assert first[0] is not second[0]  # independent layer cache objects


# ---------------------------------------------------------------------------
# _TurboQuantV2LayerCache.update_and_fetch()
# ---------------------------------------------------------------------------


class TestLayerCacheUpdateAndFetch:
    def test_update_and_fetch_without_rotation_delegates_to_inner(self) -> None:
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=256, rotation=None)
        keys = np.ones((1, 2, 4, 8))
        values = np.ones((1, 2, 4, 8))
        k_out, v_out = lc.update_and_fetch(keys, values)
        # Inner FakeQuantizedKVCache returns the same arrays
        assert k_out is keys
        assert v_out is values

    def test_update_and_fetch_with_rotation_rotates_then_delegates(self) -> None:
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        head_dim = 8
        rotation = np.eye(head_dim)  # identity — rotation is a no-op for this test
        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=256, rotation=rotation)
        keys = np.random.randn(1, 2, 4, head_dim)
        values = np.random.randn(1, 2, 4, head_dim)
        k_out, v_out = lc.update_and_fetch(keys, values)
        # With identity rotation: rotated == original; inner returns as-is
        np.testing.assert_array_almost_equal(k_out, keys)
        np.testing.assert_array_almost_equal(v_out, values)

    def test_update_and_fetch_increments_offset(self) -> None:
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=256, rotation=None)
        assert lc.offset == 0
        lc.update_and_fetch(np.ones((1, 1, 1, 4)), np.ones((1, 1, 1, 4)))
        assert lc.offset == 1

    def test_state_empty_before_first_call(self) -> None:
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=256, rotation=None)
        assert lc.state == []

    def test_state_nonempty_after_call(self) -> None:
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=256, rotation=None)
        lc.update_and_fetch(np.ones((1, 1, 1, 4)), np.ones((1, 1, 1, 4)))
        assert len(lc.state) > 0

    def test_nbytes_zero_before_first_call(self) -> None:
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=256, rotation=None)
        assert lc.nbytes == 0

    def test_step_forwarded_to_inner_cache(self) -> None:
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=512, rotation=None)
        lc._ensure_inner()
        assert lc._inner.step == 512


# ---------------------------------------------------------------------------
# make_rotation_matrix
# ---------------------------------------------------------------------------


class TestMakeRotationMatrix:
    def test_returns_square_matrix(self) -> None:
        from apple_basefm._kv.rotation import make_rotation_matrix

        Q = make_rotation_matrix(64)
        assert Q.shape == (64, 64)

    def test_returns_orthogonal_matrix(self) -> None:
        from apple_basefm._kv.rotation import make_rotation_matrix

        Q = make_rotation_matrix(16)
        # Q @ Q.T should be close to identity
        product = Q @ Q.T
        np.testing.assert_array_almost_equal(product, np.eye(16), decimal=10)

    def test_deterministic_with_same_head_dim(self) -> None:
        from apple_basefm._kv.rotation import make_rotation_matrix

        Q1 = make_rotation_matrix(32)
        Q2 = make_rotation_matrix(32)
        np.testing.assert_array_equal(Q1, Q2)

    def test_mlx_not_available_raises_runtime_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "mlx.core", None)  # type: ignore[call-overload]
        # Force reload so the lazy import fires
        for key in list(sys.modules):
            if "apple_basefm._kv.rotation" in key:
                del sys.modules[key]
        from apple_basefm._kv import rotation  # noqa: PLC0415

        with pytest.raises((RuntimeError, ImportError)):
            rotation.make_rotation_matrix(64)

    def test_missing_qr_raises_runtime_error_with_hint(self, monkeypatch) -> None:
        """Simulate MLX < 0.16 where mx.linalg.qr does not exist."""
        mx_linalg_no_qr = types.ModuleType("mlx.core.linalg")
        # No qr attribute — accessing it raises AttributeError
        monkeypatch.setattr(sys.modules["mlx.core"], "linalg", mx_linalg_no_qr)
        for key in list(sys.modules):
            if "apple_basefm._kv.rotation" in key:
                del sys.modules[key]
        from apple_basefm._kv import rotation as rot_mod  # noqa: PLC0415

        with pytest.raises(RuntimeError, match="qr"):
            rot_mod.make_rotation_matrix(64)


# ---------------------------------------------------------------------------
# AppleLocalLM — kv_cache param integration
# ---------------------------------------------------------------------------


def _make_local_instance(apple_local_mod: Any, **extra_kwargs: Any) -> Any:
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        return apple_local_mod.AppleLocalLM("mlx-community/test-model-4bit", **extra_kwargs)


@pytest.fixture()
def apple_local_mod(fake_mlx_modules: Any) -> Any:  # noqa: ARG001
    import importlib

    if "apple_basefm.apple_local" in sys.modules:
        del sys.modules["apple_basefm.apple_local"]
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
    ):
        return importlib.import_module("apple_basefm.apple_local")


class TestAppleLocalLMKvCache:
    def test_default_no_kv_cache(self, apple_local_mod: Any) -> None:
        lm = _make_local_instance(apple_local_mod)
        assert lm._kv_strategy is None
        assert lm._head_dim is None
        assert lm._n_layers is None

    def test_string_preset_turboquant_v2(self, apple_local_mod: Any) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        lm = _make_local_instance(apple_local_mod, kv_cache="turboquant-v2")
        assert isinstance(lm._kv_strategy, TurboQuantV2Cache)
        assert lm._kv_strategy.use_rotation is True
        assert lm._head_dim == 64
        assert lm._n_layers == 1

    def test_string_preset_turboquant_v2_lean(self, apple_local_mod: Any) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        lm = _make_local_instance(apple_local_mod, kv_cache="turboquant-v2-lean")
        assert isinstance(lm._kv_strategy, TurboQuantV2Cache)
        assert lm._kv_strategy.use_rotation is False

    def test_unknown_string_preset_raises_value_error(self, apple_local_mod: Any) -> None:
        with pytest.raises(ValueError, match="unknown kv_cache preset"):
            _make_local_instance(apple_local_mod, kv_cache="not-a-preset")

    def test_invalid_type_raises_type_error(self, apple_local_mod: Any) -> None:
        with pytest.raises(TypeError, match="kv_cache must be"):
            _make_local_instance(apple_local_mod, kv_cache=42)  # type: ignore[arg-type]

    def test_custom_strategy_object_accepted(self, apple_local_mod: Any) -> None:
        from apple_basefm._kv import KVCacheStrategy

        class MyCache:
            def build(self, n_layers: int, head_dim: int) -> list:
                return []

            def describe(self) -> str:
                return "MyCache()"

        assert isinstance(MyCache(), KVCacheStrategy)
        lm = _make_local_instance(apple_local_mod, kv_cache=MyCache())
        assert lm._kv_strategy is not None

    def test_each_preset_call_returns_new_instance(self, apple_local_mod: Any) -> None:
        lm1 = _make_local_instance(apple_local_mod, kv_cache="turboquant-v2")
        lm2 = _make_local_instance(apple_local_mod, kv_cache="turboquant-v2")
        assert lm1._kv_strategy is not lm2._kv_strategy

    def test_forward_passes_prompt_cache_to_generate(
        self, apple_local_mod: Any, fake_mlx_modules: Any
    ) -> None:
        """Verify that forward() calls mlx_lm.generate with prompt_cache kwarg."""
        captured: dict[str, Any] = {}

        def capturing_generate(model: Any, tokenizer: Any, prompt: str = "", **kwargs: Any) -> str:
            captured.update(kwargs)
            return "response"

        fake_mlx_modules.generate = capturing_generate

        lm = _make_local_instance(apple_local_mod, kv_cache="turboquant-v2-lean")
        lm.forward(messages=[{"role": "user", "content": "hello"}], cache=False)

        assert "prompt_cache" in captured
        assert isinstance(captured["prompt_cache"], list)

    def test_forward_no_prompt_cache_when_kv_none(
        self, apple_local_mod: Any, fake_mlx_modules: Any
    ) -> None:
        """Verify that forward() does NOT pass prompt_cache when kv_cache=None."""
        captured: dict[str, Any] = {}

        def capturing_generate(model: Any, tokenizer: Any, prompt: str = "", **kwargs: Any) -> str:
            captured.update(kwargs)
            return "response"

        fake_mlx_modules.generate = capturing_generate

        lm = _make_local_instance(apple_local_mod)
        lm.forward(messages=[{"role": "user", "content": "hello"}], cache=False)

        assert "prompt_cache" not in captured
