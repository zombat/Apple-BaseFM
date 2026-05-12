"""Unit tests for apple_basefm._kv — TurboQuant KV cache backend.

All tests run on Linux (no MLX hardware required). mlx.core and
mlx_lm.models.cache are mocked at the module level so the lazy-import
pattern in _kv/ is exercised correctly.

conftest.py provides:
  - fake_mlx_modules  (autouse) — mlx_lm + mlx_lm.models.cache in sys.modules
  - fake_mlx_core     (autouse) — mlx.core and mlx.linalg in sys.modules

Test markers:
  - pytest.mark.unit  — all tests in this file
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit


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
        assert c.use_rotation is False  # safe default until attention_v2.py ships
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

    def test_describe_includes_bits_for_2bit(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=2, use_rotation=False)
        assert "2" in c.describe()

    def test_describe_includes_bits_for_8bit(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=8, use_rotation=True)
        assert "8" in c.describe()

    def test_describe_includes_group_size(self) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=4, group_size=32, use_rotation=False)
        assert "32" in c.describe()


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

    def test_build_propagates_bits_group_size_step(self) -> None:
        """Layer caches must receive the bits, group_size, and step from the parent."""
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=2, group_size=32, step=128, use_rotation=False)
        layers = c.build(n_layers=1, head_dim=64)
        layer = layers[0]
        assert layer._bits == 2
        assert layer._group_size == 32
        assert layer._step == 128

    def test_build_group_size_larger_than_head_dim_does_not_raise(self) -> None:
        """group_size > head_dim is allowed — mlx handles it at quantize time."""
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=4, group_size=128, use_rotation=False)
        layers = c.build(n_layers=2, head_dim=64)  # group_size > head_dim
        assert len(layers) == 2

    def test_build_head_dim_zero_raises_value_error(self) -> None:
        """head_dim=0 must raise ValueError before any MLX call is made."""
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=False)
        with pytest.raises(ValueError, match="head_dim must be >= 1"):
            c.build(n_layers=2, head_dim=0)

    def test_build_head_dim_negative_raises_value_error(self) -> None:
        """head_dim=-1 must raise ValueError."""
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(use_rotation=False)
        with pytest.raises(ValueError, match="head_dim must be >= 1"):
            c.build(n_layers=2, head_dim=-1)

    def test_build_group_size_non_divisible_emits_warning(self, caplog) -> None:
        """group_size that doesn't divide head_dim should emit a WARNING."""
        import logging

        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=4, group_size=48, use_rotation=False)
        with caplog.at_level(logging.WARNING, logger="apple_basefm._kv.cache_v2"):
            c.build(n_layers=1, head_dim=64)  # 64 % 48 != 0
        assert any("not evenly divisible" in r.message for r in caplog.records)

    def test_use_rotation_true_emits_warning(self, caplog) -> None:
        """Constructing TurboQuantV2Cache with use_rotation=True should emit a WARNING."""
        import logging

        from apple_basefm._kv import TurboQuantV2Cache

        with caplog.at_level(logging.WARNING, logger="apple_basefm._kv.cache_v2"):
            TurboQuantV2Cache(use_rotation=True)
        assert any("rotated" in r.message.lower() or "attention_v2" in r.message for r in caplog.records)


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

    def test_ensure_inner_is_idempotent(self) -> None:
        """Calling _ensure_inner() twice must not replace the existing inner cache."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=256, rotation=None)
        lc._ensure_inner()
        first_inner = lc._inner
        lc._ensure_inner()
        assert lc._inner is first_inner

    def test_bits_propagated_to_inner_cache(self) -> None:
        """bits=2 and bits=8 must be forwarded to QuantizedKVCache."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        for bits in (2, 8):
            lc = _TurboQuantV2LayerCache(bits=bits, group_size=64, step=256, rotation=None)
            lc._ensure_inner()
            assert lc._inner.bits == bits

    def test_non_identity_rotation_transforms_keys(self) -> None:
        """A 90° rotation permutation must change the key values passed to inner."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        head_dim = 4
        # Permutation matrix — rotates axes but preserves norms
        rotation = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                            dtype=float)
        lc = _TurboQuantV2LayerCache(bits=4, group_size=4, step=256, rotation=rotation)

        keys = np.array([[[[1, 2, 3, 4]]]], dtype=float)  # shape (1,1,1,4)
        values = np.zeros((1, 1, 1, head_dim))

        # Capture what inner receives by recording update_and_fetch calls
        received: list[Any] = []
        original_uaf = lc._ensure_inner

        def _patched_ensure():
            original_uaf()
            orig = lc._inner.update_and_fetch

            def _capture(k, v):
                received.append((k.copy(), v.copy()))
                return orig(k, v)

            lc._inner.update_and_fetch = _capture  # type: ignore[method-assign]

        lc._ensure_inner = _patched_ensure  # type: ignore[method-assign]
        lc.update_and_fetch(keys, values)

        assert len(received) == 1
        # After permutation rotation: [1,2,3,4] @ P = [4,1,2,3]
        # P column-j picks row-j of input, so output[j] = input[P_col_j]
        np.testing.assert_array_almost_equal(received[0][0], [[[[4, 1, 2, 3]]]])

    def test_shape_mismatch_raises_runtime_error_with_hint(self) -> None:
        """Rotation applied to keys with mismatched last-dim must raise RuntimeError
        with a diagnostic message naming the shapes involved."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        rotation = np.eye(8)  # expects head_dim=8
        lc = _TurboQuantV2LayerCache(bits=4, group_size=8, step=256, rotation=rotation)

        # keys last dim is 4 — does not match rotation shape 8x8
        keys = np.ones((1, 1, 1, 4))
        values = np.zeros((1, 1, 1, 4))
        with pytest.raises(RuntimeError, match="rotation matmul failed"):
            lc.update_and_fetch(keys, values)

    def test_inner_reset_on_update_and_fetch_failure(self) -> None:
        """If update_and_fetch on the inner cache raises, _inner should be reset to
        None so the next call gets a fresh, clean inner cache."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=4, step=256, rotation=None)
        # Force _inner to exist first
        lc._ensure_inner()
        assert lc._inner is not None

        # Patch inner's update_and_fetch to raise
        lc._inner.update_and_fetch = lambda k, v: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="boom"):
            lc.update_and_fetch(np.ones((1, 1, 1, 4)), np.zeros((1, 1, 1, 4)))

        # _inner must be cleared so next call rebuilds cleanly
        assert lc._inner is None

    def test_ensure_inner_step_attribute_error_does_not_raise(self) -> None:
        """If inner.step is read-only in a future mlx-lm, _ensure_inner must not raise."""
        from unittest.mock import MagicMock, patch

        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=64, step=512, rotation=None)

        class _ReadOnlyStepCache(MagicMock):
            @property
            def step(self):
                return 256

            @step.setter
            def step(self, _value):
                raise AttributeError("step is read-only")

        with patch("mlx_lm.models.cache.QuantizedKVCache", return_value=_ReadOnlyStepCache()):
            # Must not raise even though step= assignment raises AttributeError
            lc._ensure_inner()
        assert lc._inner is not None


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
        assert lm._kv_strategy.use_rotation is False  # preset uses LEAN until attention_v2.py ships
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
