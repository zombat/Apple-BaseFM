"""Property-based fuzz tests for apple_basefm._kv.

Uses Hypothesis to verify that the TurboQuant V2 KV cache components:
  - Never crash on arbitrary integer inputs to constructor parameters
  - Always raise a specific ValueError/TypeError (never an unhandled exception)
    for out-of-range inputs
  - Never crash in update_and_fetch() on valid tensor shapes
  - Preserve the orthogonality invariant for any head_dim > 0

Run with:
    pytest tests/fuzz/ -m fuzz -v
    pytest tests/fuzz/ -m fuzz --hypothesis-seed=0  # deterministic

All tests run on Linux (no Apple Silicon or MLX required).
mlx.core is mocked with numpy-backed stubs identical to test_kv_cache.py.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis import HealthCheck

pytestmark = pytest.mark.fuzz

# ---------------------------------------------------------------------------
# Fake mlx.core — same stub as test_kv_cache.py
# ---------------------------------------------------------------------------


def _make_fake_mlx_core() -> types.ModuleType:
    mx = types.ModuleType("mlx.core")
    mx_random = types.ModuleType("mlx.core.random")
    mx_random.seed = lambda seed: np.random.seed(seed)  # type: ignore[attr-defined]
    mx_random.normal = lambda shape=None, **kwargs: np.random.randn(*shape)  # type: ignore[attr-defined]
    mx.random = mx_random  # type: ignore[attr-defined]
    mx_linalg = types.ModuleType("mlx.core.linalg")

    def _qr(a, **kwargs):
        Q, _ = np.linalg.qr(a)
        return Q, None

    mx_linalg.qr = _qr  # type: ignore[attr-defined]
    mx.linalg = mx_linalg  # type: ignore[attr-defined]
    return mx


@pytest.fixture(autouse=True)
def fake_mlx_core(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mx = _make_fake_mlx_core()
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.core.random", mx.random)
    monkeypatch.setitem(sys.modules, "mlx.core.linalg", mx.linalg)
    return mx


# ---------------------------------------------------------------------------
# Fuzz: TurboQuantV2Cache constructor — never raises anything except ValueError
# ---------------------------------------------------------------------------


class TestTurboQuantV2CacheConstructorFuzz:
    @given(
        bits=st.integers(min_value=-100, max_value=100),
        group_size=st.integers(min_value=-100, max_value=200),
        step=st.integers(min_value=-100, max_value=1000),
    )
    @settings(max_examples=300, deadline=None)
    def test_constructor_never_raises_unexpected_exception(
        self, bits: int, group_size: int, step: int
    ) -> None:
        """Any combination of ints must either succeed or raise ValueError.

        An unhandled exception (TypeError, AttributeError, etc.) is a bug.
        """
        from apple_basefm._kv import TurboQuantV2Cache

        try:
            TurboQuantV2Cache(bits=bits, group_size=group_size, step=step)
        except ValueError:
            pass  # expected for out-of-range inputs

    @given(
        bits=st.sampled_from([2, 4, 8]),
        group_size=st.integers(min_value=1, max_value=256),
        step=st.integers(min_value=1, max_value=1024),
        use_rotation=st.booleans(),
    )
    @settings(max_examples=200)
    def test_valid_params_always_construct(
        self, bits: int, group_size: int, step: int, use_rotation: bool
    ) -> None:
        """Valid parameter combinations must always succeed."""
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(
            bits=bits, group_size=group_size, step=step, use_rotation=use_rotation
        )
        assert c.bits == bits
        assert c.group_size == group_size
        assert c.step == step

    @given(
        bits=st.integers().filter(lambda x: x not in {2, 4, 8}),
    )
    @settings(max_examples=100)
    def test_invalid_bits_always_raises_value_error(self, bits: int) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="bits must be one of"):
            TurboQuantV2Cache(bits=bits)

    @given(group_size=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_nonpositive_group_size_always_raises_value_error(
        self, group_size: int
    ) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="group_size must be >= 1"):
            TurboQuantV2Cache(group_size=group_size)

    @given(step=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_nonpositive_step_always_raises_value_error(self, step: int) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        with pytest.raises(ValueError, match="step must be >= 1"):
            TurboQuantV2Cache(step=step)


# ---------------------------------------------------------------------------
# Fuzz: make_rotation_matrix — orthogonality invariant for any head_dim
# ---------------------------------------------------------------------------


class TestMakeRotationMatrixFuzz:
    @given(head_dim=st.integers(min_value=1, max_value=128))
    @settings(max_examples=50)
    def test_result_is_always_square(self, head_dim: int) -> None:
        from apple_basefm._kv.rotation import make_rotation_matrix

        Q = make_rotation_matrix(head_dim)
        assert Q.shape == (head_dim, head_dim)

    @given(head_dim=st.integers(min_value=1, max_value=64))
    @settings(max_examples=50)
    def test_result_is_always_orthogonal(self, head_dim: int) -> None:
        """Q @ Q.T must be close to identity for every head_dim."""
        from apple_basefm._kv.rotation import make_rotation_matrix

        Q = make_rotation_matrix(head_dim)
        product = Q @ Q.T
        np.testing.assert_array_almost_equal(product, np.eye(head_dim), decimal=8)


# ---------------------------------------------------------------------------
# Fuzz: _TurboQuantV2LayerCache.update_and_fetch — never crashes on valid shapes
# ---------------------------------------------------------------------------


class TestLayerCacheUpdateAndFetchFuzz:
    @given(
        batch=st.integers(min_value=1, max_value=4),
        heads=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=16),
        head_dim=st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=200)
    def test_update_and_fetch_never_crashes_without_rotation(
        self, batch: int, heads: int, seq_len: int, head_dim: int
    ) -> None:
        """Valid tensor shapes must never cause an unhandled exception."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=1, step=1, rotation=None)
        keys = np.random.randn(batch, heads, seq_len, head_dim)
        values = np.random.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = lc.update_and_fetch(keys, values)
        # Must return arrays (any shape — inner cache decides accumulation)
        assert k_out is not None
        assert v_out is not None

    @given(
        head_dim=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=100)
    def test_update_and_fetch_never_crashes_with_identity_rotation(
        self, head_dim: int, seq_len: int
    ) -> None:
        """Identity rotation must be a transparent no-op for any head_dim."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        rotation = np.eye(head_dim)
        lc = _TurboQuantV2LayerCache(bits=4, group_size=1, step=1, rotation=rotation)
        keys = np.random.randn(1, 1, seq_len, head_dim)
        values = np.random.randn(1, 1, seq_len, head_dim)
        k_out, v_out = lc.update_and_fetch(keys, values)
        np.testing.assert_array_almost_equal(k_out, keys)
        np.testing.assert_array_almost_equal(v_out, values)

    @given(
        n_calls=st.integers(min_value=2, max_value=10),
        head_dim=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=50)
    def test_offset_monotonically_increases_with_each_call(
        self, n_calls: int, head_dim: int
    ) -> None:
        """offset must increase by 1 for each update_and_fetch call."""
        from apple_basefm._kv.cache_v2 import _TurboQuantV2LayerCache

        lc = _TurboQuantV2LayerCache(bits=4, group_size=1, step=1, rotation=None)
        for i in range(n_calls):
            assert lc.offset == i
            lc.update_and_fetch(
                np.ones((1, 1, 1, head_dim)), np.ones((1, 1, 1, head_dim))
            )
        assert lc.offset == n_calls


# ---------------------------------------------------------------------------
# Fuzz: describe() — never crashes, always returns a non-empty string
# ---------------------------------------------------------------------------


class TestDescribeFuzz:
    @given(
        bits=st.sampled_from([2, 4, 8]),
        group_size=st.integers(min_value=1, max_value=256),
        use_rotation=st.booleans(),
    )
    @settings(max_examples=150)
    def test_describe_always_returns_nonempty_string(
        self, bits: int, group_size: int, use_rotation: bool
    ) -> None:
        from apple_basefm._kv import TurboQuantV2Cache

        c = TurboQuantV2Cache(bits=bits, group_size=group_size, use_rotation=use_rotation)
        result = c.describe()
        assert isinstance(result, str)
        assert len(result) > 0
