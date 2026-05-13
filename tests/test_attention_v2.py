"""Unit tests for apple_basefm._kv.attention_v2 — SDPA patch.

All tests run on Linux (no MLX hardware required). mlx.fast is mocked via a
per-test fixture that injects a fake module into sys.modules.

Test markers:
  - pytest.mark.unit  — all tests in this file
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip(
    "numpy",
    reason="attention_v2 tests require numpy. Install with: pip install numpy",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mlx_fast(monkeypatch):
    """Inject a fake mlx.fast into sys.modules and reset attention_v2 state after each test."""
    mlx_mod = types.ModuleType("mlx")
    fast_mod = types.ModuleType("mlx.fast")

    def _sentinel_sdpa(queries, keys, values, scale, mask=None, **kwargs):
        return (queries, keys, values)

    fast_mod.scaled_dot_product_attention = _sentinel_sdpa
    mlx_mod.fast = fast_mod

    monkeypatch.setitem(sys.modules, "mlx", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx.fast", fast_mod)

    yield fast_mod

    # Reset attention_v2 module-level state so tests don't bleed into each other.
    from apple_basefm._kv import attention_v2

    attention_v2._original_sdpa = None
    attention_v2._active_rotation = None


# ---------------------------------------------------------------------------
# install / uninstall / is_installed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInstall:
    def test_install_replaces_sdpa(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        original = mlx_fast.scaled_dot_product_attention
        attention_v2.install(np.eye(4, dtype=np.float32))

        assert mlx_fast.scaled_dot_product_attention is not original

    def test_install_sets_is_installed(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        assert not attention_v2.is_installed()
        attention_v2.install(np.eye(4))
        assert attention_v2.is_installed()

    def test_install_stores_original_sdpa(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        original = mlx_fast.scaled_dot_product_attention
        attention_v2.install(np.eye(4))

        assert attention_v2._original_sdpa is original

    def test_double_install_raises(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        attention_v2.install(np.eye(4))
        with pytest.raises(RuntimeError, match="already installed"):
            attention_v2.install(np.eye(4))

    def test_install_none_rotation_raises(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        with pytest.raises(ValueError, match="must not be None"):
            attention_v2.install(None)

    def test_uninstall_restores_original(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        original = mlx_fast.scaled_dot_product_attention
        attention_v2.install(np.eye(4))
        attention_v2.uninstall()

        assert mlx_fast.scaled_dot_product_attention is original

    def test_uninstall_clears_is_installed(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        attention_v2.install(np.eye(4))
        attention_v2.uninstall()

        assert not attention_v2.is_installed()

    def test_uninstall_clears_rotation(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        attention_v2.install(np.eye(4))
        attention_v2.uninstall()

        assert attention_v2._active_rotation is None

    def test_uninstall_noop_when_not_installed(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2

        assert not attention_v2.is_installed()
        attention_v2.uninstall()  # must not raise
        assert not attention_v2.is_installed()


# ---------------------------------------------------------------------------
# rotated_sdpa_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRotatedSdpaContext:
    def test_context_installs_on_enter(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2
        from apple_basefm._kv.attention_v2 import rotated_sdpa_context

        with rotated_sdpa_context(np.eye(4)):
            assert attention_v2.is_installed()

    def test_context_uninstalls_on_exit(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2
        from apple_basefm._kv.attention_v2 import rotated_sdpa_context

        with rotated_sdpa_context(np.eye(4)):
            pass

        assert not attention_v2.is_installed()

    def test_context_uninstalls_on_exception(self, mlx_fast) -> None:
        from apple_basefm._kv import attention_v2
        from apple_basefm._kv.attention_v2 import rotated_sdpa_context

        with pytest.raises(RuntimeError):
            with rotated_sdpa_context(np.eye(4)):
                raise RuntimeError("boom")

        assert not attention_v2.is_installed()

    def test_context_restores_original_sdpa_on_exception(self, mlx_fast) -> None:
        from apple_basefm._kv.attention_v2 import rotated_sdpa_context

        original = mlx_fast.scaled_dot_product_attention
        with pytest.raises(ValueError):
            with rotated_sdpa_context(np.eye(4)):
                raise ValueError("fail")

        assert mlx_fast.scaled_dot_product_attention is original


# ---------------------------------------------------------------------------
# Query rotation correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueryRotationCorrectness:
    def test_q_at_r_compensates_k_rotation(self) -> None:
        """(Q @ R) @ (K @ R).T == Q @ K.T for any orthogonal R.

        This validates the math that justifies the SDPA patch:
        K is stored as K@R in the cache; rotating Q by the same R
        before scoring produces the correct unrotated attention scores.
        """
        rng = np.random.default_rng(0)
        head_dim = 8

        raw = rng.standard_normal((head_dim, head_dim))
        R, _ = np.linalg.qr(raw)
        R = R.astype(np.float32)

        Q = rng.standard_normal((3, head_dim)).astype(np.float32)
        K = rng.standard_normal((5, head_dim)).astype(np.float32)

        expected = Q @ K.T
        actual = (Q @ R) @ (K @ R).T

        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_patched_sdpa_receives_rotated_queries(self, mlx_fast) -> None:
        """The installed patch passes Q @ R to the underlying SDPA, not Q."""
        from apple_basefm._kv.attention_v2 import rotated_sdpa_context

        received: list = []

        def capturing_sdpa(queries, keys, values, scale, mask=None, **kwargs):
            received.append(queries.copy())
            return (queries, keys, values)

        mlx_fast.scaled_dot_product_attention = capturing_sdpa

        rng = np.random.default_rng(2)
        head_dim = 4
        raw = rng.standard_normal((head_dim, head_dim))
        R, _ = np.linalg.qr(raw)
        R = R.astype(np.float32)

        Q = rng.standard_normal((2, head_dim)).astype(np.float32)
        K = rng.standard_normal((2, head_dim)).astype(np.float32)
        V = rng.standard_normal((2, head_dim)).astype(np.float32)

        with rotated_sdpa_context(R):
            mlx_fast.scaled_dot_product_attention(Q, K, V, scale=1.0)

        assert len(received) == 1
        np.testing.assert_allclose(received[0], Q @ R, atol=1e-6)

    def test_dtype_mismatch_casts_rotation(self, mlx_fast) -> None:
        """When Q dtype differs from rotation dtype, rotation is cast to match Q."""
        from apple_basefm._kv.attention_v2 import rotated_sdpa_context

        received: list = []

        def capturing_sdpa(queries, keys, values, scale, mask=None, **kwargs):
            received.append(queries)
            return (queries, keys, values)

        mlx_fast.scaled_dot_product_attention = capturing_sdpa

        R = np.eye(4, dtype=np.float64)  # rotation in float64
        Q = np.ones((2, 4), dtype=np.float32)  # queries in float32

        with rotated_sdpa_context(R):
            mlx_fast.scaled_dot_product_attention(Q, Q, Q, scale=1.0)

        assert received[0].dtype == np.float32
