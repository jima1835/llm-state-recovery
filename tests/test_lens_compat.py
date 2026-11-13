"""harness.lens_compat: capture and steering act on the hidden state, for array and tuple layers.

Uses stand-in wrapper classes with the same shape as mlx-lens 0.1.2's, and numpy arrays, so it runs
without a GPU.
"""
import types

import numpy as np
import pytest

from harness import lens_compat


def fake_module():
    class _Base:
        __slots__ = ("_original", "_vector", "_scale", "output")

        def __init__(self, original, vector=None, scale=0.0):
            object.__setattr__(self, "_original", original)
            object.__setattr__(self, "_vector", vector)
            object.__setattr__(self, "_scale", scale)
            object.__setattr__(self, "output", None)

    class _CaptureLayer(_Base):
        def __call__(self, x):  # mlx-lens 0.1.2 behaviour: assumes an array
            out = self._original(x)
            object.__setattr__(self, "output", out)
            return out

    class _SteeredLayer(_Base):
        def __call__(self, x):
            return self._original(x) + self._scale * self._vector

    class _SteeredCaptureLayer(_Base):
        def __call__(self, x):
            out = self._original(x) + self._scale * self._vector
            object.__setattr__(self, "output", out)
            return out

    return types.SimpleNamespace(_CaptureLayer=_CaptureLayer, _SteeredLayer=_SteeredLayer,
                                 _SteeredCaptureLayer=_SteeredCaptureLayer)


X = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
V = np.ones(3, dtype=np.float32)
array_layer = lambda x: x * 2            # Gemma 3 / Qwen3 / Llama style  # noqa: E731
tuple_layer = lambda x: (x * 2, "kv", 7)  # Gemma 4 style: (h, shared_kv, offset)  # noqa: E731


def test_array_layers_unchanged():
    ref, patched = fake_module(), fake_module()
    lens_compat.install(patched, version="0.1.2")
    for name, args in (("_CaptureLayer", ()), ("_SteeredLayer", (V, 0.5)), ("_SteeredCaptureLayer", (V, 0.5))):
        a, b = getattr(ref, name)(array_layer, *args), getattr(patched, name)(array_layer, *args)
        np.testing.assert_array_equal(a(X), b(X))
        if hasattr(a, "output") and name != "_SteeredLayer":
            np.testing.assert_array_equal(a.output, b.output)


def test_tuple_layers():
    m = fake_module()
    lens_compat.install(m, version="0.1.2")
    cap = m._CaptureLayer(tuple_layer)
    out = cap(X)
    assert out[1:] == ("kv", 7)
    np.testing.assert_array_equal(cap.output, X * 2)          # capture keeps the hidden state only

    st = m._SteeredLayer(tuple_layer, V, 0.5)
    h, kv, off = st(X)
    np.testing.assert_array_equal(h, X * 2 + 0.5 * V)          # steering adds to the hidden state
    assert (kv, off) == ("kv", 7)                              # the rest passes through

    sc = m._SteeredCaptureLayer(tuple_layer, V, 0.5)
    h, kv, off = sc(X)
    np.testing.assert_array_equal(sc.output, X * 2 + 0.5 * V)
    np.testing.assert_array_equal(h, sc.output)


def test_unsupported_version_refuses():
    with pytest.raises(RuntimeError):
        lens_compat.install(fake_module(), version="0.2.0")
