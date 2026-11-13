"""Make mlx-lens 0.1.2 work with decoder layers that return a tuple.

mlx-lens wraps a decoder layer and assumes the layer returns the hidden-state array. Gemma 4's
layers (mlx-lm `gemma4_text.DecoderLayer`) return `(h, shared_kv, offset)`. Without this patch,
capture stores the whole tuple, and steering tries to add a vector to a tuple.

`install()` replaces the `__call__` of the three mlx-lens wrappers so they act on the hidden state
only (the first element of a tuple) and pass the other elements through unchanged. For layers that
return an array (Gemma 3, Qwen3, Llama 3.1), the result is the same as mlx-lens's own code.
"""
from __future__ import annotations

SUPPORTED = ("0.1.",)  # mlx-lens versions whose wrapper classes this patch was written against


def hidden(out):
    """The hidden-state array in a layer's output."""
    return out[0] if isinstance(out, tuple) else out


def replace_hidden(out, h):
    """The layer's output with its hidden state replaced by `h`."""
    return (h, *out[1:]) if isinstance(out, tuple) else h


def _capture_call(self, x, *args, **kwargs):
    out = self._original(x, *args, **kwargs)
    object.__setattr__(self, "output", hidden(out))
    return out


def _steer_call(self, x, *args, **kwargs):
    out = self._original(x, *args, **kwargs)
    return replace_hidden(out, hidden(out) + self._scale * self._vector)


def _steer_capture_call(self, x, *args, **kwargs):
    out = self._original(x, *args, **kwargs)
    h = hidden(out) + self._scale * self._vector
    object.__setattr__(self, "output", h)
    return replace_hidden(out, h)


def install(module=None, version: str | None = None) -> None:
    """Patch the mlx-lens wrapper classes. Safe to call more than once."""
    if module is None:
        import importlib.metadata

        from mlx_lens import model as module
        version = importlib.metadata.version("mlx-lens")
    if version is not None and not version.startswith(SUPPORTED):
        raise RuntimeError(f"harness.lens_compat was written for mlx-lens {SUPPORTED}, found {version}. "
                           "Check its wrapper classes before running.")
    module._CaptureLayer.__call__ = _capture_call
    module._SteeredLayer.__call__ = _steer_call
    module._SteeredCaptureLayer.__call__ = _steer_capture_call
