"""Smoke test: can this Mac load a small model, generate, and capture one layer's activations?

Run:  .venv/bin/python smoke_test.py [model_id]   (default: Gemma 3 4B)
(Use the venv's python explicitly: if your shell aliases `python`, `source .venv/bin/activate` is not enough.)
Expected: a short reply, a residual array of shape (1, tokens, hidden_size),
and peak memory well under 8 GB for the 4B model.
"""
import sys
import time

import mlx.core as mx
from mlx_lens import LensModel  # v0.1.2; wraps mlx_lm.load and exposes capture/steer

from harness.lens_compat import install

install()  # lets capture/steer handle Gemma 4's tuple-returning layers; no change for other models

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/gemma-3-4b-it-4bit"

t0 = time.time()
lens = LensModel(MODEL)  # load once; lens.model / lens.tokenizer are the mlx_lm objects
tokenizer = lens.tokenizer
print(f"Loaded {MODEL} in {time.time() - t0:.1f}s  (layers={lens.n_layers}, d_model={lens.d_model})")

messages = [{"role": "user", "content": "In one sentence, what is a residual stream in a transformer?"}]
# enable_thinking=False puts Qwen3 in non-thinking mode (per configs/models.yaml); other templates ignore it.
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
reply = lens.generate(prompt, max_tokens=60, verbose=False)
print("Reply:", reply.strip())

# Residual capture. Use forward() (one pass over the whole text), not generate():
# capture keeps only the layer's LAST call, which during generation is a single decode step.
try:
    layer = lens.n_layers // 2
    tokens = tokenizer.encode("I failed again and again today.")
    with lens.capture(layers=[layer]) as cache:
        lens.forward(tokens)
    arr = cache[layer]
    mx.eval(arr)
    print(f"Captured layer {layer} of {lens.n_layers}: shape {arr.shape} for {len(tokens)} tokens, dtype {arr.dtype}")
except Exception as e:  # report, don't hide
    print("mlx-lens capture did not run as written:", repr(e))
    print("Next: .venv/bin/python -c 'from mlx_lens import LensModel; help(LensModel)' and adjust the capture call.")

peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
print(f"Peak memory: {peak / 1e9:.2f} GB")
