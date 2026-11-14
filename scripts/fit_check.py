"""Does this Mac run MODEL fast enough for Gate 0? Loads it, writes one 60-token reply, reads every layer.

Run:  .venv/bin/python scripts/fit_check.py MODEL_ID [MIN_TOKENS_PER_S]
Prints load time, tokens per second and peak memory. Exits 1 when generation is slower than
MIN_TOKENS_PER_S (default 3), which on a 24 GB Mac means the model is being paged to disk.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.backends import make_backend  # noqa: E402

model = sys.argv[1]
floor = float(sys.argv[2]) if len(sys.argv) > 2 else float(os.environ.get("FIT_FLOOR", "3"))
t0 = time.time()
b = make_backend("mlx", model, {"qwen3_thinking": False})
load_s = time.time() - t0
msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Describe a walk through a city park in about sixty words."}]
b.generate(msgs, 8, 0.0, 0)  # warm-up, not timed
t0 = time.time()
gen = b.generate(msgs, 60, 0.0, 0)
rate = len(gen.reply_ids) / (time.time() - t0)
t0 = time.time()
r = b.reply_residuals(gen, list(range(b.n_layers)), first_k=64)
read_s = time.time() - t0
print(f"fit check {model}: layers={b.n_layers} d_model={b.d_model} load={load_s:.0f}s "
      f"speed={rate:.1f} tokens/s read-all-layers={read_s:.1f}s peak memory={b.peak_memory_gb():.1f} GB "
      f"readings shape={r['all'].shape}", flush=True)
try:  # how much memory macOS lets the GPU keep resident (raised with: sudo sysctl iogpu.wired_limit_mb=N)
    info = (b.mx.device_info if hasattr(b.mx, "device_info") else b.mx.metal.device_info)()
    print(f"fit check: GPU working-set limit {info['max_recommended_working_set_size'] / 1e9:.1f} GB, "
          f"machine memory {info['memory_size'] / 1e9:.1f} GB", flush=True)
except Exception as e:  # informational only
    print(f"fit check: could not read the GPU memory limit ({e})", flush=True)
if rate < floor:
    print(f"fit check FAILED: {rate:.1f} tokens/s is below {floor}; too slow for Gate 0 on this machine")
    sys.exit(1)
