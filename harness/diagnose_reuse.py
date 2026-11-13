"""Why does the fast questionnaire method differ slightly from the slow one?

    .venv/bin/python -u -m harness.diagnose_reuse runs/pilot-benzion-v2 --model gemma -n 2

For each questionnaire item it computes the option probabilities four ways:
  A  slow: one full pass, no cache (what the pilot used)
  A2 one full pass, but through a fresh cache
  B  two passes through a fresh cache per item: shared part, then the item (nothing is shared or copied)
  C  fast: the shared part computed once and copied for every item (what v2.0 would use)

Reading the result:
  B and C agree, A differs  -> the difference comes from computing in two steps (rounding in the
                               model's 16-bit arithmetic), not from sharing. The fast method is sound.
  B and C differ            -> sharing/copying changes the answer: a bug in the fast method.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .backends import common_prefix_len, make_backend
from .measures import _stai_forks
from .run import system_text
from .sources import BenZion


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default="gemma", help="substring of the model id")
    ap.add_argument("-n", type=int, default=2, help="cells")
    a = ap.parse_args(argv)

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cfg = json.loads((a.run_dir / "protocol.json").read_text())["config"]
    rows = [json.loads(l) for l in (a.run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if a.model in r["model"]][: a.n]
    stai = BenZion().stai()
    sys_fork = system_text(cfg.get("fork_system_prompt", "benzion"))
    backend = make_backend("mlx", rows[0]["model"], cfg["decoding"])
    model = backend.lens.model

    def probs_from(logits, sel):
        p = mx.softmax(logits[0, -1].astype(mx.float32), axis=-1)[sel]
        mx.eval(p)
        p = np.array(p, dtype=np.float64)
        return p / p.sum()

    worst = defaultdict(float)
    for r in rows:
        base_seed = int(r["cell_id"][:8], 16)
        for name, cp in r["checkpoints"].items():
            k = cp["after_step"]
            body = r["transcript"][1: 1 + 2 * (k + 1)]
            fork = ([{"role": "system", "content": sys_fork}] if sys_fork else []) + body
            _, forks, options = _stai_forks(fork, stai, base_seed + 1000 + k)
            prefill, ids = backend._option_tokens(tuple(options))
            sel = mx.array(ids)
            prompts = [backend._ids(m, add_generation_prompt=True, suffix=prefill) for m in forks]
            n = common_prefix_len(prompts, min_suffix=2)

            shared = make_prompt_cache(model)
            model(mx.array(prompts[0][:n])[None], cache=shared)
            mx.eval([c.keys for c in shared] + [c.values for c in shared])

            A, A2, B, C = [], [], [], []
            for p in prompts:
                A.append(probs_from(model(mx.array(p)[None]), sel))
                A2.append(probs_from(model(mx.array(p)[None], cache=make_prompt_cache(model)), sel))
                fresh = make_prompt_cache(model)
                model(mx.array(p[:n])[None], cache=fresh)
                mx.eval([c.keys for c in fresh] + [c.values for c in fresh])
                B.append(probs_from(model(mx.array(p[n:])[None], cache=fresh), sel))
                C.append(probs_from(model(mx.array(p[n:])[None], cache=[copy.copy(c) for c in shared]), sel))
            A, A2, B, C = map(np.array, (A, A2, B, C))
            d = {"A-A2": np.abs(A - A2).max(), "A-B": np.abs(A - B).max(), "B-C": np.abs(B - C).max(),
                 "A-C": np.abs(A - C).max()}
            for key, v in d.items():
                worst[key] = max(worst[key], float(v))
            print(f"  {r['arm']:<16} {name:<16} tokens {len(prompts[0])} (shared {n}) | largest probability gap: "
                  + "  ".join(f"{key} {v:.4f}" for key, v in d.items()), flush=True)

    print("WORST: " + "  ".join(f"{key} {v:.4f}" for key, v in worst.items()))
    if worst["B-C"] <= 0.01 and worst["A-B"] > 0.01:
        print("VERDICT: two-step rounding. Sharing is exact (B = C); the gap to the slow method comes from "
              "computing the conversation in two steps.")
    elif worst["B-C"] > 0.01:
        print("VERDICT: sharing changes the answer (B != C). Treat the fast method as buggy; use --safe.")
    else:
        print("VERDICT: all methods agree on these items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
