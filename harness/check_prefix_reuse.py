"""Check that prefix reuse gives the same questionnaire answers as one full pass per item.

    .venv/bin/python -m harness.check_prefix_reuse runs/pilot-benzion-v2 [--model gemma] [-n 2]

For a few finished cells it rebuilds each checkpoint's questionnaire fork exactly as the runner did
(same system prompt, same conversation, same option shuffles), scores it both ways, and compares
against what the run stored. Prints the largest differences, the speed-up, and PASS/FAIL.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .backends import make_backend
from .measures import fork_battery
from .sources import norm
from .run import system_text
from .sources import BenZion

# Agreement needed to switch the confirmatory run to prefix reuse.
MAX_TOTAL_DIFF = 0.5    # STAI total ranges 20-80
MAX_ITEM_DIFF = 0.10    # per-item expected score ranges 1-4
MIN_ARGMAX_AGREE = 19   # of 20 items


def compare(a: dict, b: dict) -> dict:
    pairs = list(zip(a["items"], b["items"]))
    return {
        "total": abs(a["total_expected"] - b["total_expected"]),
        "item": max(abs(x["expected"] - y["expected"]) for x, y in pairs),
        "prob": max(abs(p - q) for x, y in pairs for p, q in zip(x["p_label"], y["p_label"])),
        "argmax_agree": sum(x["argmax"] == y["argmax"] for x, y in pairs),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default=None, help="substring of the model id (default: every model in the run)")
    ap.add_argument("-n", type=int, default=2, help="cells per model")
    a = ap.parse_args(argv)

    cfg = json.loads((a.run_dir / "protocol.json").read_text())["config"]
    rows = [json.loads(l) for l in (a.run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    stai = BenZion().stai()
    sys_fork = system_text(cfg.get("fork_system_prompt", "benzion"))
    vas_q = norm(cfg["texts"]["vas_question"]) if cfg.get("measures", {}).get("vas") else None
    ok_all = True

    for model, model_rows in by_model.items():
        if a.model and a.model not in model:
            continue
        backend = make_backend("mlx", model, cfg["decoding"])
        t_full = t_reuse = 0.0
        worst = {"total": 0.0, "item": 0.0, "prob": 0.0, "argmax_agree": 20, "vs_stored": 0.0}
        n_cp = vas_same = 0
        for r in model_rows[: a.n]:
            base_seed = int(r["cell_id"][:8], 16)
            for name, cp in r["checkpoints"].items():
                k = cp["after_step"]
                body = r["transcript"][1: 1 + 2 * (k + 1)]  # drop the main system prompt, keep turns 0..k
                fork = ([{"role": "system", "content": sys_fork}] if sys_fork else []) + body
                seed = base_seed + 1000 + k

                backend.reuse_prefix = False
                t0 = time.time(); bf = fork_battery(backend, fork, stai, vas_q, seed); t_full += time.time() - t0
                backend.reuse_prefix = True
                t0 = time.time(); br = fork_battery(backend, fork, stai, vas_q, seed); t_reuse += time.time() - t0
                full, reuse = bf["stai"], br["stai"]

                d = compare(full, reuse)
                vs_stored = abs(full["total_expected"] - cp["stai"]["total_expected"])
                n_cp += 1
                same_vas = (bf.get("vas") or {}).get("value") == (br.get("vas") or {}).get("value")
                vas_same += same_vas
                print(f"  {r['arm']:<16} {name:<16} total full={full['total_expected']:.2f} reuse={reuse['total_expected']:.2f} "
                      f"stored={cp['stai']['total_expected']:.2f} | max item diff {d['item']:.4f}, "
                      f"argmax agree {d['argmax_agree']}/20 | 0-100 full={bf.get('vas', {}).get('value')} "
                      f"reuse={br.get('vas', {}).get('value')} stored={cp.get('vas', {}).get('value')}")
                for key in ("total", "item", "prob"):
                    worst[key] = max(worst[key], d[key])
                worst["argmax_agree"] = min(worst["argmax_agree"], d["argmax_agree"])
                worst["vs_stored"] = max(worst["vs_stored"], vs_stored)

        ok = (worst["total"] <= MAX_TOTAL_DIFF and worst["item"] <= MAX_ITEM_DIFF
              and worst["argmax_agree"] >= MIN_ARGMAX_AGREE and (vas_q is None or vas_same >= n_cp - 1))
        ok_all &= ok
        print(f"{model}: {n_cp} checkpoints | worst total diff {worst['total']:.3f}, item {worst['item']:.4f}, "
              f"prob {worst['prob']:.4f}, argmax agree {worst['argmax_agree']}/20 | full pass vs stored run "
              f"{worst['vs_stored']:.3f} | 0-100 identical {vas_same}/{n_cp} | self-report time {t_full:.0f}s -> {t_reuse:.0f}s "
              f"({t_full / max(t_reuse, 1e-9):.1f}x faster) | {'PASS' if ok else 'FAIL'}")
        backend.close()
        del backend
    print("OVERALL:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
