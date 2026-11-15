"""Text-only baseline (plan section 3, P2; test HE): a judge model rates the wording of every reply.

    .venv/bin/python -m analysis.study2_judge runs/study2 [--models ID ...]

For each conversation and checkpoint, the judge reads only the model's reply to that turn (no
conversation, no arm label) with the fixed `judge_prompt` from the run's own protocol, and the score
is the expected option, 1 (not anxious) to 4 (extremely). Writes runs/<run>/judge.jsonl, one line per
conversation; finished conversations are skipped on a re-run. It reads reply text and writes ratings;
it never reads questionnaire scores or readings, so it can run before the results are opened.

Judge: Llama 3.1 8B; the two Llama models are judged by Qwen3 8B, so that no model rates its own text
(the same rule as Gate 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from harness.backends import make_backend
from harness.sources import norm

JUDGE = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
JUDGE_FOR_LLAMA = "mlx-community/Qwen3-8B-4bit"
OPTIONS = ["1", "2", "3", "4"]


def judge_for(model: str) -> str:
    return JUDGE_FOR_LLAMA if "llama" in model.lower() else JUDGE


def score(backend, prompt: str, text: str) -> tuple[float, float]:
    p, mass = backend.option_probs([{"role": "user", "content": prompt.replace("{t}", text)}], OPTIONS)
    p = np.asarray(p, float)
    return float(p @ np.arange(1, 5) / p.sum()), float(mass)


def replies(row: dict) -> dict:
    """checkpoint name -> the assistant reply to that turn."""
    said = [m["content"] for m in row["transcript"] if m["role"] == "assistant"]
    return {name: said[cp["after_step"]] for name, cp in row["checkpoints"].items()}


def main(argv=None, backend_kind: str = "mlx") -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--backend", default=backend_kind)
    args = ap.parse_args(argv)
    prompt = norm(json.loads((args.run / "protocol.json").read_text())["config"]["texts"]["judge_prompt"])
    rows = [json.loads(l) for l in (args.run / "results.jsonl").read_text().splitlines() if l.strip()]
    if args.models:
        rows = [r for r in rows if r["model"] in args.models]
    out = args.run / "judge.jsonl"
    done = {json.loads(l)["cell_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [r for r in rows if r["cell_id"] not in done]
    print(f"{len(rows)} conversations, {len(done & {r['cell_id'] for r in rows})} already judged, {len(todo)} to judge")
    by_judge: dict = {}
    for r in todo:
        by_judge.setdefault(judge_for(r["model"]), []).append(r)
    for jm, group in by_judge.items():
        jb = make_backend(args.backend, jm, {"qwen3_thinking": False})
        print(f"judge {jm}: {len(group)} conversations")
        with open(out, "a") as fh:
            for i, r in enumerate(group, 1):
                sc = {name: score(jb, prompt, text) for name, text in replies(r).items()}
                fh.write(json.dumps({"cell_id": r["cell_id"], "model": r["model"], "judge": jm,
                                     "scores": {k: v[0] for k, v in sc.items()},
                                     "mass": {k: v[1] for k, v in sc.items()}}) + "\n")
                fh.flush()
                if i % 100 == 0:
                    print(f"  {i}/{len(group)}", flush=True)
        jb.close()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
