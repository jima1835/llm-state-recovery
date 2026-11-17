"""POST HOC (Study 1 is already unblinded): the second text-only judge on Study 1's replies.

    ~/.venvs/openai/bin/python -m analysis.study1_judge_api runs/wave1 runs/wave2 runs/wave2-8bit
    ~/.venvs/openai/bin/python -m analysis.study1_judge_api runs/wave1 --limit 5      # smoke test

The same judge, prompt and score as analysis/study2_judge_api.py (gpt-6-luna, expected option 1-4 from
token probabilities); that file is imported unchanged. Study 1's protocols carry no judge prompt, so the
prompt is read from Study 2's protocol (--prompt-run, default runs/study2). The judge sees only the
model's reply to each of the four checkpoint turns. Writes <run>/judge_api.jsonl per run folder; finished
conversations are skipped on a re-run. Needs only the openai package and OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from analysis import study2_judge_api as A

ROOT = Path(__file__).resolve().parents[1]


def judge_run(run: Path, prompt: str, client, limit: int | None, workers: int) -> int:
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    out = run / "judge_api.jsonl"
    done = {json.loads(l)["cell_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [r for r in rows if r["cell_id"] not in done][:limit]
    print(f"{run}: {len(rows)} conversations, {len(done)} already judged, {len(todo)} to judge now", flush=True)
    low = 0

    def work(r):
        return r, {name: A.judge_one(client, prompt, text) for name, text in A.replies(r).items()}

    with open(out, "a") as fh, ThreadPoolExecutor(workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(work, r) for r in todo]), 1):
            r, sc = fut.result()
            low += sum(v["mass"] < 0.9 for v in sc.values())
            fh.write(json.dumps({"cell_id": r["cell_id"], "model": r["model"], "judge": A.JUDGE,
                                 "scores": {k: v["score"] for k, v in sc.items()},
                                 "mass": {k: v["mass"] for k, v in sc.items()},
                                 "pos": {k: v["pos"] for k, v in sc.items()},
                                 "text": {k: v["text"] for k, v in sc.items()}}) + "\n")
            fh.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ratings with mass < 0.9 so far: {low}", flush=True)
    return len(todo)


def main(argv=None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--prompt-run", type=Path, default=ROOT / "runs" / "study2")
    ap.add_argument("--limit", type=int, default=None, help="judge at most this many new conversations per run")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if client is None:
        from openai import OpenAI
        client = OpenAI(max_retries=6)
    prompt = A.norm(json.loads((args.prompt_run / "protocol.json").read_text())["config"]["texts"]["judge_prompt"])
    for run in args.runs:
        judge_run(run, prompt, client, args.limit, args.workers)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
