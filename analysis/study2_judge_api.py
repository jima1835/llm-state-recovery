"""EXPLORATORY robustness check, not part of the registered plan: a second, stronger text-only judge.

    python analysis/study2_judge_api.py runs/study2-pilot --limit 5      # smoke test on already-seen pilot data
    python analysis/study2_judge_api.py runs/study2 --models ID ...      # the real run

Fixed before any Study 2 confirmatory result was opened (registration/study2_addendum_api_judge.md).
Same judge prompt as the registered judge (`judge_prompt` in the run's own protocol, whitespace-normalised
the same way), same replies (the model's reply to each checkpoint turn, no conversation, no arm label),
same score: the expected option over "1".."4" from the judge's token probabilities, renormalised over the
four options, with the probability mass on them kept as a check. Only the judge model differs:
gpt-6-luna through the OpenAI Responses API, reasoning effort "none", default temperature.

Reads reply text only; never reads questionnaire scores or readings, so it can run while results are blind.
Writes runs/<run>/judge_api.jsonl, one line per conversation; finished conversations are skipped on a
re-run. Self-contained (json + the openai package), so it can run from any Python environment:
    pip install openai ; export OPENAI_API_KEY=...   (never paste the key into a chat)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

JUDGE = "gpt-6-luna"
OPTIONS = ("1", "2", "3", "4")
SCAN = 3  # output positions searched for the first digit (the answer may follow a space token)


def norm(text: str) -> str:
    """Same as harness.sources.norm."""
    return " ".join(text.split())


def replies(row: dict) -> dict:
    """checkpoint name -> the assistant reply to that turn. Same as analysis.study2_judge.replies."""
    said = [m["content"] for m in row["transcript"] if m["role"] == "assistant"]
    return {name: said[cp["after_step"]] for name, cp in row["checkpoints"].items()}


def _get(x, key):
    return x.get(key) if isinstance(x, dict) else getattr(x, key, None)


def token_logprobs(response) -> list:
    """The logprob entries of the first output_text of the response (SDK object or plain dict)."""
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for c in _get(item, "content") or []:
            if _get(c, "type") == "output_text":
                return list(_get(c, "logprobs") or [])
    return []


def score_from_logprobs(entries: list) -> tuple[float, float, int]:
    """-> (expected option 1-4, probability mass on the four options, output position used).
    Uses the first of the first SCAN positions whose candidate list contains any option; NaN if none."""
    for pos, e in enumerate(entries[:SCAN]):
        cands = _get(e, "top_logprobs") or [{"token": _get(e, "token"), "logprob": _get(e, "logprob")}]
        p = {k: 0.0 for k in OPTIONS}
        for c in cands:
            t = (_get(c, "token") or "").strip()
            if t in p:
                p[t] += math.exp(_get(c, "logprob"))
        mass = sum(p.values())
        if mass > 0:
            return sum(int(k) * v for k, v in p.items()) / mass, mass, pos
    return float("nan"), 0.0, -1


def judge_one(client, prompt: str, text: str) -> dict:
    r = client.responses.create(
        model=JUDGE,
        input=prompt.replace("{t}", text),
        reasoning={"effort": "none"},
        include=["message.output_text.logprobs"],
        top_logprobs=20,
        max_output_tokens=16,
    )
    s, mass, pos = score_from_logprobs(token_logprobs(r))
    return {"score": s, "mass": mass, "pos": pos, "text": _get(r, "output_text") or ""}


def main(argv=None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="judge at most this many new conversations")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if client is None:
        from openai import OpenAI
        client = OpenAI(max_retries=6)
    prompt = norm(json.loads((args.run / "protocol.json").read_text())["config"]["texts"]["judge_prompt"])
    rows = []
    for line in (args.run / "results.jsonl").read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line the running queue is still writing
                continue
    if args.models:
        rows = [r for r in rows if r["model"] in args.models]
    out = args.run / "judge_api.jsonl"
    done = {json.loads(l)["cell_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [r for r in rows if r["cell_id"] not in done][: args.limit]
    print(f"{len(rows)} conversations, {len(done & {r['cell_id'] for r in rows})} already judged, {len(todo)} to judge now")
    lock, low = threading.Lock(), 0

    def work(r):
        return r, {name: judge_one(client, prompt, text) for name, text in replies(r).items()}

    with open(out, "a") as fh, ThreadPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(work, r) for r in todo]), 1):
            r, sc = fut.result()
            low += sum(v["mass"] < 0.9 for v in sc.values())
            with lock:
                fh.write(json.dumps({"cell_id": r["cell_id"], "model": r["model"], "judge": JUDGE,
                                     "scores": {k: v["score"] for k, v in sc.items()},
                                     "mass": {k: v["mass"] for k, v in sc.items()},
                                     "pos": {k: v["pos"] for k, v in sc.items()},
                                     "text": {k: v["text"] for k, v in sc.items()}}) + "\n")
                fh.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ratings with mass < 0.9 so far: {low}", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
