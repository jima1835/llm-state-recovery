"""Read transcripts before statistics.

    .venv/bin/python -m harness.show runs/pilot-benzion-v2 [--arm relaxation] [-n 10] [--full]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--model", default=None, help="substring of the model id")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--full", action="store_true", help="print user turns in full")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in (a.run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if (a.arm is None or r["arm"] == a.arm) and (a.model is None or a.model in r["model"])]
    for r in rows[: a.n]:
        print("=" * 100)
        print(f"{r['arm']}  {r['model']}  {r['factors']}  seed={r['seed']}  cell={r['cell_id']}  {r['duration_s']}s")
        steps = {s["step"]: s for s in r["steps"]}
        cps = {cp["after_step"]: (name, cp) for name, cp in r["checkpoints"].items()}
        turn = -1
        for m in r["transcript"]:
            if m["role"] == "system":
                print(f"  [system] {m['content']}")
                continue
            if m["role"] == "user":
                turn += 1
                body = m["content"] if a.full else textwrap.shorten(m["content"], 160)
                print(f"\n  [user {turn}: {steps[turn]['spec']}] {body}")
            else:
                print(textwrap.indent(textwrap.fill(m["content"], 96), "  [model] ", lambda _: True))
                print(f"    finish={steps[turn]['finish']} tokens={steps[turn].get('reply_tokens', '?')}")
                if turn in cps:
                    name, cp = cps[turn]
                    s, v = cp.get("stai", {}), cp.get("vas", {})
                    print(f"    >> {name}: STAI {s.get('total_expected', float('nan')):.1f} (argmax {s.get('total_argmax')}, "
                          f"min option mass {s.get('min_mass')})  0-100: {v.get('value')} (raw {v.get('raw')!r})")
    print(f"\n{min(len(rows), a.n)} of {len(rows)} matching cells shown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
