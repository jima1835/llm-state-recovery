"""Print raw conversations from a run, to read what the models wrote. Reads results.jsonl only.

    python3 scripts/show_replies.py                                   # list models and arms with counts
    python3 scripts/show_replies.py gemma-3-4b relax_story            # 3 conversations, fixed choice
    python3 scripts/show_replies.py mistral voice_story -n 5 --full   # 5, with the full user turns
    python3 scripts/show_replies.py gemma-3-27b relax_neutral --probe # also the unrelated-question answers

The model name is any part of the model ID. Conversations are chosen with a fixed seed, so the same
command shows the same ones. User turns are cut to 200 characters unless --full is given.
Content note: the induction turns are narratives of violence, accidents and combat (Ben-Zion et al.).
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model", nargs="?")
ap.add_argument("arm", nargs="?")
ap.add_argument("-n", type=int, default=3)
ap.add_argument("--run", type=Path, default=ROOT / "runs" / "study2")
ap.add_argument("--full", action="store_true", help="print user turns in full")
ap.add_argument("--probe", action="store_true", help="also print the answer to the unrelated question at each checkpoint")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = [json.loads(l) for l in (a.run / "results.jsonl").read_text().splitlines() if l.strip()]
if not a.model or not a.arm:
    counts = Counter((r["model"].split("/")[-1], r["arm"]) for r in rows)
    for (m, arm), n in sorted(counts.items()):
        print(f"{n:4d}  {m:40s} {arm}")
    raise SystemExit

pick = [r for r in rows if a.model.lower() in r["model"].lower() and r["arm"] == a.arm]
if not pick:
    raise SystemExit(f"No conversations for model containing '{a.model}' and arm '{a.arm}'. Run without arguments to list them.")
random.Random(a.seed).shuffle(pick)
names = list(pick[0]["checkpoints"])                      # checkpoint names, in order
for r in pick[: a.n]:
    print("=" * 100)
    print(f"{r['model'].split('/')[-1]} | arm {r['arm']} | {r['factors']} | seed {r['seed']} | id {r['cell_id']}")
    turn = 0
    for msg in r["transcript"]:
        text = msg["content"]
        if msg["role"] != "assistant" and not a.full and len(text) > 200:
            text = text[:200] + f" … [{len(msg['content'])} characters]"
        print(f"\n[{msg['role'].upper()}]\n{text}")
        if msg["role"] == "assistant":
            cp = r["checkpoints"][names[turn]]
            print(f"\n    -- checkpoint {names[turn]}: questionnaire total {cp['stai']['total_expected']:.1f} (20 to 80)")
            if a.probe:
                print(f"    -- answer to the unrelated question: {cp['probe_turn']['reply']}")
            turn += 1
