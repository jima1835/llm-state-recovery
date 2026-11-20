"""Independent recomputation of the Study 2 questionnaire statistics. Shares no code with analysis/.

    python3 scripts/recompute_check.py

Standard library only. Reads runs/study2/results.jsonl, rebuilds the arm means, drops, narrative-specific
drops and the three contrasts on the questionnaire total from the raw per-conversation scores, and
compares them with the registered report (runs/study2/analysis-batch-all/study2.json).
What this covers: reading the results file, the arm labels, the checkpoint order and the arithmetic of
drop, specific, A, C, D and rise. What it does not cover: the reply reading (probe fit and projection
of activations), the bootstrap, and the judge regression; those run only through analysis/.
Exit code 1 on any difference above 1e-9.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "study2"
KINDS = ("relax", "filler", "calm", "voice", "report")
CP = ("baseline", "post_induction", "post_regulation")


def mean(xs):
    return sum(xs) / len(xs)


rows = defaultdict(lambda: defaultdict(list))            # model -> arm -> [(baseline, induction, regulation)]
for line in (RUN / "results.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        rows[r["model"]][r["arm"]].append(tuple(float(r["checkpoints"][c]["stai"]["total_expected"]) for c in CP))

report = json.loads((RUN / "analysis-batch-all" / "study2.json").read_text())
worst, n, bad = 0.0, 0, []
for model, arms in rows.items():
    mine = {}
    for kind in KINDS:
        for cond in ("story", "neutral"):
            xs = arms[f"{kind}_{cond}"]
            assert len(xs) == 60, f"{model} {kind}_{cond}: {len(xs)} conversations, expected 60"
            mine[f"drop_stai_{kind}_{cond}"] = mean([x[1] - x[2] for x in xs])
        mine[f"specific_stai_{kind}"] = mine[f"drop_stai_{kind}_story"] - mine[f"drop_stai_{kind}_neutral"]
    mine["A_stai"] = mine["specific_stai_relax"] - mine["specific_stai_filler"]
    mine["C_stai"] = mine["specific_stai_relax"] - mine["specific_stai_calm"]
    mine["D_stai"] = mine["drop_stai_voice_story"] - mine["drop_stai_relax_story"]
    story = [x for kind in KINDS for x in arms[f"{kind}_story"]]
    mine["rise_stai"] = mean([x[1] - x[0] for x in story])
    theirs = report["models"][model]["extras"]
    for key, value in mine.items():
        diff = abs(value - theirs[key]["estimate"])
        worst, n = max(worst, diff), n + 1
        if diff > 1e-9:
            bad.append(f"{model} {key}: recomputed {value:.6f}, report {theirs[key]['estimate']:.6f}")
    print(f"{model.split('/')[-1]:40s} A {mine['A_stai']:+6.2f}  C {mine['C_stai']:+6.2f}  D {mine['D_stai']:+6.2f}  rise {mine['rise_stai']:5.1f}")
print(f"\n{n} statistics over {len(rows)} models recomputed; largest difference from the report {worst:.2e}")
for b in bad:
    print("DIFFERENT:", b)
print("RESULT:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
