"""POST HOC (19 Nov, after Study 2 was unblinded and after review): calm description against filler.

    .venv/bin/python -m analysis.study2_posthoc_calm runs/study2

The pre-specified tests compare relaxation with filler (H-A) and with a calm description (H-C). A
reviewer of the draft pointed out that the third side of that triangle was never tested:

    specific(calm description) - specific(filler),   reply reading, SD units,

where specific(k) is the mean drop after a narrative minus the mean drop after a neutral passage.
Positive means that a calm text with no exercise in it removes more of a narrative's rise than a
text that is not calm.

Rules, written before this script was run on the data (same as analysis/study2_posthoc_hd.py):
  - analysis.study2 is used unchanged: same loading, statistics, bootstrap and seeds
    (10,000 draws per model; 2,000 draws that also resample models for the mean).
  - Two-sided 95% percentile intervals. No p-values, no correction; outside the corrected family;
    reported as post hoc.
  - "General" by the pre-specified rule: same sign in at least six of seven models and an interval
    with models resampled that excludes zero; otherwise model-dependent.
Known before running: the seven point estimates follow from the published table of narrative-specific
drops (-0.40, +0.77, +1.01, +0.24, -0.04, +1.25, +0.12 in the order below; five of seven positive).
What this adds is the intervals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis import study2 as S


def main(argv=None) -> int:
    run = Path((argv or sys.argv[1:])[0])
    models = list(S.BATCH_A + S.BATCH_B)
    data, info = S.load(run, S.ROOT / "runs", models)
    est = {m: S.model_stats({arm: data[m][arm]["X"] for arm in S.ARMS}) for m in models}
    stat = lambda d: d["specific_reply_calm"] - d["specific_reply_filler"]
    boot = S.bootstrap(data, models, 10000)
    wide = S.bootstrap(data, models, 2000, seed=S.BOOT_SEED + 1, resample_models=True)["_set"]
    print(f"{info['n_rows']} conversations read, {len(info['excluded'])} excluded")
    vals, out = [], {"models": {}}
    for m in models:
        v = float(stat(est[m])); vals.append(v)
        lo, hi = (float(x) for x in np.percentile(stat(boot[m]), [2.5, 97.5]))
        out["models"][m] = {"estimate": v, "ci": [lo, hi]}
        print(f"{m.split('/')[-1]:40s} {v:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    vals = np.array(vals)
    lo, hi = (float(x) for x in np.percentile(stat(wide), [2.5, 97.5]))
    same = int(max((vals > 0).sum(), (vals < 0).sum()))
    general = bool(same >= len(models) - 1 and (lo > 0 or hi < 0))
    out.update({"mean": float(vals.mean()), "ci_models": [lo, hi], "positive": int((vals > 0).sum()),
                "n": len(models), "general": general})
    print(f"\nmean of {len(models)}: {vals.mean():+.2f}  interval with models resampled [{lo:+.2f}, {hi:+.2f}]  "
          f"positive in {(vals > 0).sum()} of {len(models)}  general: {'yes' if general else 'no'}")
    (run / "analysis-batch-all" / "posthoc_calm.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
