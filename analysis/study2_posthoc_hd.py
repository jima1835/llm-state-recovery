"""POST HOC (19 Nov, after Study 2 was unblinded): H-D with a no-narrative baseline.

    .venv/bin/python -m analysis.study2_posthoc_hd runs/study2

The pre-specified H-D compares the voice instruction with relaxation after a narrative only. This asks
the same question with each text's own no-narrative baseline, as H-A does for filler:

    H-D' = specific(relaxation) - specific(voice instruction),   reply reading, SD units,

where specific(k) is the mean drop after a narrative minus the mean drop after a neutral passage.
Positive means relaxation's narrative-specific drop is the larger one.

Rules, written before this script was run on the data:
  - analysis.study2 is used unchanged: same loading, same statistics, same bootstrap and seeds
    (10,000 draws per model; 2,000 draws that also resample models for the mean).
  - Two-sided 95% percentile intervals. No p-values, no correction: this is outside the corrected
    family and is reported as post hoc.
  - "General" by the pre-specified rule: same sign in at least six of seven models, and the interval
    that resamples models excludes zero.
Known before running: the seven point estimates follow from the published table of narrative-specific
drops (1.28, 1.17, 0.35, -0.07, -0.22, 1.35, 1.50 in the order below), so five of seven are positive
and the result cannot be general by that rule. What this adds is the intervals.
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
    stat = lambda d: d["specific_reply_relax"] - d["specific_reply_voice"]
    boot = S.bootstrap(data, models, 10000)
    wide = S.bootstrap(data, models, 2000, seed=S.BOOT_SEED + 1, resample_models=True)["_set"]
    print(f"{info['n_rows']} conversations read, {len(info['excluded'])} excluded")
    print("model                                     H-D'   95% interval      (pre-specified H-D)")
    vals = []
    for m in models:
        v = stat(est[m]); vals.append(v)
        lo, hi = np.percentile(stat(boot[m]), [2.5, 97.5])
        print(f"{m.split('/')[-1]:40s} {v:+.2f}  [{lo:+.2f}, {hi:+.2f}]    ({est[m]['HD']:+.2f})")
    vals = np.array(vals)
    lo, hi = np.percentile(stat(wide), [2.5, 97.5])
    same = int(max((vals > 0).sum(), (vals < 0).sum()))
    general = bool(same >= len(models) - 1 and (lo > 0 or hi < 0))
    print(f"\nmean of {len(models)}: {vals.mean():+.2f}  interval with models resampled [{lo:+.2f}, {hi:+.2f}]  "
          f"positive in {(vals > 0).sum()} of {len(models)}  general: {'yes' if general else 'no'}")
    # The same numbers as a file, for the paper's table (scripts/make_appendix.py).
    out = {"models": {m: {"estimate": float(stat(est[m])),
                          "ci": [float(x) for x in np.percentile(stat(boot[m]), [2.5, 97.5])]} for m in models},
           "mean": float(vals.mean()), "ci_models": [float(lo), float(hi)], "positive": int((vals > 0).sum()),
           "n": len(models), "general": general}
    (run / "analysis-batch-all" / "posthoc_hd.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
