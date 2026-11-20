"""Independent recomputation of the Study 2 reply reading and its five tests. Shares no code with
analysis/, validation/ or harness/: numpy and the standard library only, written from the paper's text.

    .venv/bin/python scripts/recompute_reading.py        # any Python with numpy; about 3 minutes

For each of the seven models it
  1. refits the anxious-versus-calm direction from the stored Gate 0 activations (Section 2 of the paper):
     the layer with the largest Cohen's d in the middle 60% of layers, the difference of pole means
     divided coordinate-wise by the pooled within-pole variance, unit length, zero halfway between the
     poles, one unit = the pooled within-pole SD of the projections on the selection set;
  2. projects the stored first-64-token activations of every Study 2 reply (and of every probe-turn
     answer) on that direction;
  3. rebuilds H-A, H-C, H-D, H-B and H-E (the reading with a straight-line fit on the judge's rating
     removed) from the per-conversation readings;
and compares layer and estimates with the registered report (runs/study2/analysis-batch-all/study2.json).

What this covers: the direction fit, the layer choice, the projection, arm labels, checkpoint order and
the test arithmetic. What it does not cover: how the activations were taken from the model (which
tokens, which layer output), the bootstrap intervals and p-values. Exit code 1 on a difference above 1e-6.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "study2"
CHECKPOINTS = ("baseline", "post_induction", "post_regulation")
KINDS = ("relax", "filler", "calm", "voice", "report")
TOL = 1e-6


def poles(npz_path):
    """Whole-reply activations of the anxious and the calm replies, each [n, layers, width], float64."""
    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    acts = z["all"].astype(np.float64)
    anxious = acts[[i for i, m in enumerate(meta) if m["pole"] == "anxious"]]
    calm = acts[[i for i, m in enumerate(meta) if m["pole"] == "calm"]]
    return anxious, calm


def pooled_variance(a, c):
    """Pooled within-group variance along axis 0, with n - 1 in each group."""
    ss = ((a - a.mean(0)) ** 2).sum(0) + ((c - c.mean(0)) ** 2).sum(0)
    return ss / (len(a) + len(c) - 2)


def direction(a, c):
    """a, c: [n, width] at one layer -> unit vector w and the midpoint of the two pole means on w."""
    var = pooled_variance(a, c)
    w = (a.mean(0) - c.mean(0)) / (var + 1e-6 * var.mean())   # the 1e-6 term only guards against zeros
    w = w / math.sqrt(float(w @ w))
    return w, 0.5 * (float((a @ w).mean()) + float((c @ w).mean()))


def fit_reading(gate0_dir):
    """-> layer, w, midpoint, scale, following Section 2."""
    train_a, train_c = poles(gate0_dir / "acts" / "pole_train.npz")
    select_a, select_c = poles(gate0_dir / "acts" / "pole_select.npz")
    n_layers = train_a.shape[1]
    best = None
    for layer in range(int(0.2 * n_layers), math.ceil(0.8 * n_layers)):      # the middle 60% of layers
        w, mid = direction(train_a[:, layer], train_c[:, layer])
        sa, sc = select_a[:, layer] @ w, select_c[:, layer] @ w
        sd = math.sqrt(float(pooled_variance(sa, sc)))
        d = (sa.mean() - sc.mean()) / sd
        if best is None or d > best[0]:
            best = (d, layer, w, mid, sd)
    return best[1:]


results = defaultdict(list)
for line in (RUN / "results.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        results[r["model"]].append(r)
judge = {}
for line in (RUN / "judge.jsonl").read_text().splitlines():
    if line.strip():
        j = json.loads(line)
        judge[j["cell_id"]] = [float(j["scores"][c]) for c in CHECKPOINTS]
report = json.loads((RUN / "analysis-batch-all" / "study2.json").read_text())

worst, bad = 0.0, []
print(f"{'model':38s} layer   H-A    H-C    H-D    H-E    H-B   largest difference from the report")
for model, rows in results.items():
    layer, w, mid, scale = fit_reading(ROOT / "runs" / f"gate0-{model.split('/')[-1]}")
    reply, probe, rating, arm_of = [], [], [], []
    for r in rows:
        z = np.load(RUN / r["acts_file"])
        reply.append([(float(z[f"{c}__first64"][layer].astype(np.float64) @ w) - mid) / scale for c in CHECKPOINTS])
        probe.append([(float(z[f"{c}__probe"][layer].astype(np.float64) @ w) - mid) / scale for c in CHECKPOINTS])
        rating.append(judge[r["cell_id"]])
        arm_of.append(r["arm"])
    reply, probe, rating, arm_of = np.array(reply), np.array(probe), np.array(rating), np.array(arm_of)
    # H-E: one straight line of the reading on the judge's rating over every checkpoint of every conversation
    slope, intercept = np.polyfit(rating.ravel(), reply.ravel(), 1)
    resid = reply - (intercept + slope * rating)

    def specific(x, kind):
        """Mean drop (after induction minus after regulation) after a narrative, minus the same after a neutral passage."""
        drop = lambda arm: float((x[arm_of == arm, 1] - x[arm_of == arm, 2]).mean())
        return drop(f"{kind}_story") - drop(f"{kind}_neutral")

    after = lambda arm: float((reply[arm_of == arm, 1] - reply[arm_of == arm, 2]).mean())
    mine = {
        "HA": specific(reply, "relax") - specific(reply, "filler"),
        "HC": specific(reply, "relax") - specific(reply, "calm"),
        "HD": after("voice_story") - after("relax_story"),
        "HE": specific(resid, "relax") - specific(resid, "filler"),
        "HB": specific(probe, "relax") - specific(probe, "filler"),
    }
    theirs = report["models"][model]["tests"]
    diffs = {k: abs(v - theirs[k]["estimate"]) for k, v in mine.items()}
    for kind in KINDS:
        diffs[f"specific {kind}"] = abs(specific(reply, kind) - report["models"][model]["extras"][f"specific_reply_{kind}"]["estimate"])
    big = max(diffs.values()); worst = max(worst, big)
    bad += [f"{model} {k}: differs by {d:.2e}" for k, d in diffs.items() if d > TOL]
    print(f"{model.split('/')[-1]:38s} {layer:5d} " + " ".join(f"{mine[k]:+6.2f}" for k in ("HA", "HC", "HD", "HE", "HB")) + f"   {big:.1e}")
print(f"\nlargest difference over {len(results)} models: {worst:.2e}")
for b in bad:
    print("DIFFERENT:", b)
print("RESULT:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
