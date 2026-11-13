"""Self-report analysis: recovery fraction per arm and channel, with bootstrap uncertainty.

    .venv/bin/python -m analysis.self_report runs/gemma-v2.0 [runs/pilot-benzion-v2 ...] [--boot 5000]

Writes runs/<run>/analysis/self_report.md and .json (inside runs/, so git-ignored) and prints the
markdown. Only questionnaire totals and 0-100 ratings are used; no item text is read or written.

Channels
    stai      20-item STAI-S expected score (primary self-report, 20-80)
    stai_max  20-item STAI-S argmax score (closest to Ben-Zion's temperature-0 answers)
    vas       0-100 anxiety rating (secondary; near two-valued in the pilot)

Primary statistic (HANDOFF.md 5.1), computed per arm as a ratio of means:
    R = mean(post_regulation - post_induction) / mean(baseline - post_induction)
R = 1 is full recovery to baseline, 0 is none, < 0 is worse after regulation, > 1 overshoot.
A ratio of means is used rather than the mean of per-conversation ratios, which explode when one
conversation's induction effect is near zero.

Baseline: v2.0 runs have a baseline checkpoint in every conversation, so R is within-conversation.
The pilot has no baseline in its induced arms; there the mean baseline of the model's no_induction
arm stands in ("group baseline"), and the report says so.

Uncertainty: 95% percentile bootstrap, two ways.
    cue   resample the trauma narratives (the paraphrase unit), then conversations within each.
          Few narratives (5), so these intervals are wide and only roughly calibrated.
    cell  resample conversations within the arm.
Contrasts between arms resample each arm independently with the same scheme.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CHANNELS = {"stai": "STAI expected (primary)", "stai_max": "STAI argmax", "vas": "0-100 rating"}
INDUCED_ARMS = ("relaxation", "neutral_filler", "instruction_only")
CONTRASTS = (("relaxation", "neutral_filler", "H1"), ("relaxation", "instruction_only", "H1"),
             ("instruction_only", "neutral_filler", "H2 report side"))
CHECKPOINTS = ("baseline", "post_induction", "post_regulation", "post_return")


# ---------------------------------------------------------------- loading

def channel_value(cp: dict, ch: str) -> float:
    if ch == "stai":
        return float(cp["stai"]["total_expected"]) if "stai" in cp else np.nan
    if ch == "stai_max":
        return float(cp["stai"]["total_argmax"]) if "stai" in cp else np.nan
    if ch == "vas":
        v = cp.get("vas", {}).get("value")
        return float(v) if v is not None else np.nan
    raise KeyError(ch)


def load_rows(run: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    bad = [r["cell_id"] for r in rows if any("nonfinite_acts" in cp for cp in r["checkpoints"].values())]
    if bad:
        print(f"note: {len(bad)} rows flag non-finite internal readings (self-report unaffected)", file=sys.stderr)
    return rows


def tidy(rows: list[dict]) -> dict:
    """{model: {arm: [ {cue, seed, cell_id, values: {checkpoint: {channel: x}}} ]}}.

    Pilot rows name checkpoints differently: no_induction has baseline / baseline_2 / post_return and
    truncation has baseline / post_regulation / post_return. They are mapped onto the v2.0 slots where
    the meaning is the same, and missing slots stay missing.
    """
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        vals = {}
        for name, cp in r["checkpoints"].items():
            slot = {"baseline_2": "post_regulation"}.get(name, name)
            vals[slot] = {ch: channel_value(cp, ch) for ch in CHANNELS}
        out[r["model"]][r["arm"]].append({"cue": r["factors"].get("trauma_cue"), "seed": r["seed"],
                                          "cell_id": r["cell_id"], "values": vals})
    return out


def matrix(cells: list[dict], ch: str, group_baseline: float | None) -> np.ndarray:
    """[n, 4] array of base, ind, reg, ret for one channel; group baseline fills a missing baseline."""
    m = np.full((len(cells), 4), np.nan)
    for i, c in enumerate(cells):
        for j, slot in enumerate(CHECKPOINTS):
            if slot in c["values"]:
                m[i, j] = c["values"][slot][ch]
        if np.isnan(m[i, 0]) and group_baseline is not None:
            m[i, 0] = group_baseline
    return m


# ---------------------------------------------------------------- statistics

def recovery(m: np.ndarray) -> float:
    base, ind, reg = m[:, 0], m[:, 1], m[:, 2]
    ok = ~(np.isnan(base) | np.isnan(ind) | np.isnan(reg))
    den = np.mean(base[ok] - ind[ok])
    return float(np.mean(reg[ok] - ind[ok]) / den) if ok.any() and den != 0 else np.nan


def resample_index(cues: np.ndarray, rng: np.random.Generator, scheme: str) -> np.ndarray:
    n = len(cues)
    if scheme == "cell" or n == 0:
        return rng.integers(0, n, n)
    groups = [np.flatnonzero(cues == u) for u in np.unique(cues)]
    picked = rng.integers(0, len(groups), len(groups))
    return np.concatenate([rng.choice(groups[g], len(groups[g]), replace=True) for g in picked])


def boot(stat, mats_and_cues, n_boot: int, scheme: str, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for stat(*matrices), resampling each (matrix, cues) pair independently."""
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = stat(*[m[resample_index(c, rng, scheme)] for m, c in mats_and_cues])
    draws = draws[np.isfinite(draws)]
    if len(draws) < n_boot * 0.9:
        return (np.nan, np.nan)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def mean_col(j):
    return lambda m: float(np.nanmean(m[:, j]))


def diff_col(j, k):
    return lambda m: float(np.nanmean(m[:, j] - m[:, k]))


def cronbach_alpha(rows: list[dict], model: str) -> tuple[float, int]:
    """STAI-S internal consistency over every checkpoint of every conversation of one model,
    using per-item expected scores (already reverse-keyed)."""
    X = np.array([[it["expected"] for it in cp["stai"]["items"]]
                  for r in rows if r["model"] == model for cp in r["checkpoints"].values() if "stai" in cp])
    if len(X) < 3:
        return np.nan, len(X)
    k = X.shape[1]
    return float(k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))), len(X)


# ---------------------------------------------------------------- report

def fmt(x, ci=None, p=2):
    if x is None or not np.isfinite(x):
        return "–"
    s = f"{x:.{p}f}"
    if ci is not None and np.all(np.isfinite(ci)):
        s += f" [{ci[0]:.{p}f}, {ci[1]:.{p}f}]"
    return s


def analyse(run: Path, n_boot: int) -> tuple[str, dict]:
    rows = load_rows(run)
    data = tidy(rows)
    md = [f"# Self-report analysis: {run.name}", "",
          f"{len(rows)} conversations. 95% bootstrap intervals in brackets: **cue** resamples trauma "
          f"narratives then conversations (the paraphrase unit; few narratives, so wide), **cell** "
          f"resamples conversations. {n_boot} draws. Exploratory pilot data.", ""]
    js: dict = {"run": run.name, "n_rows": len(rows), "n_boot": n_boot, "models": {}}

    for model, arms in data.items():
        short = model.split("/")[-1]
        within = all("baseline" in c["values"] for a in INDUCED_ARMS for c in arms.get(a, []))
        gb = {}
        if not within:
            ref = arms.get("no_induction", [])
            gb = {ch: float(np.nanmean([c["values"]["baseline"][ch] for c in ref])) for ch in CHANNELS} if ref else {}
        alpha, n_alpha = cronbach_alpha(rows, model)
        mj: dict = {"baseline": "within-conversation" if within else "group (no_induction mean)",
                    "stai_alpha": alpha, "stai_alpha_n": n_alpha, "channels": {}}
        md += [f"## {short}", "",
               f"Baseline: **{mj['baseline']}**. STAI-S Cronbach's alpha over {n_alpha} checkpoints: "
               f"{fmt(alpha)}.", ""]
        if not within:
            md += ["The group baseline is a fixed number here, so its own uncertainty is not in the "
                   "intervals for R (they are too narrow by that amount).", ""]

        for ch, label in CHANNELS.items():
            cj: dict = {"means": {}, "recovery": {}, "contrasts": {}, "return": {}}
            mats = {a: matrix(cs, ch, gb.get(ch)) for a, cs in arms.items()}
            cues = {a: np.array([c["cue"] for c in cs]) for a, cs in arms.items()}
            md += [f"### {label}", "", "| Arm | n | Baseline | After induction | After regulation | After return |",
                   "| --- | --- | --- | --- | --- | --- |"]
            for a in sorted(arms, key=lambda a: (a not in INDUCED_ARMS, a)):
                m = mats[a]
                cells = []
                for j, slot in enumerate(CHECKPOINTS):
                    if np.all(np.isnan(m[:, j])):
                        cells.append("–"); continue
                    mu = mean_col(j)(m)
                    ci = boot(mean_col(j), [(m, cues[a])], n_boot // 5, "cell", seed=j)
                    cells.append(fmt(mu, ci, 1))
                    cj["means"].setdefault(a, {})[slot] = {"mean": mu, "ci_cell": ci}
                md.append(f"| {a} | {len(m)} | " + " | ".join(cells) + " |")
            md.append("")

            md += ["Recovery fraction R (1 = back to baseline, 0 = no recovery):", "",
                   "| Arm | Induction effect | R | R, cue CI | R, cell CI |", "| --- | --- | --- | --- | --- |"]
            for a in INDUCED_ARMS:
                if a not in mats:
                    continue
                m, c = mats[a], cues[a]
                eff = diff_col(1, 0)(m)
                r = recovery(m)
                ci_cue = boot(recovery, [(m, c)], n_boot, "cue", seed=11)
                ci_cell = boot(recovery, [(m, c)], n_boot, "cell", seed=12)
                cj["recovery"][a] = {"R": r, "ci_cue": ci_cue, "ci_cell": ci_cell, "induction_effect": eff}
                md.append(f"| {a} | {fmt(eff, p=1)} | {fmt(r)} | {fmt(ci_cue[0])}, "
                          f"{fmt(ci_cue[1])} | {fmt(ci_cell[0])}, {fmt(ci_cell[1])} |")
            md.append("")

            md += ["Differences in R between arms:", "", "| Contrast | For | ΔR | cue CI | cell CI |",
                   "| --- | --- | --- | --- | --- |"]
            for a, b, why in CONTRASTS:
                if a in mats and b in mats:
                    stat = lambda x, y: recovery(x) - recovery(y)  # noqa: E731
                    pair = [(mats[a], cues[a]), (mats[b], cues[b])]
                    d = stat(mats[a], mats[b])
                    ci_cue = boot(stat, pair, n_boot, "cue", seed=21)
                    ci_cell = boot(stat, pair, n_boot, "cell", seed=22)
                    cj["contrasts"][f"{a}-{b}"] = {"dR": d, "ci_cue": ci_cue, "ci_cell": ci_cell}
                    md.append(f"| {a} − {b} | {why} | {fmt(d)} | {fmt(ci_cue[0])}, {fmt(ci_cue[1])} | "
                              f"{fmt(ci_cell[0])}, {fmt(ci_cell[1])} |")
            md.append("")

            if "no_induction" in mats:
                md += ["Return test: level after the second narrative, minus the no-first-narrative arm "
                       "(> 0 means more distress when a first narrative came before):", "",
                       "| Arm | After return | minus no_induction | cell CI | Rise from after-regulation |",
                       "| --- | --- | --- | --- | --- |"]
                ref = mats["no_induction"]
                for a in INDUCED_ARMS + ("truncation",):
                    if a not in mats:
                        continue
                    m = mats[a]
                    stat = lambda x, y: float(np.nanmean(x[:, 3]) - np.nanmean(y[:, 3]))  # noqa: E731
                    d = stat(m, ref)
                    ci = boot(stat, [(m, cues[a]), (ref, cues["no_induction"])], n_boot, "cell", seed=31)
                    rise = diff_col(3, 2)(m)
                    cj["return"][a] = {"vs_no_induction": d, "ci_cell": ci, "rise_from_regulation": rise}
                    md.append(f"| {a} | {fmt(mean_col(3)(m), p=1)} | {fmt(d, p=1)} | {fmt(ci[0], p=1)}, "
                              f"{fmt(ci[1], p=1)} | {fmt(rise, p=1)} |")
                md.append("")

            if ch == "vas":
                vals = np.concatenate([mats[a][:, 1:].ravel() for a in mats])
                vals = vals[np.isfinite(vals)]
                u, n = np.unique(vals, return_counts=True)
                top = sorted(zip(n, u), reverse=True)[:5]
                share = sum(k for k, _ in top[:2]) / len(vals) if len(vals) else np.nan
                md += [f"0-100 rating: the two most common values cover {share:.0%} of answers "
                       f"(most common: " + ", ".join(f"{int(v)} ×{k}" for k, v in top) + ").", ""]
                cj["vas_top_values"] = [(float(v), int(k)) for k, v in top]
            mj["channels"][ch] = cj
        js["models"][model] = mj
    return "\n".join(md), js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args(argv)
    for run in args.runs:
        md, js = analyse(run, args.boot)
        out = run / "analysis"
        out.mkdir(exist_ok=True)
        (out / "self_report.md").write_text(md + "\n")
        (out / "self_report.json").write_text(json.dumps(js, indent=2, default=float))
        print(md)
        print(f"\n(written to {out}/self_report.md and .json)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
