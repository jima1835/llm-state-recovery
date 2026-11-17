"""POST HOC sensitivity check on Study 1: does relaxation still beat the other arms once the wording of
the reply is removed from the internal reading?

    .venv/bin/python -m analysis.study1_wording runs/wave1 runs/wave2 [--boot 10000]
    .venv/bin/python -m analysis.study1_wording runs/wave2-8bit --sensitivity

Not registered. Study 1's results were already open when this was written (16 November 2026); the rules
below were fixed before any judge rating was compared with any reading. It needs <run>/judge_api.jsonl
(analysis/study1_judge_api.py). Nothing in the registered analysis (analysis/confirmatory.py) changes; its
loader and bootstrap scheme are reused.

Readings
    internal   the registered reading (first 64 reply tokens, Gate 0 probe, SD units)
    resid      internal minus its least-squares prediction from the judge's rating of the same reply:
               one straight line per model, fitted once over every checkpoint of every conversation
               (the rule of Study 2's H-E)
Statistics, per model, on each reading
    drop(arm) = mean(after narrative - after regulation), in SD units
    D1 = drop(relaxation) - drop(neutral_filler)        one-sided, expect > 0     MAIN RESULT
    D2 = drop(relaxation) - drop(instruction_only)      one-sided, expect > 0     MAIN RESULT
    T1, T2 = the registered recovery-fraction contrasts (drop divided by the narrative's rise)
A recovery fraction divides by the narrative's rise. On the residual that rise can be small, because the
judge sees the narrative in the words too. So T1 and T2 on the residual are shown for a model only if
its residual keeps at least half of the raw rise (kept >= 0.5); the drop contrasts need no such rule.
Sets: the unweighted mean over models, strict and extended, as registered.
Uncertainty: the registered bootstrap (narratives, shared across arms and models, then conversations),
same seed; the line is not refitted inside a draw. p-values are uncorrected and descriptive.
Conversations with a missing rating are left out of this analysis and counted. If more than 5% of
ratings have probability mass below 0.9 on the options 1-4, the result is marked not interpretable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analysis import confirmatory as C

READINGS = ("internal", "resid")
KEEP_RULE = 0.5
f, fci, short = C.f, C.fci, C.short


def attach_judge(data: dict, info: dict, runs: list[Path], models) -> dict:
    """Adds X["judge"] ([n, 4]) to every arm, aligned with the loader's row order, and drops conversations
    with a missing rating. -> quality info."""
    judge, mass = {}, []
    for run in runs:
        jf = run / "judge_api.jsonl"
        if not jf.exists():
            raise SystemExit(f"{jf} not found: run analysis/study1_judge_api.py first")
        for line in jf.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                judge[j["cell_id"]] = j["scores"]
                mass += list(j["mass"].values())
    excluded = {e["cell_id"] for e in info["excluded"]}
    acc: dict = {m: {arm: {"judge": [], "cue": [], "seed": []} for arm in data[m]} for m in data}
    for run in runs:
        for line in (run / "results.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["model"] not in data or r["cell_id"] in excluded or (models is not None and r["model"] not in models):
                continue
            a = acc[r["model"]][r["arm"]]
            s = judge.get(r["cell_id"], {})
            a["judge"].append([float(s[c]) if s.get(c) is not None else np.nan for c in C.CHECKPOINTS])
            a["cue"].append(r["factors"].get("trauma_cue"))
            a["seed"].append(r["seed"])
    missing = 0
    for m, arms in data.items():
        for arm, d in arms.items():
            a = acc[m][arm]
            if not (np.array_equal(np.array(a["cue"]), d["cue"]) and np.array_equal(np.array(a["seed"]), d["seed"])):
                raise SystemExit(f"{m} {arm}: judge rows do not line up with the loader's rows")
            J = np.array(a["judge"], float).reshape(len(d["cue"]), len(C.CHECKPOINTS))
            ok = np.isfinite(J).all(axis=1)
            missing += int((~ok).sum())
            d["cue"], d["seed"] = d["cue"][ok], d["seed"][ok]
            d["X"] = {ch: X[ok] for ch, X in d["X"].items()}
            d["X"]["judge"] = J[ok]
    mass = np.array(mass, float)
    return {"n_ratings": int(len(mass)), "low_mass_share": float((mass < 0.9).mean()) if len(mass) else np.nan,
            "missing_conversations": missing}


def add_residual(data: dict) -> dict:
    """-> per model {"slope", "intercept", "r"}; adds X["resid"]."""
    fit = {}
    for m, arms in data.items():
        z = np.concatenate([d["X"]["internal"].ravel() for d in arms.values()])
        j = np.concatenate([d["X"]["judge"].ravel() for d in arms.values()])
        if len(j) < 10 or np.std(j) == 0:
            b, a, r = 0.0, 0.0, np.nan
        else:
            b, a = np.polyfit(j, z, 1)
            r = float(np.corrcoef(j, z)[0, 1])
        fit[m] = {"slope": float(b), "intercept": float(a), "r": r}
        for d in arms.values():
            d["X"]["resid"] = d["X"]["internal"] - (a + b * d["X"]["judge"])
    return fit


def model_stats(arms: dict) -> dict:
    """arms[arm] = {reading: [n, 4]}."""
    out = {}
    for ch in READINGS:
        drop = {a: float(np.mean(arms[a][ch][:, 1] - arms[a][ch][:, 2])) for a in C.INDUCED}
        R = {a: C.recovery(arms[a][ch]) for a in C.INDUCED}
        ind = np.concatenate([arms[a][ch] for a in C.INDUCED])
        out.update({f"D1_{ch}": drop["relaxation"] - drop["neutral_filler"],
                    f"D2_{ch}": drop["relaxation"] - drop["instruction_only"],
                    f"T1_{ch}": R["relaxation"] - R["neutral_filler"],
                    f"T2_{ch}": R["relaxation"] - R["instruction_only"],
                    f"E_{ch}": float(np.mean(ind[:, 1] - ind[:, 0]))})
    out["kept"] = out["E_resid"] / out["E_internal"] if out["E_internal"] != 0 else np.nan
    return out


def bootstrap(data: dict, models, n_boot: int, seed: int = C.BOOT_SEED) -> dict:
    """The registered scheme (C.bootstrap, "narrative"), with this file's statistics."""
    rng = np.random.default_rng(seed)
    cues = sorted({c for m in models for a in data[m].values() for c in a["cue"]})
    groups = {m: {arm: {c: np.flatnonzero(a["cue"] == c) for c in cues} for arm, a in data[m].items()
                  if arm != "truncation"} for m in models}
    draws: dict = {}
    for _ in range(n_boot):
        picked = [cues[i] for i in rng.integers(0, len(cues), len(cues))]
        per = []
        for m in models:
            arms = {}
            for arm, g in groups[m].items():
                idx = np.concatenate([g[c][rng.integers(0, len(g[c]), len(g[c]))] for c in picked if len(g[c])])
                arms[arm] = {ch: data[m][arm]["X"][ch][idx] for ch in READINGS}
            per.append(model_stats(arms))
        for k in per[0]:
            draws.setdefault(k, []).append(float(np.mean([p[k] for p in per])))
    return {k: np.array(v) for k, v in draws.items()}


def analyse_set(data: dict, models, n_boot: int) -> dict:
    per = {m: model_stats({arm: a["X"] for arm, a in data[m].items() if arm in C.INDUCED}) for m in models}
    est = {k: float(np.mean([per[m][k] for m in models])) for k in next(iter(per.values()))}
    dr = bootstrap(data, models, n_boot)
    tests = {k: {"estimate": est[k], "ci": C.ci(dr[k]), "p": C.p_value(dr[k], "greater"),
                 "positive_models": int(sum(per[m][k] > 0 for m in models))}
             for k in est if k[:2] in ("D1", "D2", "T1", "T2")}
    return {"models": list(models), "tests": tests, "per_model": per, "n_boot": n_boot,
            "ratio_ok": bool(all(per[m]["kept"] >= KEEP_RULE for m in models))}


def report(data: dict, quality: dict, fit: dict, n_boot: int, sensitivity: bool) -> tuple[str, dict]:
    present = [m for m in (C.EXTENDED if not sensitivity else sorted(data)) if m in data]
    sets = [("Sensitivity set", tuple(present))] if sensitivity else \
        [(n, ms) for n, ms in (("Strict set", C.STRICT), ("Extended set", C.EXTENDED)) if all(m in data for m in ms)]
    bad = quality["low_mass_share"] > 0.05
    md = ["# Study 1, wording check (post hoc, not registered)", "",
          "Judge: gpt-6-luna on the reply text alone. The residual is the registered internal reading minus what "
          "the judge's rating of the same reply predicts (one line per model). Positive D1 or D2 on the residual "
          "means relaxation lowered the reading more than the comparison arm did, beyond what the words explain.", "",
          f"Ratings: {quality['n_ratings']}; share with probability mass below 0.9 on the options: "
          f"{quality['low_mass_share']:.3%}" + (" — **above 5%: not interpretable**" if bad else "") +
          f"; conversations left out for a missing rating: {quality['missing_conversations']}.", ""]
    js = {"quality": quality, "fit": fit, "sets": {}}
    for name, models in sets:
        res = analyse_set(data, models, n_boot)
        js["sets"][name] = res
        md += [f"## {name} ({len(models)} model{'s' if len(models) > 1 else ''})", "",
               "| Contrast | Reading | Estimate | 95% interval | p (one-sided, uncorrected) | Models positive |",
               "| --- | --- | --- | --- | --- | --- |"]
        for key, label in (("D1", "D1 drop: relaxation − filler (SD)"), ("D2", "D2 drop: relaxation − instruction (SD)"),
                           ("T1", "T1 recovery fraction: relaxation − filler"), ("T2", "T2 recovery fraction: relaxation − instruction")):
            for ch in READINGS:
                t = res["tests"][f"{key}_{ch}"]
                note = "" if key[0] == "D" or ch == "internal" or res["ratio_ok"] else " (not interpretable: a model keeps under half of the rise)"
                md.append(f"| {label} | {ch} | {f(t['estimate'])}{note} | {fci(t['ci'])} | {t['p']:.4f} | "
                          f"{t['positive_models']} of {len(models)} |")
        md.append("")
    md += ["## Per model", "",
           "| Model | r(judge, reading) | Rise, raw | Rise, residual | Kept | D1 raw | D1 resid | D2 raw | D2 resid | T1 resid | T2 resid |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in present:
        s = model_stats({arm: a["X"] for arm, a in data[m].items() if arm in C.INDUCED})
        ok = s["kept"] >= KEEP_RULE
        md.append(f"| {short(m)} | {f(fit[m]['r'])} | {f(s['E_internal'])} | {f(s['E_resid'])} | {f(s['kept'])} | "
                  f"{f(s['D1_internal'])} | {f(s['D1_resid'])} | {f(s['D2_internal'])} | {f(s['D2_resid'])} | "
                  f"{f(s['T1_resid']) if ok else 'n/a'} | {f(s['T2_resid']) if ok else 'n/a'} |")
    md += ["", "## Judge rating by arm and checkpoint (1 = not anxious, 4 = extremely; descriptive)", ""]
    for m in present:
        md += [f"### {short(m)}", ""] + C.means_table(data, m, "judge", 2) + [""]
    return "\n".join(md), js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--sensitivity", action="store_true", help="analyse the runs given as one set (8-bit repeat)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--gate0-root", type=Path, default=C.ROOT / "runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    models = None if args.sensitivity else C.EXTENDED
    data, info = C.load_runs(args.runs, args.gate0_root, models)
    quality = attach_judge(data, info, args.runs, models)
    fit = add_residual(data)
    md, js = report(data, quality, fit, args.boot, args.sensitivity)
    out = args.out or (C.ROOT / "runs" / ("analysis-wording-sensitivity" if args.sensitivity else "analysis-wording"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "study1_wording.md").write_text(md + "\n")
    (out / "study1_wording.json").write_text(json.dumps(js, indent=2, default=float))
    print(md)
    print(f"\n(written to {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
