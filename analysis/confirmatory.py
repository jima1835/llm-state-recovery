"""Confirmatory analysis: the six registered tests, for the strict set and the extended set.

    .venv/bin/python -m analysis.confirmatory runs/wave1 runs/wave2 --unblind [--boot 10000]
    .venv/bin/python -m analysis.confirmatory runs/wave2-8bit --unblind --sensitivity

Written and tested on synthetic data before any confirmatory result was opened (HANDOFF.md section 11
lists every choice made here that the registration left open). It refuses to read a real run without
--unblind, so that opening the results is a deliberate, logged step.

Channels (one number per conversation checkpoint)
    internal       registered reading: Martorell & Bianchi probe at the Gate 0 layer, first 64 reply
                   tokens, in SD units (0 = halfway between the poles). PRIMARY internal channel.
    internal_all   the same probe over the whole reply (secondary)
    internal_mean  plain mean-difference probe, first 64 tokens (secondary)
    stai           20-item questionnaire, expected score. PRIMARY self-report channel.
    vas            0-100 rating (secondary)
The probe is refitted from the stored Gate 0 replies exactly as the Gate 0 report did (same code), and
its layer is checked against the Gate 0 report. Nothing is fitted on confirmatory data.

Statistics, per model
    R(arm, channel) = mean(post_induction - post_regulation) / mean(post_induction - baseline)
    E               = mean(post_induction - baseline) on the internal channel, over the induced arms
    T1 H1a       R_internal(relaxation) - R_internal(neutral_filler)            expect > 0
    T2 H1b       R_internal(relaxation) - R_internal(instruction_only)          expect > 0
    T3 H2a       R_stai(instruction_only)                                       expect > 0
    T4 H2b       R_stai(instruction_only) - R_internal(instruction_only)        expect > 0
    T5 Return a  [mean of the three induced arms' post_return - no_induction's post_return] / E   expect > 0
    T6 Return b  [post_return(relaxation) - post_return(neutral_filler)] / E    two-sided
A set's statistic is the unweighted mean of its models' statistics, so every model counts equally and
each channel is scaled to its own induction effect within each model.

Uncertainty (registered: "bootstrap, resampling narratives and then conversations")
    One draw resamples the five narratives with replacement (the same draw for every arm and model,
    because the narratives are the same texts everywhere), then resamples conversations within each
    narrative x arm x model cell. 95% percentile intervals. One-sided p = share of draws at or below 0
    (two-sided for T6). Benjamini-Hochberg at 5% across the six tests, separately for each set.
    A second interval that resamples conversations only (narratives fixed) is shown for comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ("baseline", "post_induction", "post_regulation", "post_return")
INDUCED = ("relaxation", "neutral_filler", "instruction_only")
ARMS = INDUCED + ("no_induction", "truncation")
EXPECTED_N = {"relaxation": 90, "neutral_filler": 30, "instruction_only": 30, "no_induction": 30, "truncation": 15}
STRICT = ("mlx-community/gemma-3-4b-it-4bit",)
EXTENDED = STRICT + ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "mlx-community/gemma-3-12b-it-4bit",
                     "mlx-community/Qwen3-14B-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit",
                     "mlx-community/Mistral-Small-24B-Instruct-2501-4bit", "mlx-community/gemma-3-27b-it-4bit")
TESTS = (("T1", "H1a: relaxation recovers more than neutral filler (internal)", "greater"),
         ("T2", "H1b: relaxation recovers more than instruction only (internal)", "greater"),
         ("T3", "H2a: self-report recovers under instruction only", "greater"),
         ("T4", "H2b: self-report recovers more than the internal reading under instruction only", "greater"),
         ("T5", "Return a: reading after the second narrative is higher after a first narrative", "greater"),
         ("T6", "Return b: that reading differs between relaxation and neutral filler", "two-sided"))
BOOT_SEED = 20261113
ALPHA = 0.05


# ---------------------------------------------------------------- loading

def short(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def load_probes(model: str, gate0_root: Path) -> dict:
    """Refit the registered reading from the stored Gate 0 replies (same code as the Gate 0 report)."""
    from validation import gate0
    d = gate0_root / f"gate0-{short(model)}"
    if not (d / "protocol.json").exists():
        raise SystemExit(f"No Gate 0 run for {model} at {d}; the reading cannot be scored.")
    fr = gate0.fit_reading(gate0.Store(d))
    rep = d / "gate0_report.json"
    if rep.exists():
        reported = json.loads(rep.read_text()).get("layer")
        if reported is not None and reported != fr["layer"]:
            raise SystemExit(f"{model}: refitted layer {fr['layer']} differs from the Gate 0 report's {reported}.")
    return {"layer": fr["layer"], "mb": fr["probes"]["mb"], "mean": fr["probes"]["mean"]}


def load_runs(runs: list[Path], gate0_root: Path, models: tuple[str, ...] | None = None):
    """-> (data, info). data[model][arm] = {"cue": [n] str, "seed": [n], "X": {channel: [n, 4]}}.
    A conversation is excluded only for a missing or non-finite internal reading or a missing
    questionnaire score (registered rule: run errors and non-finite activations); each is listed."""
    rows = []
    for run in runs:
        for line in (run / "results.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_run"] = run
                rows.append(r)
    seen = Counter(r["cell_id"] for r in rows)
    dup = [c for c, n in seen.items() if n > 1]
    if dup:
        raise SystemExit(f"{len(dup)} conversation ids appear more than once across the runs given.")
    probes, excluded = {}, []
    items: dict = defaultdict(list)  # per model: questionnaire item scores at every checkpoint
    acc: dict = defaultdict(lambda: defaultdict(lambda: {"cue": [], "seed": [], "X": defaultdict(list)}))
    for r in rows:
        m = r["model"]
        if models is not None and m not in models:
            continue
        if m not in probes:
            probes[m] = load_probes(m, gate0_root)
        p = probes[m]
        why = None
        if any(c not in r["checkpoints"] for c in CHECKPOINTS):
            why = "missing checkpoint"
        vals: dict = defaultdict(list)
        if why is None:
            z = np.load(r["_run"] / r["acts_file"])
            for c in CHECKPOINTS:
                first, whole = z[f"{c}__first64"][p["layer"]], z[c][p["layer"]]
                if not (np.isfinite(first).all() and np.isfinite(whole).all()):
                    why = "non-finite activations"
                    break
                cp = r["checkpoints"][c]
                vals["internal"].append(float(p["mb"].z(first[None])[0]))
                vals["internal_all"].append(float(p["mb"].z(whole[None])[0]))
                vals["internal_mean"].append(float(p["mean"].z(first[None])[0]))
                vals["stai"].append(float(cp["stai"]["total_expected"]) if "stai" in cp else np.nan)
                v = cp.get("vas", {}).get("value")
                vals["vas"].append(float(v) if v is not None else np.nan)
            if why is None and not np.isfinite(vals["stai"]).all():
                why = "missing questionnaire score"
        if why:
            excluded.append({"cell_id": r["cell_id"], "model": m, "arm": r["arm"], "reason": why})
            continue
        items[m] += [[it["expected"] for it in r["checkpoints"][c]["stai"]["items"]] for c in CHECKPOINTS
                     if "items" in r["checkpoints"][c].get("stai", {})]
        a = acc[m][r["arm"]]
        a["cue"].append(r["factors"].get("trauma_cue"))
        a["seed"].append(r["seed"])
        for ch, v in vals.items():
            a["X"][ch].append(v)
    data = {m: {arm: {"cue": np.array(a["cue"]), "seed": np.array(a["seed"]),
                      "X": {ch: np.array(v, float) for ch, v in a["X"].items()}}
                for arm, a in arms.items()} for m, arms in acc.items()}
    rel = {}
    for m, arms in data.items():
        a = np.concatenate([x["X"]["internal"].ravel() for x in arms.values()])
        b = np.concatenate([x["X"]["internal_all"].ravel() for x in arms.values()])
        rel[m] = {"stai_alpha": cronbach_alpha(np.array(items[m], float)),
                  "readout_r": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan}
    info = {"n_rows": len(rows), "excluded": excluded, "reliability": rel,
            "layers": {m: p["layer"] for m, p in probes.items()},
            "counts": {m: {arm: len(a["cue"]) for arm, a in arms.items()} for m, arms in data.items()},
            "seeds": sorted({int(s) for arms in data.values() for a in arms.values() for s in a["seed"]})}
    return data, info


# ---------------------------------------------------------------- statistics

def cronbach_alpha(X: np.ndarray) -> float:
    """Internal consistency of the questionnaire. X: [observations, items], already reverse-keyed."""
    if X.ndim != 2 or len(X) < 3:
        return np.nan
    k = X.shape[1]
    total = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / total)) if total > 0 else np.nan


def recovery(X: np.ndarray) -> float:
    den = np.mean(X[:, 1] - X[:, 0])
    return float(np.mean(X[:, 1] - X[:, 2]) / den) if den != 0 else np.nan


def model_stats(arms: dict, internal: str = "internal", report: str = "stai") -> dict:
    """The six test statistics and their ingredients for one model. arms[arm] = {channel: [n, 4]}."""
    I = {a: arms[a][internal] for a in arms}
    S = {a: arms[a][report] for a in arms}
    out = {f"R_int_{a}": recovery(I[a]) for a in INDUCED}
    out.update({f"R_rep_{a}": recovery(S[a]) for a in INDUCED})
    induced = np.concatenate([I[a] for a in INDUCED])
    E = float(np.mean(induced[:, 1] - induced[:, 0]))
    out["E_int"] = E
    out["E_rep"] = float(np.nanmean(np.concatenate([S[a] for a in INDUCED])[:, 1]
                                    - np.concatenate([S[a] for a in INDUCED])[:, 0]))
    ret = {a: float(np.mean(I[a][:, 3])) for a in I}
    out["T1"] = out["R_int_relaxation"] - out["R_int_neutral_filler"]
    out["T2"] = out["R_int_relaxation"] - out["R_int_instruction_only"]
    out["T3"] = out["R_rep_instruction_only"]
    out["T4"] = out["R_rep_instruction_only"] - out["R_int_instruction_only"]
    out["T5"] = (np.mean([ret[a] for a in INDUCED]) - ret["no_induction"]) / E if E != 0 else np.nan
    out["T6"] = (ret["relaxation"] - ret["neutral_filler"]) / E if E != 0 else np.nan
    return out


def set_stats(per_model: dict, models) -> dict:
    keys = next(iter(per_model.values())).keys()
    return {k: float(np.mean([per_model[m][k] for m in models])) for k in keys}


def bootstrap(data: dict, models, n_boot: int, scheme: str = "narrative", internal: str = "internal",
              report: str = "stai", seed: int = BOOT_SEED) -> dict:
    """-> {statistic: [n_boot] draws of the set statistic}. scheme: "narrative" (registered) resamples
    the narratives, shared across arms and models, then conversations; "conversation" keeps the
    narratives fixed."""
    rng = np.random.default_rng(seed)
    cues = sorted({c for m in models for a in data[m].values() for c in a["cue"]})
    groups = {m: {arm: {c: np.flatnonzero(a["cue"] == c) for c in cues} for arm, a in data[m].items()
                  if arm != "truncation"} for m in models}
    mats = {m: {arm: {ch: a["X"][ch] for ch in (internal, report)} for arm, a in data[m].items()
                if arm != "truncation"} for m in models}
    draws: dict = defaultdict(list)
    for _ in range(n_boot):
        picked = [cues[i] for i in rng.integers(0, len(cues), len(cues))] if scheme == "narrative" else cues
        per = {}
        for m in models:
            arms = {}
            for arm, g in groups[m].items():
                idx = np.concatenate([g[c][rng.integers(0, len(g[c]), len(g[c]))] for c in picked if len(g[c])])
                arms[arm] = {ch: X[idx] for ch, X in mats[m][arm].items()}
            per[m] = model_stats(arms, internal, report)
        for k, v in set_stats(per, models).items():
            draws[k].append(v)
    return {k: np.array(v) for k, v in draws.items()}


def p_value(draws: np.ndarray, side: str) -> float:
    d = draws[np.isfinite(draws)]
    lo = (1 + np.sum(d <= 0)) / (1 + len(d))
    if side == "greater":
        return float(lo)
    hi = (1 + np.sum(d >= 0)) / (1 + len(d))
    return float(min(1.0, 2 * min(lo, hi)))


def benjamini_hochberg(p: list[float]) -> list[float]:
    """Adjusted p-values (step-up); a test is confirmed when its adjusted p is below the level."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 1.0
    for rank, i in zip(range(len(p), 0, -1), order[::-1]):
        running = min(running, p[i] * len(p) / rank)
        adj[i] = running
    return adj.tolist()


def ci(draws: np.ndarray) -> tuple[float, float]:
    d = draws[np.isfinite(draws)]
    if len(d) < 0.9 * len(draws):
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.percentile(d, [2.5, 97.5]))


def analyse_set(data: dict, models, n_boot: int, internal: str = "internal", report: str = "stai") -> dict:
    per = {m: model_stats({arm: a["X"] for arm, a in data[m].items()}, internal, report) for m in models}
    est = set_stats(per, models)
    nar = bootstrap(data, models, n_boot, "narrative", internal, report)
    conv = bootstrap(data, models, max(1000, n_boot // 5), "conversation", internal, report, seed=BOOT_SEED + 1)
    ps = [p_value(nar[t], side) for t, _, side in TESTS]
    adj = benjamini_hochberg(ps)
    tests = {}
    for (t, label, side), p, q in zip(TESTS, ps, adj):
        tests[t] = {"label": label, "side": side, "estimate": est[t], "ci": ci(nar[t]), "ci_conversation": ci(conv[t]),
                    "p": p, "p_bh": q, "confirmed": bool(q < ALPHA and (side == "two-sided" or est[t] > 0))}
    extras = {k: {"estimate": est[k], "ci": ci(nar[k])} for k in est if k not in tests}
    return {"models": list(models), "tests": tests, "extras": extras, "per_model": per, "n_boot": n_boot}


# ---------------------------------------------------------------- report

def f(x, p=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def fci(c, p=2):
    return f"[{f(c[0], p)}, {f(c[1], p)}]"


def tests_table(res: dict) -> list[str]:
    md = ["| Test | Estimate | 95% interval (narratives, then conversations) | Interval, conversations only | p | "
          "p after correction | Confirmed |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for t, _, _ in TESTS:
        r = res["tests"][t]
        md.append(f"| {t} {r['label']} | {f(r['estimate'])} | {fci(r['ci'])} | {fci(r['ci_conversation'])} | "
                  f"{r['p']:.4f} | {r['p_bh']:.4f} | {'**yes**' if r['confirmed'] else 'no'} |")
    return md


def means_table(data: dict, model: str, ch: str, p: int) -> list[str]:
    md = ["| Arm | n | Baseline | After narrative | After regulation | After second narrative |",
          "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        if arm in data[model]:
            X = data[model][arm]["X"][ch]
            md.append(f"| {arm} | {len(X)} | " + " | ".join(f(v, p) for v in np.nanmean(X, axis=0)) + " |")
    return md


def code_version() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "analysis/confirmatory.py"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return h + (" (analysis file has uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"


def check_complete(info: dict, models) -> list[str]:
    problems = []
    for m in models:
        got = info["counts"].get(m)
        if got is None:
            problems.append(f"{short(m)}: no conversations found")
            continue
        n_ex = Counter(e["arm"] for e in info["excluded"] if e["model"] == m)
        for arm, n in EXPECTED_N.items():
            if got.get(arm, 0) + n_ex.get(arm, 0) != n:
                problems.append(f"{short(m)} {arm}: {got.get(arm, 0)} analysed + {n_ex.get(arm, 0)} excluded, expected {n}")
    return problems


def report(data: dict, info: dict, n_boot: int, strict=STRICT, extended=EXTENDED, sensitivity: bool = False):
    present = [m for m in data]
    md = ["# Confirmatory analysis", "",
          f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. Code version: {code_version()}.",
          f"{info['n_rows']} conversations read, {len(info['excluded'])} excluded. Seeds: {info['seeds']}. "
          f"Bootstrap draws: {n_boot}, seed {BOOT_SEED}.", ""]
    js: dict = {"info": {k: v for k, v in info.items()}, "sets": {}, "secondary": {}}
    if info["excluded"]:
        md += ["Excluded conversations (registered rule: run errors or non-finite activations only):", ""]
        md += [f"- {short(e['model'])}, {e['arm']}, `{e['cell_id']}`: {e['reason']}" for e in info["excluded"]] + [""]
    sets = [("Sensitivity run (belongs to neither set)", tuple(present))] if sensitivity else \
        [("Strict set (passes Gate 0 as registered)", strict), ("Extended set (rests on deviation 1)", extended)]
    for name, models in sets:
        missing = [m for m in models if m not in data]
        if missing:
            raise SystemExit(f"{name}: no data for {[short(m) for m in missing]}. Give every run of the set.")
        problems = check_complete(info, models)
        res = analyse_set(data, models, n_boot)
        js["sets"][name] = res
        md += [f"## {name}", "", "Models: " + ", ".join(short(m) for m in models) + ".", ""]
        if problems:
            md += ["**Incomplete cells:** " + "; ".join(problems) + ".", ""]
        md += tests_table(res) + ["",
               "Estimates are in units of recovery fraction (T1 to T4) or of the model's own induction effect "
               "(T5, T6). Correction: Benjamini-Hochberg across the six tests of this set.", "",
               "Ingredients (set means, 95% interval):", "",
               "| Quantity | Estimate | 95% interval |", "| --- | --- | --- |"]
        for k, v in res["extras"].items():
            md.append(f"| {k} | {f(v['estimate'])} | {fci(v['ci'])} |")
        md += ["", "Induction check: E_int and E_rep are the rise from baseline to after the narrative. "
               "R is only meaningful where that rise is clearly above zero.", ""]
        if len(models) > 1:
            md += ["Per model (descriptive):", "", "| Model | Layer | " + " | ".join(t for t, _, _ in TESTS) +
                   " | E_int | E_rep |", "| --- | --- | " + " | ".join("---" for _ in range(8)) + " |"]
            for m in models:
                s = res["per_model"][m]
                md.append(f"| {short(m)} | {info['layers'][m]} | " + " | ".join(f(s[t]) for t, _, _ in TESTS) +
                          f" | {f(s['E_int'])} | {f(s['E_rep'], 1)} |")
            md.append("")
        md += ["Secondary readouts (same tests, not confirmatory):", "",
               "| Variant | " + " | ".join(t for t, _, _ in TESTS) + " |", "| --- | " + " | ".join("---" for _ in TESTS) + " |"]
        for label, internal, rep in (("whole-reply readout", "internal_all", "stai"),
                                     ("mean-difference probe", "internal_mean", "stai"),
                                     ("0-100 rating as the self-report", "internal", "vas")):
            sec = analyse_set(data, models, max(1000, n_boot // 5), internal, rep)
            js["secondary"][f"{name} | {label}"] = sec
            md.append(f"| {label} | " + " | ".join(f"{f(sec['tests'][t]['estimate'])} {fci(sec['tests'][t]['ci'])}"
                                                     for t, _, _ in TESTS) + " |")
        md.append("")
    if info.get("reliability"):
        md += ["## Reliability of the two channels (descriptive)", "",
               "No reliability correction is applied to R: it is a ratio of mean differences, and random "
               "measurement error averages out of means (it widens the intervals, which the bootstrap already "
               "reflects). The figures below show how noisy each channel is.", "",
               "| Model | Questionnaire: Cronbach's alpha | Internal reading: correlation of the first-64-token "
               "and whole-reply readouts |", "| --- | --- | --- |"]
        md += [f"| {short(m)} | {f(v['stai_alpha'])} | {f(v['readout_r'])} |" for m, v in info["reliability"].items()]
        md.append("")
    md += ["## Means by arm and checkpoint (descriptive)", ""]
    for m in present:
        md += [f"### {short(m)}", "", "Internal reading (SD units; 0 is halfway between the poles):", ""]
        md += means_table(data, m, "internal", 2) + ["", "Questionnaire (20 to 80):", ""] + means_table(data, m, "stai", 1) + [""]
    from analysis import mixed
    md += ["## Mixed-effects regression (second analysis)", ""]
    if mixed.available():
        md += [mixed.INTRO, ""]
        js["mixed"] = {}
        for name, models in sets:
            res = mixed.analyse_set(data, models)
            js["mixed"][name] = res
            md += mixed.section(name, res, js["sets"][name])
    else:
        md += ["Not run: the `scipy` package is not installed in this environment.", ""]
    return "\n".join(md), js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--unblind", action="store_true", help="required to read a real (non-fake) run")
    ap.add_argument("--sensitivity", action="store_true", help="analyse the runs given as one set (8-bit repeat)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--gate0-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    real = [r for r in args.runs if not r.name.startswith("fake")]
    if real and not args.unblind:
        print("These are real runs. Opening confirmatory results is a deliberate step: add --unblind, and log "
              "the date and the code version in TRACKING.md first.")
        return 2
    models = None if args.sensitivity else EXTENDED
    data, info = load_runs(args.runs, args.gate0_root, models)
    md, js = report(data, info, args.boot, sensitivity=args.sensitivity)
    out = args.out or (ROOT / "runs" / ("analysis-sensitivity" if args.sensitivity else "analysis-confirmatory"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirmatory.md").write_text(md + "\n")
    (out / "confirmatory.json").write_text(json.dumps(js, indent=2, default=float))
    print(md)
    print(f"\n(written to {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
