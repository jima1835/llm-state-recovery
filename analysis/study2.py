"""Study 2 confirmatory analysis (registration/study2_plan.md, section 4).

    .venv/bin/python -m analysis.study2 runs/study2 --batch A --unblind
    .venv/bin/python -m analysis.study2 runs/study2 --batch all --unblind

Written and tested on synthetic data before any Study 2 confirmatory result was opened. Refuses a real
run without --unblind, and refuses a batch whose models are not all complete.

Readings (SD units, Gate 0 probe and layer, nothing refitted)
    reply    first 64 tokens of the reply to the turn                         primary
    probe    reply to the fixed neutral question on a discarded fork          secondary
    resid    reply reading minus what the judge's rating of the reply's words predicts (H-E)
Outcome per conversation: drop = reading after the induction slot - reading after the regulation slot.
    specific(kind) = mean drop(kind after a narrative) - mean drop(kind after a neutral passage)

Tests, per model
    HA  specific(relax) - specific(filler), reply reading               one-sided, expect > 0
    HC  specific(relax) - specific(calm), reply reading                 one-sided, expect > 0
    HD  drop(voice after a narrative) - drop(relax after a narrative)   two-sided
    HE  HA on the resid reading                                         one-sided, expect > 0
    HB  HA on the probe reading                                         two-sided
Benjamini-Hochberg across HA, HC, HD, HE. HB joins that family for a model only if its probe-turn
reading rises by at least 0.5 SD after a narrative (interval above zero) in this run; otherwise it is
reported outside the family.

Uncertainty: one draw resamples the five narratives (the same draw for every story arm), then
conversations within narrative x arm; in the neutral-passage arms conversations are resampled within
passage (two passages are too few to resample). 10,000 draws, percentile intervals.
Across models: sign count, and a second bootstrap that also resamples models. A result is called
general only if every model of batch A (or at least six of seven with both batches) has the same sign
and that interval excludes zero.
Second analysis: mixed-effects regression of the drop on the four cells of each contrast, with random
intercepts for model and narrative and a random slope of the contrast across models.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analysis import confirmatory as C

ROOT = C.ROOT
CP = ("baseline", "post_induction", "post_regulation")
KINDS = ("relax", "filler", "calm", "voice", "report")
ARMS = tuple(f"{k}_{s}" for k in KINDS for s in ("story", "neutral"))
N_PER_ARM = 60
BATCH_A = ("mlx-community/gemma-3-4b-it-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit",
           "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "mlx-community/gemma-3-12b-it-4bit")
BATCH_B = ("mlx-community/Qwen3-14B-4bit", "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
           "mlx-community/gemma-3-27b-it-4bit")
PRIMARY = BATCH_A[0]
TESTS = (("HA", "Relaxation lowers the reading more after a narrative than after neutral text, beyond filler", "greater", "reply"),
         ("HC", "The same, beyond calm text that is not an exercise", "greater", "reply"),
         ("HD", "A calm-voice instruction against relaxation, after a narrative", "two-sided", "reply"),
         ("HE", "HA after removing what the words of the reply predict", "greater", "resid"),
         ("HB", "HA on the reply to the fixed neutral question", "two-sided", "probe"))
FAMILY = ("HA", "HC", "HD", "HE")
P1_MIN_RISE = 0.5
BOOT_SEED = 20261115


# ---------------------------------------------------------------- loading

def load(run: Path, gate0_root: Path, models) -> tuple[dict, dict]:
    """-> data[model][arm] = {"cue": [n], "X": {reading: [n, 3]}}, info."""
    judge = {}
    jf = run / "judge.jsonl"
    if jf.exists():
        for line in jf.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                judge[j["cell_id"]] = j["scores"]
    acc: dict = defaultdict(lambda: defaultdict(lambda: {"cue": [], "X": defaultdict(list)}))
    probes, excluded, n_rows, seen = {}, [], 0, Counter()
    for line in (run / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = r["model"]
        if m not in models:
            continue
        n_rows += 1
        seen[r["cell_id"]] += 1
        if m not in probes:
            probes[m] = C.load_probes(m, gate0_root)
        p, why, vals = probes[m], None, {}
        if any(c not in r["checkpoints"] for c in CP):
            why = "missing checkpoint"
        else:
            z = np.load(run / r["acts_file"])
            a = {"reply": [z[f"{c}__first64"][p["layer"]] for c in CP], "probe": [z[f"{c}__probe"][p["layer"]] for c in CP],
                 "reply_all": [z[c][p["layer"]] for c in CP]}
            if not all(np.isfinite(v).all() for vs in a.values() for v in vs):
                why = "non-finite activations"
            else:
                vals = {k: [float(p["mb"].z(v[None])[0]) for v in vs] for k, vs in a.items()}
                vals["stai"] = [float(r["checkpoints"][c]["stai"]["total_expected"]) for c in CP]
                js = judge.get(r["cell_id"])
                vals["judge"] = [float(js[c]) for c in CP] if js else [np.nan] * 3
        if why:
            excluded.append({"cell_id": r["cell_id"], "model": m, "arm": r["arm"], "reason": why})
            continue
        f = r["factors"]
        d = acc[m][r["arm"]]
        d["cue"].append(f.get("trauma_cue") or f.get("neutral_cue"))
        for k, v in vals.items():
            d["X"][k].append(v)
    if any(n > 1 for n in seen.values()):
        raise SystemExit("A conversation id appears more than once.")
    data = {m: {arm: {"cue": np.array(d["cue"]), "X": {k: np.array(v, float) for k, v in d["X"].items()}}
                for arm, d in arms.items()} for m, arms in acc.items()}
    for m in data:
        add_residual(data[m])
    info = {"n_rows": n_rows, "excluded": excluded, "layers": {m: p["layer"] for m, p in probes.items()},
            "counts": {m: {arm: len(d["cue"]) for arm, d in arms.items()} for m, arms in data.items()},
            "judged": {m: bool(np.isfinite(np.concatenate([d["X"]["judge"].ravel() for d in arms.values()])).all())
                       for m, arms in data.items()}}
    return data, info


def add_residual(arms: dict) -> None:
    """resid = reply reading minus its least-squares prediction from the judge's rating of the same
    reply, fitted once per model over every checkpoint of every conversation."""
    z = np.concatenate([d["X"]["reply"].ravel() for d in arms.values()])
    j = np.concatenate([d["X"]["judge"].ravel() for d in arms.values()])
    ok = np.isfinite(j)
    if ok.sum() < 10 or np.std(j[ok]) == 0:
        for d in arms.values():
            d["X"]["resid"] = np.full_like(d["X"]["reply"], np.nan)
        return
    b, a = np.polyfit(j[ok], z[ok], 1)
    for d in arms.values():
        d["X"]["resid"] = d["X"]["reply"] - (a + b * d["X"]["judge"])


def incomplete(info: dict, models) -> list[str]:
    out = []
    for m in models:
        got = info["counts"].get(m, {})
        ex = Counter(e["arm"] for e in info["excluded"] if e["model"] == m)
        out += [f"{C.short(m)} {arm}: {got.get(arm, 0)} + {ex.get(arm, 0)} excluded of {N_PER_ARM}"
                for arm in ARMS if got.get(arm, 0) + ex.get(arm, 0) != N_PER_ARM]
    return out


# ---------------------------------------------------------------- statistics

def drop(X: np.ndarray) -> float:
    return float(np.mean(X[:, 1] - X[:, 2]))


def model_stats(arms: dict) -> dict:
    """arms[arm] = {reading: [n, 3]} -> the five test statistics and their ingredients."""
    out = {}
    for rd in ("reply", "probe", "resid", "stai", "reply_all"):
        d = {arm: drop(arms[arm][rd]) for arm in ARMS}
        sp = {k: d[f"{k}_story"] - d[f"{k}_neutral"] for k in KINDS}
        out.update({f"drop_{rd}_{arm}": v for arm, v in d.items()})
        out.update({f"specific_{rd}_{k}": v for k, v in sp.items()})
        out[f"A_{rd}"] = sp["relax"] - sp["filler"]
        out[f"C_{rd}"] = sp["relax"] - sp["calm"]
        out[f"D_{rd}"] = d["voice_story"] - d["relax_story"]
        story = np.concatenate([arms[f"{k}_story"][rd] for k in KINDS])
        out[f"rise_{rd}"] = float(np.mean(story[:, 1] - story[:, 0]))
    out.update({"HA": out["A_reply"], "HC": out["C_reply"], "HD": out["D_reply"], "HE": out["A_resid"],
                "HB": out["A_probe"]})
    return out


def bootstrap(data: dict, models, n_boot: int, seed: int = BOOT_SEED, resample_models: bool = False) -> dict:
    """-> {model or "_set": {statistic: [n_boot]}}. "_set" is the mean over (resampled) models."""
    rng = np.random.default_rng(seed)
    cues = sorted({c for m in models for arm in ARMS if arm.endswith("_story") for c in data[m][arm]["cue"]})
    groups = {m: {arm: {c: np.flatnonzero(data[m][arm]["cue"] == c) for c in sorted(set(data[m][arm]["cue"]))}
                  for arm in ARMS} for m in models}
    draws: dict = {m: defaultdict(list) for m in list(models) + ["_set"]}
    for _ in range(n_boot):
        picked = [cues[i] for i in rng.integers(0, len(cues), len(cues))]
        per = {}
        for m in models:
            arms = {}
            for arm in ARMS:
                g = groups[m][arm]
                use = picked if arm.endswith("_story") else list(g)
                idx = np.concatenate([g[c][rng.integers(0, len(g[c]), len(g[c]))] for c in use if c in g and len(g[c])])
                arms[arm] = {rd: X[idx] for rd, X in data[m][arm]["X"].items()}
            per[m] = model_stats(arms)
            for k, v in per[m].items():
                draws[m][k].append(v)
        ms = [models[i] for i in rng.integers(0, len(models), len(models))] if resample_models else list(models)
        for k in per[models[0]]:
            draws["_set"][k].append(float(np.mean([per[m][k] for m in ms])))
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in draws.items()}


def analyse(data: dict, models, n_boot: int) -> dict:
    est = {m: model_stats({arm: data[m][arm]["X"] for arm in ARMS}) for m in models}
    boot = bootstrap(data, models, n_boot)
    res = {"models": {}, "n_boot": n_boot}
    for m in models:
        rise = (est[m]["rise_probe"], *C.ci(boot[m]["rise_probe"]))
        p1 = bool(rise[0] >= P1_MIN_RISE and rise[1] > 0)
        family = FAMILY + (("HB",) if p1 else ())
        tests = {}
        for t, label, side, rd in TESTS:
            d = boot[m][t]
            tests[t] = {"label": label, "side": side, "estimate": est[m][t], "ci": C.ci(d),
                        "p": C.p_value(d, side) if np.isfinite(est[m][t]) else np.nan, "in_family": t in family}
        fam = [t for t in family if np.isfinite(tests[t]["p"])]
        for t, q in zip(fam, C.benjamini_hochberg([tests[t]["p"] for t in fam])):
            tests[t]["p_bh"] = q
            tests[t]["confirmed"] = bool(q < C.ALPHA and (tests[t]["side"] == "two-sided" or tests[t]["estimate"] > 0))
        res["models"][m] = {"tests": tests, "probe_rise": rise, "p1": p1,
                            "extras": {k: {"estimate": v, "ci": C.ci(boot[m][k])} for k, v in est[m].items()
                                       if k not in tests}}
    if len(models) > 1:
        wide = bootstrap(data, models, max(2000, n_boot // 5), seed=BOOT_SEED + 1, resample_models=True)["_set"]
        need = len(models) if len(models) <= 4 else len(models) - 1
        res["across"] = {}
        for t, label, side, rd in TESTS:
            vals = np.array([est[m][t] for m in models])
            if not np.isfinite(vals).all():
                continue
            lo, hi = C.ci(wide[t])
            same = int(max((vals > 0).sum(), (vals < 0).sum()))
            res["across"][t] = {"mean": float(vals.mean()), "ci_models": (lo, hi), "positive": int((vals > 0).sum()),
                                "n": len(models), "general": bool(same >= need and (lo > 0 or hi < 0))}
        res["mixed"] = {name: mixed_contrast(data, models, rd, kinds) for name, rd, kinds in
                        (("HA", "reply", ("relax", "filler")), ("HC", "reply", ("relax", "calm")),
                         ("HB", "probe", ("relax", "filler")))}
    return res


def mixed_contrast(data: dict, models, reading: str, kinds: tuple[str, str]) -> dict:
    """Drop ~ four cell means (kind x induction), random intercepts for model and narrative or passage,
    and a random slope across models of the contrast being tested. REML, Wald interval."""
    from scipy.optimize import minimize

    from analysis.mixed import _reml
    y, cell, mod, cue, w = [], [], [], [], []
    for m in models:
        for i, k in enumerate(kinds):
            for j, s in enumerate(("story", "neutral")):
                X = data[m][f"{k}_{s}"]["X"][reading]
                n = len(X)
                y += list(X[:, 1] - X[:, 2]); cell += [2 * i + j] * n; mod += [m] * n
                cue += list(data[m][f"{k}_{s}"]["cue"]); w += [(1 if i == 0 else -1) * (1 if j == 0 else -1)] * n
    y, cell, mod, cue, w = np.array(y), np.array(cell), np.array(mod), np.array(cue), np.array(w, float)
    Xf = (cell[:, None] == np.arange(4)[None, :]).astype(float)
    Zm = (mod[:, None] == np.array(sorted(set(mod)))[None, :]).astype(float)
    Zc = (cue[:, None] == np.array(sorted(set(cue)))[None, :]).astype(float)
    ZZ = [Zm @ Zm.T, Zc @ Zc.T, (Zm * w[:, None]) @ (Zm * w[:, None]).T]
    best = None
    for start in (0.01, 0.1, 1.0):
        try:
            opt = minimize(lambda t: _reml(t, y, Xf, ZZ)[0], np.full(3, start), method="L-BFGS-B", bounds=[(0, None)] * 3)
        except np.linalg.LinAlgError:
            continue
        if best is None or opt.fun < best.fun - 1e-9:
            best = opt
    if best is None:
        return {"ok": False}
    _, beta, s2, Ainv = _reml(np.maximum(best.x, 0), y, Xf, ZZ)
    L = np.array([1.0, -1.0, -1.0, 1.0])
    est, se = float(L @ beta), float(np.sqrt(L @ (s2 * Ainv) @ L))
    zval = est / se if se > 0 else np.nan
    return {"ok": True, "estimate": est, "se": se, "ci": (est - 1.959964 * se, est + 1.959964 * se),
            "p_two_sided": float(math.erfc(abs(zval) / math.sqrt(2))) if np.isfinite(zval) else np.nan,
            "variances": {"model": float(best.x[0] * s2), "narrative or passage": float(best.x[1] * s2),
                          "contrast across models": float(best.x[2] * s2), "residual": float(s2)}, "n": int(len(y))}


# ---------------------------------------------------------------- report

def report(data: dict, info: dict, models, n_boot: int, batch: str) -> tuple[str, dict]:
    f, fci, sh = C.f, C.fci, C.short
    res = analyse(data, models, n_boot)
    md = ["# Study 2 confirmatory analysis", "",
          f"Batch {batch}. Code version: {C.code_version()}. {info['n_rows']} conversations read, "
          f"{len(info['excluded'])} excluded. Bootstrap draws: {n_boot}, seed {BOOT_SEED}. Units: SD of the reading.", ""]
    md += [f"- Excluded: {sh(e['model'])}, {e['arm']}, `{e['cell_id']}`: {e['reason']}" for e in info["excluded"]]
    not_judged = [sh(m) for m in models if not info["judged"].get(m)]
    if not_judged:
        md += ["", f"**No judge scores for: {', '.join(not_judged)}. HE is not computed for them** "
               "(run analysis.study2_judge first).", ""]
    for m in models:
        r = res["models"][m]
        tag = " (primary set)" if m == PRIMARY else ""
        md += [f"## {sh(m)}{tag}", "", f"Layer {info['layers'][m]}. Probe-turn rise after a narrative: "
               f"{f(r['probe_rise'][0])} {fci(r['probe_rise'][1:])}; HB is {'inside' if r['p1'] else 'outside'} the corrected family.", "",
               "| Test | Estimate | 95% interval | p | p after correction | Confirmed |", "| --- | --- | --- | --- | --- | --- |"]
        for t, _, _, _ in TESTS:
            x = r["tests"][t]
            conf = ("**yes**" if x.get("confirmed") else "no") if x["in_family"] else "not in family"
            md.append(f"| {t} {x['label']} ({x['side']}) | {f(x['estimate'])} | {fci(x['ci'])} | {f(x['p'], 4)} | "
                      f"{f(x.get('p_bh'), 4)} | {conf} |")
        md += ["", "| Regulation kind | Drop after a narrative | Drop after a neutral passage | Specific (difference) | "
               "Specific, probe turn | Specific, questionnaire (points) |", "| --- | --- | --- | --- | --- | --- |"]
        e = r["extras"]
        for k in KINDS:
            md.append(f"| {k} | {f(e[f'drop_reply_{k}_story']['estimate'])} | {f(e[f'drop_reply_{k}_neutral']['estimate'])} | "
                      f"{f(e[f'specific_reply_{k}']['estimate'])} {fci(e[f'specific_reply_{k}']['ci'])} | "
                      f"{f(e[f'specific_probe_{k}']['estimate'])} {fci(e[f'specific_probe_{k}']['ci'])} | "
                      f"{f(e[f'specific_stai_{k}']['estimate'], 1)} {fci(e[f'specific_stai_{k}']['ci'], 1)} |")
        md += ["", f"Secondary, whole-reply readout: HA {f(e['A_reply_all']['estimate'])} {fci(e['A_reply_all']['ci'])}, "
               f"HC {f(e['C_reply_all']['estimate'])} {fci(e['C_reply_all']['ci'])}. Rise after a narrative, reply reading: "
               f"{f(e['rise_reply']['estimate'])}.", ""]
    if "across" in res:
        md += ["## Across models", "", "| Test | Mean of models | Interval resampling models too | Models with a positive sign | General |",
               "| --- | --- | --- | --- | --- |"]
        for t, x in res["across"].items():
            md.append(f"| {t} | {f(x['mean'])} | {fci(x['ci_models'])} | {x['positive']} of {x['n']} | {'**yes**' if x['general'] else 'no'} |")
        md += ["", "Mixed-effects regression (second analysis; random slope of the contrast across models):", "",
               "| Contrast | Estimate | Standard error | 95% interval (Wald) | p (two-sided) | Variance of the contrast across models |",
               "| --- | --- | --- | --- | --- | --- |"]
        for t, x in res["mixed"].items():
            md.append(f"| {t} | {f(x['estimate'])} | {f(x['se'])} | {fci(x['ci'])} | {f(x['p_two_sided'], 4)} | "
                      f"{x['variances']['contrast across models']:.3f} |" if x.get("ok") else f"| {t} | did not fit | | | | |")
        md.append("")
    return "\n".join(md), res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--batch", choices=("A", "all"), default="A")
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--gate0-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--models", nargs="*", default=None, help="for tests on fake runs")
    args = ap.parse_args(argv)
    if not args.run.name.startswith("fake") and not args.unblind:
        print("This is the confirmatory run. Opening its results is a deliberate step: commit the analysis code, "
              "log the commit in TRACKING.md, then add --unblind.")
        return 2
    models = tuple(args.models) if args.models else (BATCH_A if args.batch == "A" else BATCH_A + BATCH_B)
    data, info = load(args.run, args.gate0_root, models)
    problems = incomplete(info, models)
    if problems:
        print("Not analysed: the batch is not complete, and results are opened once per batch.\n  " + "\n  ".join(problems[:12]))
        return 3
    md, res = report(data, info, models, args.boot, args.batch)
    out = args.run / f"analysis-batch-{args.batch}"
    out.mkdir(exist_ok=True)
    (out / "study2.md").write_text(md + "\n")
    (out / "study2.json").write_text(json.dumps(res, indent=2, default=float))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
