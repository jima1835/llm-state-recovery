"""The registered mixed-effects regression: a second analysis next to the bootstrap.

Stage 1, section 6: "mixed-effects regression with model, narrative and seed as random effects. Each
channel is scaled to its own induction effect." The bootstrap in analysis/confirmatory.py stays the
primary statistic and alone decides what is confirmed. Written and tested on synthetic data before
any confirmatory result was opened.

Outcomes, one row per conversation i of model m (the truncation arm is left out, as in the bootstrap)
    E[m, channel] = mean(post_induction - baseline) over the model's three induced arms
    y_int = (post_induction - post_regulation) / E[m, internal]      recovery, internal reading
    y_rep = (post_induction - post_regulation) / E[m, self-report]   recovery, self-report
    z     = post_return (internal) / E[m, internal]                  level after the second narrative

Three regressions, each with crossed random intercepts for model, narrative and seed
    A  y_int ~ arm (three induced arms)          T1 = relaxation - neutral_filler
                                                 T2 = relaxation - instruction_only
    B  instruction-only arm, intercept only      T3 = mean of y_rep
                                                 T4 = mean of (y_rep - y_int), paired by conversation
    C  z ~ arm (three induced arms, no_induction) T5 = mean of the induced arms - no_induction
                                                 T6 = relaxation - neutral_filler
T3 and T4 are two fits (one outcome each). With one model in the set, the model term is dropped.

Fitted by REML in this file (numpy, with scipy for the optimiser): Var(y) = s2 * (I + sum_k theta_k Z_k Z_k'),
variances kept at or above zero. tests/test_mixed.py checks the fit against statsmodels MixedLM. Tests are
Wald z tests on the fixed-effect contrast: one-sided where the hypothesis is directional, two-sided for T6.
With 5 narratives, 6 seeds and at most 7 models the variance estimates are rough, and a z test is
somewhat optimistic; this is one more reason the bootstrap decides.

How this differs from the bootstrap estimate, by construction: the arm means are divided by the
model's pooled induction effect, not the arm's own; conversations are pooled across models with a
random model intercept, not averaged model by model. The two should agree closely when arms have
similar induction effects. The random terms are intercepts only (as registered), so the standard
errors do not allow an arm effect to differ between models or narratives; the narrative bootstrap does.
"""
from __future__ import annotations

import math

import numpy as np

from analysis import confirmatory as C

RANDOM = ("model", "narrative", "seed")
RETURN_ARMS = C.INDUCED + ("no_induction",)


def available() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def frames(data: dict, models, internal: str = "internal", report: str = "stai") -> dict:
    """-> columns (arrays of equal length), one entry per conversation, with the scaled outcomes."""
    cols: dict = {k: [] for k in ("model", "arm", "narrative", "seed", "y_int", "y_rep", "z")}
    for m in models:
        ind_i = np.concatenate([data[m][a]["X"][internal] for a in C.INDUCED])
        ind_s = np.concatenate([data[m][a]["X"][report] for a in C.INDUCED])
        e_int = float(np.mean(ind_i[:, 1] - ind_i[:, 0]))
        e_rep = float(np.mean(ind_s[:, 1] - ind_s[:, 0]))
        for arm in RETURN_ARMS:
            a = data[m][arm]
            I, S = a["X"][internal], a["X"][report]
            n = len(I)
            cols["model"] += [m] * n
            cols["arm"] += [arm] * n
            cols["narrative"] += [str(c) for c in a["cue"]]
            cols["seed"] += [str(s) for s in a["seed"]]
            cols["y_int"] += list((I[:, 1] - I[:, 2]) / e_int)
            cols["y_rep"] += list((S[:, 1] - S[:, 2]) / e_rep)
            cols["z"] += list(I[:, 3] / e_int)
    df = {k: np.array(v) for k, v in cols.items()}
    df["y_gap"] = df["y_rep"] - df["y_int"]
    df["one"] = np.ones(len(df["arm"]))
    for arm in RETURN_ARMS:
        df[f"a_{arm}"] = (df["arm"] == arm).astype(float)
    return df


def subset(df: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in df.items()}


def _reml(theta: np.ndarray, y: np.ndarray, X: np.ndarray, ZZ: list[np.ndarray]):
    """Profiled REML criterion (-2 log likelihood up to a constant) at variance ratios theta, where
    Var(y) = s2 * (I + sum_k theta_k Z_k Z_k'). -> (criterion, beta, s2, inverse of X' V^-1 X)."""
    n, p = X.shape
    V = np.eye(n)
    for t, zz in zip(theta, ZZ):
        V += t * zz
    L = np.linalg.cholesky(V)
    Xw, yw = np.linalg.solve(L, X), np.linalg.solve(L, y)
    A = Xw.T @ Xw
    beta = np.linalg.solve(A, Xw.T @ yw)
    r = yw - Xw @ beta
    s2 = float(r @ r) / (n - p)
    crit = (n - p) * np.log(s2) + 2 * np.log(np.diag(L)).sum() + np.linalg.slogdet(A)[1]
    return crit, beta, s2, np.linalg.inv(A)


def fit(df: dict, outcome: str, columns: list[str]) -> dict:
    """REML fit of outcome on the given 0/1 columns (no separate intercept), with crossed random
    intercepts for every factor in RANDOM that has more than one level. Variances are kept at or above
    zero; the best of several starting points is used. -> coefficients, covariance, variances."""
    from scipy.optimize import minimize
    y = np.asarray(df[outcome], float)
    X = np.column_stack([df[c] for c in columns]).astype(float)
    random = [r for r in RANDOM if len(set(df[r])) > 1]
    ZZ = []
    for r in random:
        levels = sorted(set(df[r]))
        Z = (df[r][:, None] == np.array(levels)[None, :]).astype(float)
        ZZ.append(Z @ Z.T)
    notes, best = [], None
    if random:
        for start in (0.01, 0.1, 1.0):
            try:
                opt = minimize(lambda t: _reml(t, y, X, ZZ)[0], np.full(len(random), start), method="L-BFGS-B",
                               bounds=[(0.0, None)] * len(random))
            except np.linalg.LinAlgError as exc:
                notes.append(f"start {start}: {type(exc).__name__}")
                continue
            if best is None or opt.fun < best.fun - 1e-9:
                best = opt
        if best is None:
            return {"ok": False, "notes": notes, "random": random}
        theta = np.maximum(best.x, 0.0)
        if not best.success:
            notes.append("optimiser did not report convergence")
    else:
        theta = np.zeros(0)
    try:
        _, beta, s2, Ainv = _reml(theta, y, X, ZZ)
    except np.linalg.LinAlgError as exc:
        return {"ok": False, "notes": notes + [type(exc).__name__], "random": random}
    variances = {r: float(t * s2) for r, t in zip(random, theta)}
    variances["residual"] = s2
    return {"ok": True, "columns": columns, "params": beta, "cov": s2 * Ainv, "variances": variances,
            "notes": notes, "random": random, "n": int(len(y))}


def contrast(fitted: dict, weights: dict, side: str) -> dict:
    if not fitted["ok"]:
        return {"estimate": np.nan, "se": np.nan, "ci": (np.nan, np.nan), "p": np.nan}
    L = np.array([weights.get(c, 0.0) for c in fitted["columns"]])
    est = float(L @ fitted["params"])
    se = float(np.sqrt(L @ fitted["cov"] @ L))
    z = est / se if se > 0 else np.nan
    upper = 0.5 * math.erfc(z / math.sqrt(2)) if np.isfinite(z) else np.nan  # P(Z >= z)
    p = upper if side == "greater" else min(1.0, 2 * min(upper, 1 - upper))
    return {"estimate": est, "se": se, "ci": (est - 1.959964 * se, est + 1.959964 * se), "p": float(p)}


def analyse_set(data: dict, models, internal: str = "internal", report: str = "stai") -> dict:
    """The six registered tests from the mixed-effects regressions, for one set of models."""
    df = frames(data, models, internal, report)
    induced = subset(df, np.isin(df["arm"], C.INDUCED))
    instr = subset(df, df["arm"] == "instruction_only")
    cols_a = [f"a_{a}" for a in C.INDUCED]
    cols_c = [f"a_{a}" for a in RETURN_ARMS]
    fa = fit(induced, "y_int", cols_a)
    fb_rep = fit(instr, "y_rep", ["one"])
    fb_gap = fit(instr, "y_gap", ["one"])
    fc = fit(df, "z", cols_c)
    spec = {"T1": (fa, {"a_relaxation": 1, "a_neutral_filler": -1}),
            "T2": (fa, {"a_relaxation": 1, "a_instruction_only": -1}),
            "T3": (fb_rep, {"one": 1}),
            "T4": (fb_gap, {"one": 1}),
            "T5": (fc, {**{f"a_{a}": 1 / 3 for a in C.INDUCED}, "a_no_induction": -1}),
            "T6": (fc, {"a_relaxation": 1, "a_neutral_filler": -1})}
    tests = {}
    for t, label, side in C.TESTS:
        fitted, w = spec[t]
        tests[t] = {"label": label, "side": side, **contrast(fitted, w, side)}
    ps = [tests[t]["p"] for t, _, _ in C.TESTS]
    if all(np.isfinite(ps)):
        for (t, _, _), q in zip(C.TESTS, C.benjamini_hochberg(ps)):
            tests[t]["p_bh"] = q
    fits = {"A recovery, internal": fa, "B self-report, instruction only": fb_rep,
            "B self-report minus internal, instruction only": fb_gap, "C return level": fc}
    return {"models": list(models), "tests": tests,
            "fits": {k: {"ok": v["ok"], "random": v["random"], "n": v.get("n"), "variances": v.get("variances", {}),
                         "notes": v["notes"]} for k, v in fits.items()}}


def section(name: str, res: dict, boot: dict | None = None) -> list[str]:
    """Markdown for one set. boot: the bootstrap result of the same set, shown alongside."""
    f, fci = C.f, C.fci
    md = [f"### {name}", "",
          "| Test | Mixed-effects estimate | Standard error | 95% interval (Wald) | p | p after correction | "
          "Bootstrap estimate |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for t, _, _ in C.TESTS:
        r = res["tests"][t]
        b = f(boot["tests"][t]["estimate"]) if boot else "–"
        md.append(f"| {t} {r['label']} | {f(r['estimate'])} | {f(r['se'])} | {fci(r['ci'])} | {f(r['p'], 4)} | "
                  f"{f(r.get('p_bh'), 4)} | {b} |")
    md += ["", "| Regression | Rows | Random intercepts | Variances (random terms, then residual) | Notes |",
           "| --- | --- | --- | --- | --- |"]
    for k, v in res["fits"].items():
        var = ", ".join(f"{n} {x:.3f}" for n, x in v["variances"].items()) or "–"
        note = "; ".join(v["notes"]) if v["notes"] else ("" if v["ok"] else "did not fit")
        md.append(f"| {k} | {v['n'] or '–'} | {', '.join(v['random']) or 'none'} | {var} | {note or 'none'} |")
    return md + [""]


INTRO = ("Registered as the model (Stage 1, section 6): random intercepts for model, narrative and seed; each "
         "channel scaled to the model's own induction effect. It is a second analysis: the bootstrap above is the "
         "registered primary statistic and alone decides what is confirmed. Random terms are intercepts only, so "
         "these standard errors do not let an arm effect vary between models or narratives; the narrative "
         "bootstrap does. A variance of 0.000 means the term explained nothing beyond the others.")
