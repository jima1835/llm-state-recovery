"""The mixed-effects regression, checked on data where the right answer is known."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from analysis import confirmatory as C
from analysis import mixed as M
from tests.test_confirmatory import CUES, MODELS, synthetic

PLANTED = dict(r_int={"relaxation": 0.6, "neutral_filler": 0.1, "instruction_only": 0.1},
               r_rep={"relaxation": 0.6, "neutral_filler": 0.1, "instruction_only": 0.8},
               ret={"relaxation": 2.2, "neutral_filler": 2.6, "instruction_only": 2.4,
                    "no_induction": 2.0, "truncation": 2.0})
WANT = {"T1": 0.5, "T2": 0.5, "T3": 0.8, "T4": 0.7, "T5": 0.2, "T6": -0.2}


def with_seeds(data, n_seeds=6):
    for arms in data.values():
        for a in arms.values():
            a["seed"] = 100 + (np.arange(len(a["cue"])) // 5) % n_seeds
    return data


def test_planted_effects_are_recovered():
    res = M.analyse_set(with_seeds(synthetic(**PLANTED)), MODELS)
    for t, v in WANT.items():
        r = res["tests"][t]
        assert r["estimate"] == pytest.approx(v, abs=0.08), t
        assert r["ci"][0] <= v <= r["ci"][1], t
        assert r["p_bh"] < 0.05, t
    assert all(v["ok"] for v in res["fits"].values())
    assert res["fits"]["A recovery, internal"]["random"] == ["model", "narrative", "seed"]


def test_agrees_with_the_bootstrap_estimate():
    data = with_seeds(synthetic(**PLANTED, seed=5))
    mixed, boot = M.analyse_set(data, MODELS), C.analyse_set(data, MODELS, 300)
    for t in WANT:
        assert mixed["tests"][t]["estimate"] == pytest.approx(boot["tests"][t]["estimate"], abs=0.05), t


def test_no_effect_is_not_significant():
    flat = {a: 0.3 for a in C.INDUCED}
    data = with_seeds(synthetic(r_int=flat, r_rep={**flat, "instruction_only": 0.0},
                                ret={a: 2.0 for a in C.ARMS}, seed=3))
    res = M.analyse_set(data, MODELS)
    for t in ("T1", "T2", "T3", "T5", "T6"):
        assert res["tests"][t]["p_bh"] > 0.05, t
    assert res["tests"]["T4"]["estimate"] < 0 and res["tests"]["T4"]["p"] > 0.9  # wrong direction


def test_false_positive_rate_under_no_effect():
    """Across many null data sets the one-sided test of T1 rejects about 5% of the time."""
    flat = {a: 0.3 for a in C.INDUCED}
    ps = [M.analyse_set(with_seeds(synthetic(flat, flat, {a: 2.0 for a in C.ARMS}, seed=s, models=("m/a", "m/b"))),
                        ("m/a", "m/b"))["tests"]["T1"]["p"] for s in range(60)]
    assert 0.0 <= np.mean(np.array(ps) < 0.05) <= 0.15


def test_one_model_drops_the_model_term():
    res = M.analyse_set(with_seeds(synthetic(**PLANTED, models=("m/a",))), ("m/a",))
    assert all("model" not in v["random"] and v["ok"] for v in res["fits"].values())
    assert res["tests"]["T1"]["estimate"] == pytest.approx(0.5, abs=0.12)


def test_a_narrative_shift_goes_to_the_narrative_variance():
    """A narrative that shifts every arm's recovery equally leaves the arm contrast alone and shows up
    as narrative variance."""
    data = with_seeds(synthetic(**PLANTED, seed=2))
    for arms in data.values():
        for a in arms.values():
            odd = a["cue"] == CUES[4]
            a["X"]["internal"][odd, 2] -= 1.0
    res = M.analyse_set(data, MODELS)
    assert res["fits"]["A recovery, internal"]["variances"]["narrative"] > 0.02
    assert res["tests"]["T1"]["estimate"] == pytest.approx(0.5, abs=0.08)


def test_report_section_renders():
    data = with_seeds(synthetic(**PLANTED))
    md = "\n".join(M.section("Extended set", M.analyse_set(data, MODELS), C.analyse_set(data, MODELS, 200)))
    assert "T6" in md and "Bootstrap estimate" in md and "narrative" in md


def test_fit_matches_statsmodels():
    """The REML fit written here against statsmodels MixedLM, on data with clear model, narrative and
    seed effects (away from zero variances, where MixedLM converges)."""
    pytest.importorskip("statsmodels")
    import pandas as pd
    import statsmodels.formula.api as smf
    rng = np.random.default_rng(11)
    data = with_seeds(synthetic(**PLANTED, seed=11, models=tuple(f"m/{i}" for i in range(7))))
    df = M.frames(data, list(data))
    for name, sd in (("model", 0.4), ("narrative", 0.3), ("seed", 0.25)):
        shift = {lv: rng.normal(0, sd) for lv in sorted(set(df[name]))}
        df["y_int"] = df["y_int"] + np.array([shift[v] for v in df[name]])
    d = M.subset(df, np.isin(df["arm"], C.INDUCED))
    cols = [f"a_{a}" for a in C.INDUCED]
    ours = M.fit(d, "y_int", cols)
    pdf = pd.DataFrame({"y": d["y_int"], "gm": d["model"], "gn": d["narrative"], "gs": d["seed"],
                        "grp": 1, **{c: d[c] for c in cols}})
    ref = smf.mixedlm("y ~ 0 + " + " + ".join(cols), pdf, groups="grp", re_formula="0",
                      vc_formula={"model": "0 + gm", "narrative": "0 + gn", "seed": "0 + gs"}).fit(reml=True)
    assert ours["params"] == pytest.approx(np.asarray(ref.params)[:3], abs=2e-3)
    assert np.sqrt(np.diag(ours["cov"])) == pytest.approx(np.asarray(ref.bse)[:3], rel=0.05)
    assert ours["variances"]["residual"] == pytest.approx(float(ref.scale), rel=0.02)
    theirs = dict(zip(ref.model.exog_vc.names, ref.vcomp))
    for k in M.RANDOM:
        assert ours["variances"][k] == pytest.approx(float(theirs[k]), rel=0.05), k
