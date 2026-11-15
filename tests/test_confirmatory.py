"""The confirmatory analysis, checked on data where the right answer is known."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import confirmatory as C
from harness import run as runner
from tests.test_harness import PoleBackend, needs_benzion

ROOT = Path(__file__).resolve().parents[1]
CUES = ("military", "disaster", "interpersonal", "accident", "ambush")
MODELS = ("m/a", "m/b", "m/c")


def synthetic(r_int, r_rep, ret, seed=0, models=MODELS, noise=0.5):
    """Conversations with planted recovery fractions and return levels. r_int / r_rep: {arm: R};
    ret: {arm: internal level after the second narrative, above baseline}. Induction effect: 2 SD
    on the internal channel and 20 points on the questionnaire."""
    rng = np.random.default_rng(seed)
    data = {}
    for m in models:
        data[m] = {}
        for arm, n in C.EXPECTED_N.items():
            cue = np.array([CUES[i % 5] for i in range(n)])
            induced = arm in C.INDUCED
            base_i, base_s = rng.normal(0, noise, n), rng.normal(40, 5 * noise, n)
            ind_i = base_i + (2.0 if induced else 0.0) + rng.normal(0, noise, n)
            ind_s = base_s + (20.0 if induced else 0.0) + rng.normal(0, 5 * noise, n)
            reg_i = ind_i - r_int.get(arm, 0) * 2.0 + rng.normal(0, noise, n)
            reg_s = ind_s - r_rep.get(arm, 0) * 20.0 + rng.normal(0, 5 * noise, n)
            ret_i = base_i + ret[arm] + rng.normal(0, noise, n)
            I = np.stack([base_i, ind_i, reg_i, ret_i], 1)
            S = np.stack([base_s, ind_s, reg_s, ind_s], 1)
            data[m][arm] = {"cue": cue, "seed": np.zeros(n, int),
                            "X": {"internal": I, "internal_all": I, "internal_mean": I, "stai": S, "vas": S}}
    return data


def test_planted_effects_are_recovered_and_confirmed():
    data = synthetic(r_int={"relaxation": 0.6, "neutral_filler": 0.1, "instruction_only": 0.1},
                     r_rep={"relaxation": 0.6, "neutral_filler": 0.1, "instruction_only": 0.8},
                     ret={"relaxation": 2.2, "neutral_filler": 2.6, "instruction_only": 2.4,
                          "no_induction": 2.0, "truncation": 2.0})
    res = C.analyse_set(data, MODELS, 1500)
    want = {"T1": 0.5, "T2": 0.5, "T3": 0.8, "T4": 0.7, "T5": 0.2, "T6": -0.2}
    for t, v in want.items():
        r = res["tests"][t]
        assert r["estimate"] == pytest.approx(v, abs=0.08), t
        assert r["ci"][0] <= v <= r["ci"][1], t
        assert r["confirmed"], t


def test_no_effect_is_not_confirmed():
    flat = {a: 0.3 for a in C.INDUCED}
    data = synthetic(r_int=flat, r_rep={**flat, "instruction_only": 0.0},
                     ret={a: 2.0 for a in C.ARMS}, seed=3)
    res = C.analyse_set(data, MODELS, 1500)
    # T3 is planted at exactly zero and T4 at -0.3 (the wrong direction for the hypothesis)
    for t in ("T1", "T2", "T3", "T5", "T6"):
        r = res["tests"][t]
        assert r["ci"][0] <= 0 <= r["ci"][1] and not r["confirmed"], t
    assert res["tests"]["T4"]["estimate"] < 0 and not res["tests"]["T4"]["confirmed"]


def test_a_wrong_direction_effect_is_never_confirmed():
    data = synthetic(r_int={"relaxation": 0.1, "neutral_filler": 0.6, "instruction_only": 0.6},
                     r_rep={"instruction_only": -0.5},
                     ret={"relaxation": 2.0, "neutral_filler": 2.0, "instruction_only": 2.0,
                          "no_induction": 2.6, "truncation": 2.0})
    res = C.analyse_set(data, MODELS, 800)
    assert not any(res["tests"][t]["confirmed"] for t in ("T1", "T2", "T3", "T4", "T5"))


def test_every_model_counts_equally():
    """A set statistic is the mean of per-model statistics, not a pool of conversations."""
    data = synthetic({"relaxation": 0.6}, {}, {a: 2.0 for a in C.ARMS}, noise=0.0)
    other = synthetic({"relaxation": 0.2}, {}, {a: 2.0 for a in C.ARMS}, noise=0.0, models=("m/z",))
    data.update(other)
    per = {m: C.model_stats({arm: a["X"] for arm, a in data[m].items()}) for m in data}
    assert C.set_stats(per, list(data))["T1"] == pytest.approx((3 * 0.6 + 0.2) / 4)


def test_recovery_and_p_values_and_correction():
    assert C.recovery(np.array([[0.0, 2, 0, 0]])) == pytest.approx(1.0)
    assert C.recovery(np.array([[0.0, 2, 2, 0]])) == pytest.approx(0.0)
    d = np.array([-1.0] * 5 + [1.0] * 95)
    assert C.p_value(d, "greater") == pytest.approx(6 / 101)
    assert C.p_value(d, "two-sided") == pytest.approx(12 / 101)
    assert C.p_value(-d, "greater") > 0.9
    # textbook example: adjusted p-values are monotone and scaled by m / rank
    adj = C.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_narrative_bootstrap_is_wider_when_narratives_differ():
    """If one narrative behaves differently, resampling narratives must show more uncertainty than
    resampling conversations alone."""
    data = synthetic({"relaxation": 0.5, "neutral_filler": 0.1}, {}, {a: 2.0 for a in C.ARMS}, models=("m/a",))
    X = data["m/a"]["relaxation"]["X"]["internal"]
    odd = data["m/a"]["relaxation"]["cue"] == "ambush"
    X[odd, 2] = X[odd, 1] + 2.0  # this narrative gets worse after relaxation
    nar = C.bootstrap(data, ("m/a",), 600, "narrative")["T1"]
    conv = C.bootstrap(data, ("m/a",), 600, "conversation")["T1"]
    assert np.std(nar) > 1.5 * np.std(conv)


def test_cronbach_alpha_on_known_cases():
    rng = np.random.default_rng(0)
    trait = rng.normal(size=(400, 1))
    assert C.cronbach_alpha(trait + rng.normal(scale=0.3, size=(400, 20))) > 0.95   # items agree
    assert abs(C.cronbach_alpha(rng.normal(size=(400, 20)))) < 0.2                  # items unrelated


def test_real_runs_need_the_unblind_flag(tmp_path, capsys):
    (tmp_path / "wave1").mkdir()
    assert C.main([str(tmp_path / "wave1")]) == 2
    assert "--unblind" in capsys.readouterr().out


@needs_benzion
def test_loader_scores_saved_readings_with_the_gate0_probe(tmp_path):
    """End to end on fake runs: Gate 0 fits the reading, the runner saves conversations, and the loader
    scores each checkpoint at the Gate 0 layer with that probe."""
    from validation import gate0
    model = "pole-fake"
    store = gate0.Store(tmp_path / f"gate0-{model}", model)
    gate0.collect(model, store, "fake", backend=PoleBackend(model))
    out = tmp_path / "runs"
    cfg = ROOT / "configs" / "wave1.yaml"
    assert runner.main([str(cfg), "--backend", "fake", "--models", model,
                        "--cooldown-sec", "0", "--out", str(out)]) == 0
    run = out / "fake-wave1"
    data, info = C.load_runs([run], tmp_path, (model,))
    assert info["n_rows"] == 195 and not info["excluded"]
    assert info["counts"][model] == C.EXPECTED_N and not C.check_complete(info, (model,))
    assert set(info["seeds"]) <= set(range(100, 106))
    # one conversation, scored by hand
    row = json.loads((run / "results.jsonl").read_text().splitlines()[0])
    fr = gate0.fit_reading(store)
    z = np.load(run / row["acts_file"])
    by_hand = [float(fr["probes"]["mb"].z(z[f"{c}__first64"][fr["layer"]][None])[0]) for c in C.CHECKPOINTS]
    arm = data[model][row["arm"]]
    assert any(np.allclose(x, by_hand) for x in arm["X"]["internal"])
    assert all(np.isfinite(a["X"][ch]).all() for a in data[model].values() for ch in ("internal", "stai"))
    assert set(info["reliability"][model]) == {"stai_alpha", "readout_r"}
    md, _ = C.report(data, info, 200, sensitivity=True)
    assert "Cronbach" in md and "Sensitivity run" in md
    # a conversation with a damaged reading is excluded and listed, not silently dropped
    bad = dict(np.load(run / row["acts_file"]))
    bad["post_return__first64"] = np.full_like(bad["post_return__first64"], np.nan)
    np.savez(run / row["acts_file"], **bad)
    data2, info2 = C.load_runs([run], tmp_path, (model,))
    assert [e["cell_id"] for e in info2["excluded"]] == [row["cell_id"]]
    assert not C.check_complete(info2, (model,))  # analysed + excluded still adds up
    assert info2["excluded"][0]["reason"] == "non-finite activations"
