"""The Study 2 analysis, checked on data where the right answer is known."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from analysis import study2 as S
from analysis import study2_judge as J
from harness import run as runner
from tests.test_harness import PoleBackend, needs_benzion

ROOT = Path(__file__).resolve().parents[1]
CUES = ("military", "disaster", "interpersonal", "accident", "ambush")
DROPS = {"relax": (2.0, 1.0), "filler": (0.8, 0.5), "calm": (1.2, 0.9), "voice": (1.5, 1.0), "report": (0.6, 0.2)}
MODELS = ("m/a", "m/b", "m/c", "m/d")


def synthetic(drops=DROPS, models=MODELS, seed=0, noise=0.4, words_explain_all=False, shift=None):
    """Planted drops per kind: (after a narrative, after a neutral passage). The judge score tracks the
    'wording' part of the reading; unless words_explain_all, the story-specific part is not in it."""
    rng, data = np.random.default_rng(seed), {}
    for mi, m in enumerate(models):
        data[m] = {}
        for k, (ds, dn) in drops.items():
            for s in ("story", "neutral"):
                n = S.N_PER_ARM
                story = s == "story"
                cue = np.array([CUES[i % 5] for i in range(n)]) if story else np.array(["neutral_2", "neutral_3"] * (n // 2))
                base = rng.normal(-1, noise, n)
                ind = base + (3.0 if story else 0.2) + rng.normal(0, noise, n)
                extra = (shift or {}).get(m, 0.0) if (k == "relax" and story) else 0.0
                wording = dn                                  # what calm wording alone does
                specific = (ds - dn + extra) if story else 0.0
                reg = ind - wording - specific + rng.normal(0, noise, n)
                X = np.stack([base, ind, reg], 1)
                jud = 2.0 + 0.5 * np.stack([base, ind, ind - wording - (specific if words_explain_all else 0.0)], 1) \
                    + rng.normal(0, 0.05, (n, 3))
                data[m][f"{k}_{s}"] = {"cue": cue, "X": {"reply": X, "reply_all": X, "probe": 0.3 * X + rng.normal(0, 0.1, (n, 3)),
                                                         "stai": 40 + 5 * X, "judge": jud}}
        S.add_residual(data[m])
    return data


def test_planted_effects_are_recovered_per_model():
    res = S.analyse(synthetic(), MODELS, 1200)
    for m in MODELS:
        t = res["models"][m]["tests"]
        assert t["HA"]["estimate"] == pytest.approx(0.7, abs=0.25) and t["HA"]["confirmed"]
        assert t["HC"]["estimate"] == pytest.approx(0.7, abs=0.25) and t["HC"]["confirmed"]
        assert t["HD"]["estimate"] == pytest.approx(-0.5, abs=0.2) and t["HD"]["ci"][1] < 0
        assert t["HE"]["estimate"] > 0.3 and t["HE"]["confirmed"]      # the specific part is not in the words
        assert t["HB"]["estimate"] == pytest.approx(0.21, abs=0.12)
    assert res["across"]["HA"]["general"] and res["across"]["HA"]["positive"] == 4
    assert res["mixed"]["HA"]["ok"] and res["mixed"]["HA"]["estimate"] == pytest.approx(0.7, abs=0.15)


def test_wording_only_effect_is_not_confirmed():
    """Calm wording lowers the reading by the same amount with or without a narrative: HA must fail."""
    flat = {k: (v[1], v[1]) for k, v in DROPS.items()}
    res = S.analyse(synthetic(flat, seed=4), MODELS, 1200)
    assert not any(res["models"][m]["tests"]["HA"].get("confirmed") for m in MODELS)
    assert not res["across"]["HA"]["general"]


def test_he_fails_when_the_words_explain_the_whole_effect():
    res = S.analyse(synthetic(words_explain_all=True, seed=2), ("m/a",), 1000)
    t = res["models"]["m/a"]["tests"]
    assert t["HA"]["confirmed"] and not t["HE"].get("confirmed")
    assert abs(t["HE"]["estimate"]) < 0.25


def test_models_that_disagree_are_not_called_general():
    data = synthetic(shift={"m/c": -1.6, "m/d": -1.6}, seed=5)       # two models reversed
    res = S.analyse(data, MODELS, 1000)
    assert res["across"]["HA"]["positive"] == 2 and not res["across"]["HA"]["general"]
    assert res["mixed"]["HA"]["variances"]["contrast across models"] > 0.05


def test_hb_joins_the_family_only_when_the_probe_turn_registers_the_story():
    data = synthetic(seed=1, models=("m/a",))
    res = S.analyse(data, ("m/a",), 600)["models"]["m/a"]
    assert res["probe_rise"][0] > 0.5 and res["p1"] and res["tests"]["HB"]["in_family"]
    for d in data["m/a"].values():
        d["X"]["probe"] = d["X"]["probe"] * 0.3                       # probe turn now barely moves
    res = S.analyse(data, ("m/a",), 600)["models"]["m/a"]
    assert not res["p1"] and not res["tests"]["HB"]["in_family"] and "p_bh" not in res["tests"]["HB"]


def test_real_runs_need_unblind_and_a_complete_batch(tmp_path, capsys):
    (tmp_path / "study2").mkdir()
    assert S.main([str(tmp_path / "study2")]) == 2
    assert "--unblind" in capsys.readouterr().out


@needs_benzion
def test_end_to_end_on_a_fake_run_with_the_judge(tmp_path, capsys):
    from validation import gate0
    model = "pole-fake"
    gate0.collect(model, gate0.Store(tmp_path / f"gate0-{model}", model), "fake", backend=PoleBackend(model))
    out = tmp_path / "runs"
    args = [str(ROOT / "configs" / "study2.yaml"), "--backend", "fake", "--models", model, "--cooldown-sec", "0", "--out", str(out)]
    assert runner.main(args + ["--limit", "50"]) == 0
    run = out / "fake-study2"
    assert S.main([str(run), "--models", model, "--gate0-root", str(tmp_path), "--boot", "100"]) == 3   # incomplete
    assert runner.main(args) == 0
    assert J.main([str(run), "--backend", "fake"]) == 0
    assert J.main([str(run), "--backend", "fake"]) == 0                                                  # resumes, adds nothing
    judged = [json.loads(l) for l in (run / "judge.jsonl").read_text().splitlines()]
    assert len(judged) == 600 and all(set(j["scores"]) == set(S.CP) and 1 <= j["scores"]["baseline"] <= 4 for j in judged)
    assert S.main([str(run), "--models", model, "--gate0-root", str(tmp_path), "--boot", "150"]) == 0
    md = (run / "analysis-batch-A" / "study2.md").read_text()
    assert "HA" in md and "HE" in md and "No judge scores" not in md and "Probe-turn rise" in md
