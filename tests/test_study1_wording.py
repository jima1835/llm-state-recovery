"""The post hoc Study 1 wording check, on data where the right answer is known; no network."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from analysis import confirmatory as C
from analysis import study1_judge_api as JA
from analysis import study1_wording as W
from harness import run as runner
from tests.test_confirmatory import MODELS, synthetic
from tests.test_harness import PoleBackend, needs_benzion

ROOT = Path(__file__).resolve().parents[1]
R_INT = {"relaxation": 0.6, "neutral_filler": 0.1, "instruction_only": 0.1}
RET = {a: 2.0 for a in C.ARMS}


def with_judge(data, world, seed=1):
    """world "wording": the reading is a straight line in the judge's rating plus a little noise, so the
    words explain everything. world "state": the rating is unrelated to the reading."""
    rng = np.random.default_rng(seed)
    for arms in data.values():
        for d in arms.values():
            I = d["X"]["internal"]
            if world == "wording":
                J = 2.0 + 0.4 * I
                d["X"]["internal"] = 2.5 * (J - 2.0) + rng.normal(0, 0.05, I.shape)
            else:
                J = rng.uniform(1, 4, I.shape)
            d["X"]["judge"] = J
    return data


def test_wording_world_the_residual_contrast_vanishes():
    data = with_judge(synthetic(R_INT, {}, RET), "wording")
    fit = W.add_residual(data)
    assert all(v["r"] > 0.99 for v in fit.values())
    res = W.analyse_set(data, MODELS, 600)
    assert res["tests"]["D1_internal"]["estimate"] == pytest.approx(1.0, abs=0.15)
    assert res["tests"]["D1_internal"]["ci"][0] > 0
    for k in ("D1_resid", "D2_resid"):
        assert abs(res["tests"][k]["estimate"]) < 0.05 and res["tests"][k]["ci"][0] <= 0 <= res["tests"][k]["ci"][1]
    assert not res["ratio_ok"] and all(p["kept"] < W.KEEP_RULE for p in res["per_model"].values())


def test_state_world_the_residual_contrast_survives():
    data = with_judge(synthetic(R_INT, {}, RET), "state")
    W.add_residual(data)
    res = W.analyse_set(data, MODELS, 600)
    for key in ("D1", "D2"):
        raw, resid = res["tests"][f"{key}_internal"], res["tests"][f"{key}_resid"]
        assert resid["estimate"] == pytest.approx(raw["estimate"], abs=0.05) and resid["ci"][0] > 0
        assert resid["positive_models"] == len(MODELS)
    assert res["ratio_ok"] and res["tests"]["T1_resid"]["estimate"] == pytest.approx(0.5, abs=0.1)


def test_raw_reading_reproduces_the_registered_bootstrap():
    """Same seed and resampling order as the registered analysis: T1 and T2 on the raw reading must match it."""
    data = with_judge(synthetic(R_INT, {}, RET), "state")
    W.add_residual(data)
    mine, registered = W.analyse_set(data, MODELS, 300), C.analyse_set(data, MODELS, 300)
    for t in ("T1", "T2"):
        assert mine["tests"][f"{t}_internal"]["estimate"] == pytest.approx(registered["tests"][t]["estimate"])
        assert mine["tests"][f"{t}_internal"]["ci"] == pytest.approx(registered["tests"][t]["ci"])
        assert mine["tests"][f"{t}_internal"]["p"] == pytest.approx(registered["tests"][t]["p"])


class FakeClient:
    """Rates a reply by its length, so ratings vary and are reproducible."""
    def __init__(self):
        self.inputs = []
        self.responses = self

    def create(self, **kw):
        self.inputs.append(kw["input"])
        p3 = min(0.9, 0.1 + (len(kw["input"]) % 9) / 10)
        lp = [{"token": "2", "logprob": math.log(1 - p3)}, {"token": "3", "logprob": math.log(p3)}]
        return {"output_text": "2", "output": [{"type": "message", "content": [
            {"type": "output_text", "logprobs": [{"token": "2", "logprob": 0.0, "top_logprobs": lp}]}]}]}


@needs_benzion
def test_end_to_end_on_a_fake_run(tmp_path, capsys):
    from validation import gate0
    model = "pole-fake"
    gate0.collect(model, gate0.Store(tmp_path / f"gate0-{model}", model), "fake", backend=PoleBackend(model))
    out = tmp_path / "runs"
    assert runner.main([str(ROOT / "configs" / "wave1.yaml"), "--backend", "fake", "--models", model,
                        "--cooldown-sec", "0", "--out", str(out)]) == 0
    run = out / "fake-wave1"
    prompt_run = tmp_path / "study2"
    prompt_run.mkdir()
    (prompt_run / "protocol.json").write_text(json.dumps({"config": {"texts": {"judge_prompt": 'Text: "{t}"  How anxious? 1-4.'}}}))
    with pytest.raises(SystemExit):                       # no ratings yet
        W.main([str(run), "--sensitivity", "--gate0-root", str(tmp_path), "--boot", "50", "--out", str(tmp_path / "o")])
    fc = FakeClient()
    assert JA.main([str(run), "--prompt-run", str(prompt_run), "--limit", "20"], client=fc) == 0
    assert JA.main([str(run), "--prompt-run", str(prompt_run)], client=fc) == 0      # resumes
    rated = [json.loads(l) for l in (run / "judge_api.jsonl").read_text().splitlines()]
    assert len(rated) == 195 and len({r["cell_id"] for r in rated}) == 195 and len(fc.inputs) == 195 * 4
    assert all(set(r["scores"]) == set(C.CHECKPOINTS) and 2 <= r["scores"]["baseline"] <= 3 for r in rated)
    assert fc.inputs[0].startswith('Text: "') and fc.inputs[0].endswith('" How anxious? 1-4.')
    # one missing rating: that conversation is left out and counted
    rated[0]["scores"]["post_regulation"] = None
    (run / "judge_api.jsonl").write_text("\n".join(json.dumps(r) for r in rated) + "\n")
    assert W.main([str(run), "--sensitivity", "--gate0-root", str(tmp_path), "--boot", "50", "--out", str(tmp_path / "o")]) == 0
    md = (tmp_path / "o" / "study1_wording.md").read_text()
    js = json.loads((tmp_path / "o" / "study1_wording.json").read_text())
    assert js["quality"]["missing_conversations"] == 1 and js["quality"]["n_ratings"] == 195 * 4
    assert "post hoc" in md and "D1 drop" in md and "Judge rating by arm" in md
    assert set(js["sets"]["Sensitivity set"]["tests"]) >= {"D1_internal", "D1_resid", "D2_resid", "T1_resid"}
