"""Study 2: the grid, the texts, the probe-turn fork and the pilot report, on the fake backend."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from harness import run as runner
from harness.sources import norm
from tests.test_harness import PoleBackend, needs_benzion

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "study2.yaml"
PILOT = ROOT / "configs" / "study2_pilot.yaml"


def test_grid_has_ten_cells_of_sixty():
    cfg = runner.load_config(CFG)
    cells = runner.build_cells(cfg, ["m"])
    per = Counter(c["arm"] for c in cells)
    assert len(per) == 10 and set(per.values()) == {60} and len(cells) == 600
    assert len({c["cell_id"] for c in cells}) == 600
    assert min(c["seed"] for c in cells) >= 200          # Study 1 used 100-105, pilots 0 and 1
    for arm in per:                                      # three turns, each measured, none after regulation
        steps = runner.arm_steps(cfg, arm)
        assert [s["measure"] for s in steps] == ["baseline", "post_induction", "post_regulation"]
        assert steps[0]["user"] == "benzion_trauma:neutral"
    story = [c for c in cells if c["arm"] == "filler_story"]
    assert Counter(c["factors"]["filler_cue"] for c in story) == {
        "benzion_relax:vacuum": 20, "literal:filler_2": 20, "literal:filler_3": 20}


def test_new_texts_are_the_intended_kind_and_length():
    t = {k: norm(v) for k, v in runner.load_config(CFG)["texts"].items()}
    for k in ("filler_2", "filler_3", "calm_1", "calm_2", "calm_3"):
        assert 117 <= len(t[k].split()) <= 202, k        # range of Ben-Zion's three relaxation scripts
    for k in ("neutral_2", "neutral_3"):
        assert 110 <= len(t[k].split()) <= 140, k
    for k in ("calm_1", "calm_2", "calm_3"):             # descriptions, not exercises: no address to the reader
        low = f" {t[k].lower()} "
        assert not any(w in low for w in (" you ", " your ", "breathe", "breath", "imagine", "relax", "calm"))
    assert "report" not in t["voice_instruction"] and "voice" in t["voice_instruction"]
    assert "{t}" in t["judge_prompt"]


def test_pilot_reuses_the_study_texts_on_a_pilot_seed():
    main, pilot = runner.load_config(CFG), runner.load_config(PILOT)
    assert pilot["texts"] == main["texts"] and pilot["decoding"] == main["decoding"]
    cells = runner.build_cells(pilot, ["m"])
    assert Counter(c["arm"] for c in cells) == {"relax_story": 15, "filler_story": 15,
                                                "relax_neutral": 6, "filler_neutral": 6}
    assert {c["seed"] for c in cells} == {0}
    assert runner.protocol_hash(pilot) != runner.protocol_hash(main)
    assert main.get("log_scores") is False


@needs_benzion
def test_probe_turn_is_saved_and_never_enters_the_conversation(tmp_path, capsys):
    from validation import gate0
    model = "pole-fake"
    gate0.collect(model, gate0.Store(tmp_path / f"gate0-{model}", model), "fake", backend=PoleBackend(model))
    out = tmp_path / "runs"
    assert runner.main([str(PILOT), "--backend", "fake", "--models", model, "--cooldown-sec", "0",
                        "--out", str(out)]) == 0
    run = out / "fake-study2-pilot"
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 42
    question = norm(runner.load_config(PILOT)["texts"]["probe_question"])
    for r in rows:
        assert len(r["transcript"]) == 7                                  # system + three turns
        assert all(question not in m["content"] for m in r["transcript"])
        z = np.load(run / r["acts_file"])
        for c in ("baseline", "post_induction", "post_regulation"):
            assert z[f"{c}__probe"].shape == z[f"{c}__first64"].shape and np.isfinite(z[f"{c}__probe"]).all()
            assert r["checkpoints"][c]["probe_turn"]["reply"]
    from analysis import study2_pilot as P
    md, verdicts = P.report(run, tmp_path)
    assert "P1" in md and set(verdicts) == {model}
    assert P.main([str(tmp_path / "study2")]) == 2                        # refuses a confirmatory run


def test_confirmatory_log_hides_scores(tmp_path, capsys):
    pytest.importorskip("yaml")
    try:
        from harness.sources import BenZion
        BenZion()
    except FileNotFoundError:
        pytest.skip("reference texts not present")
    assert runner.main([str(CFG), "--backend", "fake", "--models", "fake-m", "--limit", "2",
                        "--cooldown-sec", "0", "--out", str(tmp_path)]) == 0
    log = (tmp_path / "fake-study2" / "run.log").read_text()
    assert "scores not logged" in log and "stai=" not in log
