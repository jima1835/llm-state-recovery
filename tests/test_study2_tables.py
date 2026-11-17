"""The Study 2 tables carry the analysis output's numbers unchanged."""
import json

from analysis import study2_tables as T


def _fake(he=0.4):
    t = lambda est, conf=False, fam=True: {"estimate": est, "ci": [est - 0.5, est + 0.5], "p": 0.01,
                                           "in_family": fam, "p_bh": 0.02, "confirmed": conf}
    extras = {f"specific_{r}_{k}": {"estimate": 1.0 + i / 10} for r in ("reply", "probe")
              for i, (k, _) in enumerate(T.KINDS)}
    extras.update({"rise_reply": {"estimate": 3.61}, "rise_probe": {"estimate": 0.55}})
    models = {f"mlx-community/{m}": {"tests": {"HA": t(0.45, True), "HC": t(-0.84), "HD": t(-1.18, True),
                                                "HE": t(he), "HB": t(-0.21, fam=(i == 0))},
                                      "probe_rise": [0.55, 0.42, 0.71], "extras": extras}
              for i, m in enumerate(("gemma-3-4b-it-4bit", "Qwen3-14B-4bit"))}
    across = {k: {"mean": 0.1, "ci_models": [-0.2, 0.3], "positive": 1, "n": 2} for k in ("HA", "HC", "HD", "HE", "HB")}
    return {"models": models, "across": across}


def test_tables(tmp_path):
    run = tmp_path / "study2"
    for sub, he in (("analysis-batch-A", 0.4), ("robust-api-judge/analysis-batch-A", 0.17)):
        (run / sub).mkdir(parents=True)
        (run / sub / "study2.json").write_text(json.dumps(_fake(he)))
    out = tmp_path / "tables"
    assert T.main([str(run), "--batch", "A", "--out", str(out)]) == 0
    tests = (out / "study2_tests.tex").read_text()
    assert "Gemma 3 4B & \\textbf{0.45} & $-$0.84 & \\textbf{$-$1.18} & 0.40 & 0.17 & $-$0.21 & 0.55" in tests
    assert "Qwen3 14B" in tests and "$-$0.21*" in tests  # outside the family in the second model
    spec = (out / "study2_specific.tex").read_text()
    assert "Gemma 3 4B & 1.00 & 1.10 & 1.20 & 1.30 & 1.40 & 3.61" in spec
