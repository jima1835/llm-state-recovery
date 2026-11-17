"""Second-judge script (analysis/study2_judge_api.py) on a fake client; no network."""
import json
import math

from analysis import study2_judge_api as A


def lp(*pairs):
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def test_expected_option_and_mass():
    s, mass, pos = A.score_from_logprobs([{"token": "2", "logprob": math.log(0.6), "top_logprobs": lp(("2", 0.6), ("3", 0.3), ("x", 0.1))}])
    assert pos == 0 and abs(mass - 0.9) < 1e-9 and abs(s - (2 * 0.6 + 3 * 0.3) / 0.9) < 1e-9


def test_skips_a_leading_space_token_and_merges_spaced_digits():
    entries = [{"token": " ", "top_logprobs": lp((" ", 0.99))},
               {"token": "3", "top_logprobs": lp(("3", 0.5), (" 3", 0.2), ("4", 0.3))}]
    s, mass, pos = A.score_from_logprobs(entries)
    assert pos == 1 and abs(mass - 1.0) < 1e-9 and abs(s - 3.3) < 1e-9


def test_no_digit_gives_nan():
    s, mass, pos = A.score_from_logprobs([{"token": "I", "top_logprobs": lp(("I", 1.0))}])
    assert math.isnan(s) and mass == 0 and pos == -1


class FakeClient:
    def __init__(self):
        self.calls = []
        self.responses = self

    def create(self, **kw):
        self.calls.append(kw)
        return {"output_text": "2", "output": [{"type": "message", "content": [
            {"type": "output_text", "logprobs": [{"token": "2", "logprob": 0.0, "top_logprobs": lp(("2", 1.0))}]}]}]}


def test_end_to_end_on_a_fake_run(tmp_path):
    run = tmp_path / "fake-run"
    run.mkdir()
    (run / "protocol.json").write_text(json.dumps({"config": {"texts": {"judge_prompt": 'Text: "{t}"  Rate   1-4.'}}}))
    rows = [{"cell_id": f"c{i}", "model": "m", "transcript": [{"role": "user", "content": "u"}, {"role": "assistant", "content": f"r{i}a"},
             {"role": "user", "content": "u"}, {"role": "assistant", "content": f"r{i}b"}],
             "checkpoints": {"baseline": {"after_step": 0}, "post": {"after_step": 1}}} for i in range(3)]
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n{partial")
    fc = FakeClient()
    assert A.main([str(run), "--limit", "2", "--workers", "2"], client=fc) == 0
    assert A.main([str(run)], client=fc) == 0  # resumes: only the third conversation is new
    out = [json.loads(l) for l in (run / "judge_api.jsonl").read_text().splitlines()]
    assert sorted(o["cell_id"] for o in out) == ["c0", "c1", "c2"] and all(o["scores"]["baseline"] == 2.0 for o in out)
    assert len(fc.calls) == 6 and fc.calls[0]["input"] in {f'Text: "r{i}a" Rate 1-4.' for i in range(3)} | {f'Text: "r{i}b" Rate 1-4.' for i in range(3)}
    assert fc.calls[0]["reasoning"] == {"effort": "none"} and fc.calls[0]["include"] == ["message.output_text.logprobs"]


def test_robust_view_reproduces_the_registered_analysis_when_the_judges_agree(tmp_path):
    """With judge_api.jsonl identical to judge.jsonl, the wrapper must print exactly the registered report."""
    import shutil
    from pathlib import Path
    import pytest
    pytest.importorskip("scipy")
    from tests.test_harness import PoleBackend, needs_benzion  # noqa: F401
    from analysis import study2 as S, study2_judge as J, study2_he_robust as R
    from harness import run as runner
    from validation import gate0
    root = Path(__file__).resolve().parents[1]
    model = "pole-fake"
    gate0.collect(model, gate0.Store(tmp_path / f"gate0-{model}", model), "fake", backend=PoleBackend(model))
    out = tmp_path / "runs"
    assert runner.main([str(root / "configs" / "study2.yaml"), "--backend", "fake", "--models", model,
                        "--cooldown-sec", "0", "--out", str(out)]) == 0
    run = out / "fake-study2"
    assert J.main([str(run), "--backend", "fake"]) == 0
    common = ["--models", model, "--gate0-root", str(tmp_path), "--boot", "100"]
    assert S.main([str(run)] + common) == 0
    shutil.copy(run / "judge.jsonl", run / "judge_api.jsonl")
    assert R.main([str(run)] + common) == 2                      # refuses without --unblind (view is not "fake")
    assert R.main([str(run)] + common + ["--unblind"]) == 0
    a = (run / "analysis-batch-A" / "study2.md").read_text()
    b = (run / "robust-api-judge" / "analysis-batch-A" / "study2.md").read_text()
    assert a == b
