"""Pipeline tests with the deterministic FakeBackend (no GPU needed).

    .venv/bin/python -m pytest -q tests
"""
import json
from pathlib import Path

import numpy as np
import pytest

from harness import run as runner
from harness.backends import FakeBackend
from harness.measures import stai_on_fork
from harness.sources import BenZion, Questionnaire

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "pilot_benzion.yaml"
HAVE_BENZION = (ROOT / "third_party" / "gpt-trauma-induction" / "src" / "prompts.py").exists()
needs_benzion = pytest.mark.skipif(not HAVE_BENZION, reason="third_party Ben-Zion repo not present")


class SpyBackend(FakeBackend):
    """Records every message list it is given, to check forks never leak into the conversation."""

    def __init__(self):
        super().__init__("spy")
        self.seen = []

    def generate(self, messages, *a, **k):
        self.seen.append(("generate", [dict(m) for m in messages]))
        return super().generate(messages, *a, **k)

    def option_probs(self, messages, options):
        self.seen.append(("option_probs", [dict(m) for m in messages]))
        return super().option_probs(messages, options)


TOY_Q = Questionnaire("Preamble.", tuple(f"Item {i}" for i in range(1, 21)),
                      ("almost never", "occasionally", "often", "almost always"),
                      (1, 2, 5, 8, 10, 11, 15, 16, 19, 20))


# ---------------------------------------------------------------- scoring

class FixedBackend(FakeBackend):
    """Always puts all probability on the option *number* that shows a given label."""

    def __init__(self, label):
        super().__init__("fixed"); self.label = label

    def option_probs(self, messages, options):
        text = messages[-1]["content"]
        for line in text.splitlines():
            if line.startswith("Option ") and line.endswith(f": {self.label}."):
                n = int(line.split()[1].rstrip(":"))
                p = np.zeros(len(options)); p[n - 1] = 1.0
                return p, 1.0
        raise AssertionError("label not shown")


def test_stai_scoring_handles_shuffling_and_reverse_items():
    # "almost always" on every item: 10 normal items score 4, 10 reversed items score 1 -> 50
    r = stai_on_fork(FixedBackend("almost always"), [], TOY_Q, seed=3)
    assert r["total_argmax"] == 10 * 4 + 10 * 1 == r["total_expected"]
    # "almost never" everywhere -> 10*1 + 10*4 = 50 as well; check per-item direction instead
    r2 = stai_on_fork(FixedBackend("almost never"), [], TOY_Q, seed=3)
    assert [x["argmax"] for x in r2["items"]][:3] == [4, 4, 1]  # items 1, 2 reversed; 3 not
    # shuffling differs across items but the score is invariant to the seed
    assert stai_on_fork(FixedBackend("often"), [], TOY_Q, seed=99)["total_argmax"] == \
           stai_on_fork(FixedBackend("often"), [], TOY_Q, seed=1)["total_argmax"]


def test_option_order_is_actually_shuffled():
    b = SpyBackend()
    stai_on_fork(b, [], TOY_Q, seed=0)
    first_lines = {m[-1]["content"].split("Option ")[1][:1] for kind, m in b.seen}
    assert len(first_lines) > 1


# ---------------------------------------------------------------- sources

@needs_benzion
def test_benzion_texts_and_stai_load():
    bz = BenZion()
    assert len(bz.trauma("military", "brief").split()) > 100
    assert "  " not in bz.relaxation("generic", "brief")
    q = bz.stai()
    assert len(q.items) == 20 and len(q.labels) == 4


# ---------------------------------------------------------------- grid

def test_grid_counts_and_ids_are_stable():
    cfg = runner.load_config(CFG)
    cells = runner.build_cells(cfg, ["m1"])
    by_arm = {}
    for c in cells:
        by_arm[c["arm"]] = by_arm.get(c["arm"], 0) + 1
    n_t, n_r, n_s = (len(cfg["factors"]["trauma_cue"]), len(cfg["factors"]["relax_cue"]), len(cfg["seeds"]))
    assert by_arm["relaxation"] == n_t * n_r * n_s
    assert by_arm["neutral_filler"] == by_arm["instruction_only"] == by_arm["no_induction"] == n_t * n_s
    assert by_arm["truncation"] == n_t * n_r * n_s
    assert len({c["cell_id"] for c in cells}) == len(cells)
    assert [c["cell_id"] for c in runner.build_cells(cfg, ["m1"])] == [c["cell_id"] for c in cells]
    for c in cells:
        if "return_cue" in c["factors"]:
            assert c["factors"]["return_cue"] != c["factors"]["trauma_cue"]


# ---------------------------------------------------------------- one cell end to end

@needs_benzion
def test_cell_forks_never_leak_into_conversation():
    cfg = runner.load_config(CFG)
    bz = BenZion()
    cell = next(c for c in runner.build_cells(cfg, ["spy"]) if c["arm"] == "relaxation")
    b = SpyBackend()
    row, acts = runner.run_cell(cell, cfg, b, bz, bz.stai(), layers=[0, 1, 2])

    main_turns = [m for kind, m in b.seen if kind == "generate" and m[-1]["content"] != cfg["texts"]["vas_question"].strip()]
    # each main-conversation generate call sees system + alternating user/assistant, no questionnaire text
    for msgs in main_turns:
        assert not any("Option 1:" in m["content"] or "how anxious" in m["content"] for m in msgs)
    transcript = row["transcript"]
    assert [m["role"] for m in transcript] == ["system"] + ["user", "assistant"] * 3
    assert not any("Option" in m["content"] for m in transcript)
    # main conversation uses the conversational system prompt; every fork uses Ben-Zion's verbatim
    assert "numeric" not in transcript[0]["content"]
    assert all(m[0]["content"].endswith("Only reply with numeric values") for k, m in b.seen if k == "option_probs")
    # forks carry the full conversation so far, unchanged
    fork_bodies = [m[1:-1] for k, m in b.seen if k == "option_probs"]
    assert all(body == transcript[1:len(body) + 1] for body in fork_bodies)
    # 3 checkpoints, each with 20 STAI forks; acts have one row per saved layer
    assert set(row["checkpoints"]) == {"post_induction", "post_regulation", "post_return"}
    assert sum(1 for k, _ in b.seen if k == "option_probs") == 3 * 20
    assert all(a.shape == (3, b.d_model) and a.dtype == np.float32 for a in acts.values())


@needs_benzion
def test_runner_writes_resumes_and_guards_protocol(tmp_path):
    out = tmp_path / "runs"
    assert runner.main([str(CFG), "--backend", "fake", "--models", "fake-a", "--limit", "4", "--out", str(out)]) == 0
    run_dir = out / "fake-pilot-benzion-v2"
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for r in rows:
        npz = np.load(run_dir / r["acts_file"])
        assert set(npz.files) - {"layers"} == set(r["checkpoints"])
    # resume: the same 4 are skipped, 3 new ones run
    assert runner.main([str(CFG), "--backend", "fake", "--models", "fake-a", "--limit", "3", "--out", str(out)]) == 0
    ids = [json.loads(l)["cell_id"] for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(ids) == 7 == len(set(ids))
    # changing the protocol must refuse to resume into the same run
    cfg_text = CFG.read_text().replace("max_tokens: 120", "max_tokens: 121")
    changed = tmp_path / "pilot_benzion.yaml"; changed.write_text(cfg_text)
    assert runner.main([str(changed), "--backend", "fake", "--models", "fake-a", "--limit", "1", "--out", str(out)]) == 2


class HugeBackend(FakeBackend):
    def reply_residuals(self, gen, layers):
        return np.full((len(layers), self.d_model), 5.0e4 * 3, dtype=np.float32)  # Gemma-3-sized values


@needs_benzion
def test_large_residuals_are_saved_without_overflow():
    cfg = runner.load_config(CFG)
    bz = BenZion()
    cell = runner.build_cells(cfg, ["huge"])[0]
    row, acts = runner.run_cell(cell, cfg, HugeBackend("huge"), bz, bz.stai(), layers=[0, 1])
    assert all(np.isfinite(a).all() for a in acts.values())
    assert not any("nonfinite_acts" in cp for cp in row["checkpoints"].values())
