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
    # each checkpoint saves the whole-reply readout and the first-64-token readout
    assert set(acts) == {n for c in row["checkpoints"] for n in (c, f"{c}__first64")}
    assert all("timing_s" in cp for cp in row["checkpoints"].values())


@needs_benzion
def test_runner_writes_resumes_and_guards_protocol(tmp_path):
    out = tmp_path / "runs"
    assert runner.main([str(CFG), "--backend", "fake", "--models", "fake-a", "--limit", "4", "--out", str(out)]) == 0
    run_dir = out / "fake-pilot-benzion-v2"
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for r in rows:
        npz = np.load(run_dir / r["acts_file"])
        assert set(npz.files) - {"layers"} == {n for c in r["checkpoints"] for n in (c, f"{c}__first64")}
    # resume: the same 4 are skipped, 3 new ones run
    assert runner.main([str(CFG), "--backend", "fake", "--models", "fake-a", "--limit", "3", "--out", str(out)]) == 0
    ids = [json.loads(l)["cell_id"] for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(ids) == 7 == len(set(ids))
    # changing the protocol must refuse to resume into the same run
    cfg_text = CFG.read_text().replace("max_tokens: 120", "max_tokens: 121")
    changed = tmp_path / "pilot_benzion.yaml"; changed.write_text(cfg_text)
    assert runner.main([str(changed), "--backend", "fake", "--models", "fake-a", "--limit", "1", "--out", str(out)]) == 2


class HugeBackend(FakeBackend):
    def reply_residuals(self, gen, layers, first_k=64):
        a = np.full((len(layers), self.d_model), 5.0e4 * 3, dtype=np.float32)  # Gemma-3-sized values
        return {"all": a, "first": a.copy()}


@needs_benzion
def test_large_residuals_are_saved_without_overflow():
    cfg = runner.load_config(CFG)
    bz = BenZion()
    cell = runner.build_cells(cfg, ["huge"])[0]
    row, acts = runner.run_cell(cell, cfg, HugeBackend("huge"), bz, bz.stai(), layers=[0, 1])
    assert all(np.isfinite(a).all() for a in acts.values())
    assert not any("nonfinite_acts" in cp for cp in row["checkpoints"].values())


# ---------------------------------------------------------------- prefix reuse plumbing

def test_common_prefix_len():
    from harness.backends import common_prefix_len
    assert common_prefix_len([[1, 2, 3, 4, 5], [1, 2, 3, 9, 9]]) == 3
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 3, 4]], min_suffix=2) == 2   # always leaves 2 tokens
    assert common_prefix_len([[7, 1], [8, 1]]) == 0


def test_option_probs_many_matches_one_by_one():
    b = FakeBackend("m")
    forks = [[{"role": "user", "content": f"q{i}"}] for i in range(5)]
    many = b.option_probs_many(forks, ["1", "2", "3", "4"])
    single = [b.option_probs(f, ["1", "2", "3", "4"]) for f in forks]
    assert all(np.allclose(x[0], y[0]) and x[1] == y[1] for x, y in zip(many, single))


def test_stai_uses_one_batched_call():
    class Counting(FakeBackend):
        many_calls = 0
        def option_probs_many(self, message_lists, options):
            Counting.many_calls += 1
            assert len(message_lists) == 20
            # every fork shares the conversation and differs only in the last (question) message
            assert all(m[:-1] == message_lists[0][:-1] for m in message_lists)
            return super().option_probs_many(message_lists, options)
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    r = stai_on_fork(Counting("c"), conv, TOY_Q, seed=5)
    assert Counting.many_calls == 1 and len(r["items"]) == 20


# ---------------------------------------------------------------- v2.0 protocol

V2 = ROOT / "configs" / "v2.0_gemma.yaml"


def test_battery_matches_separate_measures():
    from harness.measures import fork_battery, vas_on_fork
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    b = FakeBackend("m")
    out = fork_battery(b, conv, TOY_Q, "How anxious, 0-100?", seed=7)
    assert out["stai"] == stai_on_fork(b, conv, TOY_Q, seed=7)
    assert out["vas"] == vas_on_fork(b, conv, "How anxious, 0-100?", seed=7)
    assert fork_battery(b, conv, None, "How anxious, 0-100?", seed=7).keys() == {"vas"}
    assert fork_battery(b, conv, TOY_Q, None, seed=7).keys() == {"stai"}


def test_v2_arms_line_up_and_truncation_has_one_seed():
    cfg = runner.load_config(V2)
    names = {arm: [s.get("measure") for s in runner.arm_steps(cfg, arm)] for arm in cfg["arms"]}
    assert all(n == ["baseline", "post_induction", "post_regulation", "post_return"] for n in names.values())
    cells = runner.build_cells(cfg, ["m"])
    assert {c["seed"] for c in cells if c["arm"] == "truncation"} == {0}
    assert {c["seed"] for c in cells if c["arm"] == "relaxation"} == {0, 1}
    assert len(cells) == 75


@needs_benzion
def test_v2_cell_runs_end_to_end_with_baseline_first():
    cfg = runner.load_config(V2)
    bz = BenZion()
    for arm in cfg["arms"]:
        cell = next(c for c in runner.build_cells(cfg, ["spy"]) if c["arm"] == arm)
        row, acts = runner.run_cell(cell, cfg, SpyBackend(), bz, bz.stai(), layers=[0, 1])
        users = [m["content"] for m in row["transcript"] if m["role"] == "user"]
        assert users[0].startswith("The bicameral legislature") and len(users) == 4
        assert list(row["checkpoints"]) == ["baseline", "post_induction", "post_regulation", "post_return"]
        if arm in ("no_induction", "truncation"):
            assert users[1].startswith("Postal codes")  # sham induction, no trauma text before the return turn


def test_mlx_backend_slow_path_does_not_recurse():
    """With reuse off, MLXBackend must answer per item (regression: it used to call itself forever)."""
    from harness.backends import Generation, MLXBackend
    b = object.__new__(MLXBackend)          # no model load: only the routing logic is under test
    b.reuse_prefix = False
    calls = []
    b.option_probs = lambda m, o: (calls.append("probs"), (np.full(len(o), 1 / len(o)), 1.0))[1]
    b.generate = lambda m, n, t, s: (calls.append("gen"), Generation("50", [1], [2], "stop"))[1]
    forks = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
    answers, gen = b.battery(forks, ["1", "2"], gen_messages=forks[0])
    assert len(answers) == 3 and gen.text == "50" and calls == ["probs"] * 3 + ["gen"]
    assert len(b.option_probs_many(forks, ["1", "2"])) == 3


# ---------------------------------------------------------------- self-report analysis

def test_recovery_fraction_is_a_ratio_of_means():
    from analysis.self_report import recovery
    # baseline 50, induction 70, regulation 60 in every conversation -> half the effect recovered
    m = np.array([[50, 70, 60, 72]] * 4, dtype=float)
    assert recovery(m) == pytest.approx(0.5)
    # full recovery, none, and overshoot
    assert recovery(np.array([[50, 70, 50, 0.0]])) == pytest.approx(1.0)
    assert recovery(np.array([[50, 70, 70, 0.0]])) == pytest.approx(0.0)
    assert recovery(np.array([[50, 70, 40, 0.0]])) == pytest.approx(1.5)
    # a conversation with no induction effect does not blow up the estimate
    m2 = np.array([[50, 70, 60, 0], [60, 60, 55, 0]], dtype=float)
    assert recovery(m2) == pytest.approx((-10 - 5) / -20)


def test_cue_bootstrap_resamples_whole_narratives():
    from analysis.self_report import resample_index
    cues = np.array(["a"] * 3 + ["b"] * 3)
    idx = resample_index(cues, np.random.default_rng(0), "cue")
    assert len(idx) == 6 and set(cues[idx[:3]]) <= {"a", "b"} and len(set(cues[idx[:3]])) == 1


# ---------------------------------------------------------------- Gate 0 validation

class PoleBackend(FakeBackend):
    """Plants an 'anxious' signal on coordinate 0 of every layer when the conversation asks for anxiety."""

    def generate(self, messages, max_tokens, temperature, seed):
        gen = super().generate(messages, max_tokens, temperature, seed)
        text = " ".join(m["content"] for m in messages if m["role"] != "assistant")
        gen.meta["anx"] = float("intensely anxious" in text or "extremely anxious" in text)
        return gen

    def reply_residuals(self, gen, layers, first_k=64):
        r = super().reply_residuals(gen, layers, first_k)
        for k in r:
            r[k] = r[k].copy(); r[k][:, 0] += 4.0 * gen.meta["anx"]
        return r


@needs_benzion
def test_gate0_recovers_a_planted_direction(tmp_path):
    from validation import gate0, probe, texts
    store = gate0.Store(tmp_path / "g0", "pole-fake")
    gate0.collect("pole-fake", store, "fake", backend=PoleBackend("pole-fake"))
    md, js = gate0.report(store)
    c1 = js["criteria"]["1_heldout"]
    assert c1["pass"] and c1["mb"]["acc"] >= 0.9 and c1["permutation"]["p"] < 0.01
    # the neutral replies carry no signal, so they sit at the calm pole, about d/2 below zero
    assert js["criteria"]["2_neutral"]["mean_z"] < -1
    # turn-8 replies under the anxious system prompt still carry the signal
    assert js["criteria"]["6a_deep_heldout"]["pass"]
    # the topic controls do not contain the trigger phrase, so they don't move the reading
    assert js["criteria"]["5_topic"]["first64"]["other_person"]["shift"] < 1
    # stored metadata stays aligned with the arrays, and question sets are disjoint
    meta, A, F = store.load("pole_test")
    assert len(meta) == len(A) == 40 and {m["pole"] for m in meta} == {"anxious", "calm"}
    assert not set(texts.TRAIN) & set(texts.TEST) and not set(texts.SELECT) & set(texts.TEST)
    # re-running skips finished parts; a changed protocol is refused
    gate0.collect("pole-fake", store, "fake", backend=PoleBackend("pole-fake"))
    assert len(store.load("pole_test")[0]) == 40


def test_probe_variance_scaling_ignores_a_noisy_giant_coordinate():
    from validation import probe
    rng = np.random.default_rng(0)
    def pole(shift):
        X = rng.standard_normal((30, 8))
        X[:, 0] *= 1e4                 # Gemma-like outlier coordinate: huge and noisy, no signal
        X[:, 3] += shift               # the real signal lives in an ordinary coordinate
        return X
    Xa, Xc = pole(2.0), pole(0.0)
    mb, mean = probe.fit(Xa, Xc, 0, "mb"), probe.fit(Xa, Xc, 0, "mean")
    assert abs(mb.w[3]) > 0.9 and abs(mean.w[3]) < 0.5
    Ta, Tc = pole(2.0), pole(0.0)
    assert probe.accuracy(mb, Ta, Tc) > probe.accuracy(mean, Ta, Tc) + 0.15
