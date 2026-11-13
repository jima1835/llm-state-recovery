"""Gate 0: validate the anxious-versus-calm reading before any confirmatory run (HANDOFF.md 5.3).

Stages (scripts/gate0.sh runs collect -> steer -> judge -> report for you):

    .venv/bin/python -m validation.gate0 collect --model mlx-community/gemma-3-4b-it-4bit
    .venv/bin/python -m validation.gate0 steer   --model mlx-community/gemma-3-4b-it-4bit
    .venv/bin/python -m validation.gate0 judge   --model mlx-community/gemma-3-4b-it-4bit
    .venv/bin/python -m validation.gate0 report  --model mlx-community/gemma-3-4b-it-4bit
    .venv/bin/python -m validation.gate0 sheet   --model ...   # blind rating sheet for a person (Day 2)
    .venv/bin/python -m validation.gate0 unblind --model ...   # after rating: does rating rise with strength?

Output goes to runs/gate0-<model>/ (git-ignored). collect and steer need the GPU; report, sheet and
unblind are numpy only. Every stage can be re-run: finished parts are skipped, and a run refuses to
continue if validation/texts.py or SETTINGS changed (use --run-name for a new run).

What is collected (all from the model under test):
    pole_train / pole_select / pole_test   replies under the anxious and calm system prompts
    neutral_test                           replies with no pole (criterion 2)
    regen_test                             pole given as a user instruction, three levels (criterion 4)
    deep_pole / deep_regen                 the same, at turn 8 after 7 neutral turns (criterion 6)
    topic                                  experiment-format conversations: neutral turn, then the
                                           induction itself or a topic control (criterion 5)
    steer.jsonl                            steered replies to neutral writing prompts (criterion 3)

Criterion 3 judge: a different model family by default (Llama 3.1 8B), which reads pairs of steered
replies blind to the strength and says which sounds more anxious. Criterion 7 (8-bit / bf16) means
running collect + report on those model files too. Criterion 8 (logic set) is not part of the current
protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harness.backends import make_backend
from harness.run import versions
from harness.sources import ROOT, BenZion

from . import probe as P
from . import texts as T

SETTINGS = {
    "pole_max_tokens": 64,          # Martorell & Bianchi: greedy, <= 64 new tokens
    "first_k": 64,
    "topic_max_tokens": 120,        # topic controls use the experiment's decoding
    "topic_temperature": 0.7,
    "topic_seeds": [0, 1],
    "steer_alphas": [0.005, 0.01, 0.02, 0.04, 0.08, 0.16],  # x median per-token residual norm at the layer
    "steer_max_tokens": 80,
}
CUES = ("military", "disaster", "interpersonal", "accident", "ambush")
DEFAULT_JUDGE = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

# Pass thresholds. All final: c3_coherent, c4_pair and c5_ratio (provisional until then) were locked at
# their existing values on Nov 12 2026, before any Gate 0 report was read (see TRACKING.md Decisions).
# Criterion 5 passes on the point ratio; its confidence interval is reported alongside.
GATE = {"c1_acc": 0.85, "c1_d": 1.5, "c2_abs_z": 0.5, "c3_acc": 0.75, "c3_coherent": 0.5,
        "c4_pair": 0.85, "c5_ratio": 0.5}
PROVISIONAL: set[str] = set()

FEELING_WORDS = ("anxious", "anxiety", "nervous", "worried", "worry", "tense", "fear", "afraid", "scared",
                 "panic", "uneasy", "calm", "relaxed", "peaceful", "at ease", "serene", "content")

JUDGE_PAIR = ("Two short texts were written in reply to the same request: \"{prompt}\"\n\nText A:\n{a}\n\n"
              "Text B:\n{b}\n\nWhich text sounds more anxious, tense or worried? Reply with Option A or Option B.")
JUDGE_COHERENT = ("Here is a short text written in reply to the request \"{prompt}\":\n\n{t}\n\n"
                  "Is this fluent, coherent writing? Option 1: yes. Option 2: no. Reply with Option 1 or Option 2.")


# ---------------------------------------------------------------- storage

def protocol_hash() -> str:
    blob = Path(T.__file__).read_text() + json.dumps(SETTINGS, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_dir_for(model: str, run_name: str | None, fake: bool) -> Path:
    name = run_name or f"gate0-{model.rstrip('/').split('/')[-1]}"
    if fake and not name.startswith("fake"):
        name = "fake-" + name
    return ROOT / "runs" / name


class Store:
    def __init__(self, d: Path, model: str | None = None):
        self.d = d
        (d / "acts").mkdir(parents=True, exist_ok=True)
        stamp = d / "protocol.json"
        h = protocol_hash()
        if stamp.exists():
            old = json.loads(stamp.read_text())
            if old["protocol_hash"] != h:
                raise SystemExit(f"{d.name}: texts or settings changed since this run started "
                                 f"({old['protocol_hash']} -> {h}). Use --run-name for a new run.")
            self.model = old["model"]
        else:
            if model is None:
                raise SystemExit(f"No Gate 0 run at {d}. Run collect first.")
            stamp.write_text(json.dumps({"protocol_hash": h, "model": model, "settings": SETTINGS, "gate": GATE,
                                         "created": datetime.now(timezone.utc).isoformat(), "env": versions()},
                                        indent=2))
            self.model = model

    def has(self, cond: str) -> bool:
        return (self.d / "acts" / f"{cond}.npz").exists()

    def save(self, cond: str, meta: list[dict], acts_all: list, acts_first: list) -> None:
        # meta goes inside the same file as the arrays, so a crash can never leave them out of step
        np.savez_compressed(self.d / "acts" / f"{cond}.npz", all=np.stack(acts_all).astype(np.float32),
                            first=np.stack(acts_first).astype(np.float32), meta=np.array(json.dumps(meta)))
        with open(self.d / "texts.jsonl", "a") as f:  # readable copy of the replies
            for m in meta:
                f.write(json.dumps({"cond": cond, **{k: v for k, v in m.items() if k != "messages"}}) + "\n")

    def load(self, cond: str):
        z = np.load(self.d / "acts" / f"{cond}.npz")
        return json.loads(str(z["meta"])), z["all"], z["first"]

    def jsonl(self, name: str) -> list[dict]:
        p = self.d / name
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ---------------------------------------------------------------- collection

def sys_msg(text: str) -> list[dict]:
    return [{"role": "system", "content": text}] if text else []


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def reply(backend, msgs, layers, max_tokens, temperature=0.0, seed=0):
    gen = backend.generate(msgs, max_tokens, temperature, seed)
    return gen, backend.reply_residuals(gen, layers, first_k=SETTINGS["first_k"])


def run_jobs(backend, layers, jobs, label):
    meta, A, F = [], [], []
    t0 = time.time()
    for k, (m, msgs, max_tokens, temp, seed) in enumerate(jobs, 1):
        gen, r = reply(backend, msgs, layers, max_tokens, temp, seed)
        meta.append({**m, "messages": msgs, "text": gen.text, "finish": gen.finish_reason,
                     "reply_tokens": len(gen.reply_ids)})
        A.append(r["all"]); F.append(r["first"])
        if k % 20 == 0 or k == len(jobs):
            say(f"  {label}: {k}/{len(jobs)} ({time.time() - t0:.0f}s)")
    return meta, A, F


def greedy(n=None):
    return (n or SETTINGS["pole_max_tokens"], 0.0, 0)


def jobs_poles(questions, set_name, poles=("anxious", "calm")):
    return [({"set": set_name, "pole": pole, "qi": i, "question": q}, sys_msg(T.pole_system(pole)) + [user(q)], *greedy())
            for pole in poles for i, q in enumerate(questions)]


def jobs_regen(prefix, extra=None):
    return [({"level": lv, "rank": r, "qi": i, "question": q, **(extra or {})},
             prefix + [user(f"{instr}\n\n{q}")], *greedy())
            for r, (lv, instr) in enumerate(T.REGEN_LEVELS) for i, q in enumerate(T.TEST)]


def filler_conversation(backend, system: str) -> list[dict]:
    """Seven neutral turns answered greedily by the model, as context for the turn-8 checks."""
    conv = sys_msg(system)
    for q in T.FILLER:
        gen = backend.generate(conv + [user(q)], SETTINGS["pole_max_tokens"], 0.0, 0)
        conv = conv + [user(q), {"role": "assistant", "content": gen.text}]
    return conv


def collect_topic(backend, layers, bz: BenZion):
    """Experiment format: Ben-Zion conversational system prompt, the neutral passage, then turn 2.
    One baseline reply per (narrative, seed) is shared by every kind of turn 2."""
    meta, A, F = [], [], []
    neutral = bz.trauma("neutral", "brief")
    sysm = sys_msg(T.pole_system(None))
    mt, temp = SETTINGS["topic_max_tokens"], SETTINGS["topic_temperature"]

    def add(m, msgs, gen, r):
        meta.append({**m, "messages": msgs, "text": gen.text, "finish": gen.finish_reason,
                     "reply_tokens": len(gen.reply_ids)})
        A.append(r["all"]); F.append(r["first"])

    for cue in CUES:
        narrative = bz.trauma(cue, "brief")
        turn2 = {"own": narrative,
                 "other_person": T.TOPIC_TEMPLATES["other_person"].format(narrative=narrative),
                 "analysis_task": T.TOPIC_TEMPLATES["analysis_task"].format(narrative=narrative),
                 "failure_log": T.TOPIC_TEMPLATES["failure_log"].format(failure_log=T.FAILURE_LOG)}
        for s in SETTINGS["topic_seeds"]:
            seed = int(hashlib.sha256(f"{cue}|{s}".encode()).hexdigest()[:8], 16)
            msgs = sysm + [user(neutral)]
            g, r = reply(backend, msgs, layers, mt, temp, seed)
            add({"kind": "baseline", "cue": cue, "seed": s}, msgs, g, r)
            conv = msgs + [{"role": "assistant", "content": g.text}]
            for kind, text in turn2.items():
                m2 = conv + [user(text)]
                g2, r2 = reply(backend, m2, layers, mt, temp, seed + 1)
                add({"kind": kind, "cue": cue, "seed": s}, m2, g2, r2)
        say(f"  topic: {cue} done")
    return meta, A, F


def collect(model: str, store: Store, backend_kind: str, backend=None) -> None:
    backend = backend or make_backend(backend_kind, model, {"qwen3_thinking": False})
    layers = list(range(backend.n_layers))
    say(f"collect: {model} ({backend.n_layers} layers, d={backend.d_model}) -> {store.d}")
    plan = [
        ("pole_train", lambda: jobs_poles(T.TRAIN, "train")),
        ("pole_select", lambda: jobs_poles(T.SELECT, "select")),
        ("pole_test", lambda: jobs_poles(T.TEST, "test")),
        ("neutral_test", lambda: [({"set": "test", "pole": None, "qi": i, "question": q},
                                   sys_msg(T.pole_system(None)) + [user(q)], *greedy()) for i, q in enumerate(T.TEST)]),
        ("regen_test", lambda: jobs_regen(sys_msg(T.pole_system(None)))),
        ("deep_pole", lambda: [({"pole": pole, "qi": i, "question": q, "turn": 8}, conv + [user(q)], *greedy())
                               for pole in ("anxious", "calm")
                               for conv in [filler_conversation(backend, T.pole_system(pole))]
                               for i, q in enumerate(T.TEST)]),
        ("deep_regen", lambda: jobs_regen(filler_conversation(backend, T.pole_system(None)), {"turn": 8})),
    ]
    for cond, make_jobs in plan:
        if store.has(cond):
            say(f"  {cond}: already done"); continue
        store.save(cond, *run_jobs(backend, layers, make_jobs(), cond))
    if not store.has("topic"):
        store.save("topic", *collect_topic(backend, layers, BenZion()))
    say("collect: done")


# ---------------------------------------------------------------- the reading

def split_poles(meta, X, key="pole", a="anxious", c="calm"):
    ia = [i for i, m in enumerate(meta) if m[key] == a]
    ic = [i for i, m in enumerate(meta) if m[key] == c]
    return X[ia], X[ic]


def fit_reading(store: Store) -> dict:
    mtr, Xtr, _ = store.load("pole_train")
    mse, Xse, _ = store.load("pole_select")
    tra, trc = split_poles(mtr, Xtr)
    sea, sec = split_poles(mse, Xse)
    n_layers = Xtr.shape[1]
    layers = P.band(n_layers)
    scans = {k: P.layer_scan(tra, trc, sea, sec, layers, k) for k in ("mb", "mean")}
    best = max(scans["mb"], key=lambda r: r["d"])["layer"]
    probes = {k: P.calibrate(P.fit(tra[:, best], trc[:, best], best, k), sea[:, best], sec[:, best])
              for k in ("mb", "mean")}
    return {"layer": best, "n_layers": n_layers, "band": layers, "scans": scans, "probes": probes,
            "steer_vec": probes["mean"].w, "train": (tra, trc), "select": (sea, sec)}


# ---------------------------------------------------------------- steering

def token_norm(backend, store: Store, layer: int):
    """Median per-token residual norm at `layer` over some training replies, and the activation dtype."""
    if not hasattr(backend, "lens"):
        return 1.0, None
    mx = backend.mx
    meta, _, _ = store.load("pole_train")
    norms, dtype = [], None
    for m in meta[:8] + meta[-8:]:
        ids = backend._ids(m["messages"], add_generation_prompt=True)
        rid = list(backend.tok.encode(m["text"], add_special_tokens=False))
        with backend.lens.capture(layers=[layer]) as cache:
            backend.lens.forward(ids + rid)
        h = cache[layer][0, len(ids):].astype(mx.float32)
        n = mx.sqrt((h * h).sum(axis=-1))
        mx.eval(n)
        norms.extend(np.array(n).tolist())
        dtype = cache[layer].dtype
    return float(np.median(norms)), dtype


def steered_generate(backend, msgs, layer, vec, scale, dtype, max_tokens):
    if not hasattr(backend, "lens"):  # FakeBackend: no real steering, but distinct outputs per strength
        return backend.generate(msgs, max_tokens, 0.0, int(round(scale * 1e6)))
    if scale == 0:
        return backend.generate(msgs, max_tokens, 0.0, 0)
    mx = backend.mx
    with backend.lens.steer(layer, mx.array(vec.astype(np.float32)).astype(dtype), float(scale)):
        return backend.generate(msgs, max_tokens, 0.0, 0)


def steer(model: str, store: Store, backend_kind: str) -> None:
    fr = fit_reading(store)
    layer, vec = fr["layer"], fr["steer_vec"]
    backend = make_backend(backend_kind, model, {"qwen3_thinking": False})
    norm, dtype = token_norm(backend, store, layer)
    alphas = [0.0] + [s * a for a in SETTINGS["steer_alphas"] for s in (1, -1)]
    (store.d / "steer_meta.json").write_text(json.dumps({"layer": layer, "token_norm": norm, "alphas": alphas}))
    done = {(r["pi"], r["alpha"]) for r in store.jsonl("steer.jsonl")}
    say(f"steer: layer {layer}, median token norm {norm:.1f}, {len(alphas)} strengths x {len(T.STEER_PROMPTS)} prompts")
    t0 = time.time()
    with open(store.d / "steer.jsonl", "a") as f:
        for pi, prompt in enumerate(T.STEER_PROMPTS):
            msgs = sys_msg(T.pole_system(None)) + [user(prompt)]
            for a in alphas:
                if (pi, a) in done:
                    continue
                gen = steered_generate(backend, msgs, layer, vec, a * norm, dtype, SETTINGS["steer_max_tokens"])
                f.write(json.dumps({"pi": pi, "prompt": prompt, "alpha": a, "layer": layer, "text": gen.text,
                                    "finish": gen.finish_reason}) + "\n"); f.flush()
            say(f"  steer: prompt {pi + 1}/{len(T.STEER_PROMPTS)} ({time.time() - t0:.0f}s)")
    say("steer: done")


def judge(store: Store, judge_model: str, backend_kind: str) -> None:
    rows = store.jsonl("steer.jsonl")
    if not rows:
        raise SystemExit("No steered replies yet. Run steer first.")
    by = {(r["pi"], r["alpha"]): r for r in rows}
    jb = make_backend(backend_kind, judge_model, {"qwen3_thinking": False})
    pos = sorted({r["alpha"] for r in rows if r["alpha"] > 0})
    say(f"judge: {judge_model} on {len(rows)} steered replies")
    pairs, coh = [], []
    for pi in sorted({r["pi"] for r in rows}):
        prompt = by[(pi, 0.0)]["prompt"]
        for a in pos:
            for hi, lo, kind in ((a, -a, "plus_vs_minus"), (a, 0.0, "plus_vs_zero"), (0.0, -a, "zero_vs_minus")):
                rng = np.random.default_rng(int(hashlib.sha256(f"{pi}|{a}|{kind}".encode()).hexdigest()[:8], 16))
                flip = bool(rng.random() < 0.5)
                th, tl = by[(pi, hi)]["text"], by[(pi, lo)]["text"]
                ta, tb = (tl, th) if flip else (th, tl)
                p, mass = jb.option_probs([user(JUDGE_PAIR.format(prompt=prompt, a=ta, b=tb))], ["A", "B"])
                pairs.append({"pi": pi, "alpha": a, "kind": kind, "p_more_positive": float(p[1] if flip else p[0]),
                              "mass": mass, "flip": flip})
        say(f"  judge: prompt {pi + 1}")
    for r in rows:
        p, mass = jb.option_probs([user(JUDGE_COHERENT.format(prompt=r["prompt"], t=r["text"]))], ["1", "2"])
        coh.append({"pi": r["pi"], "alpha": r["alpha"], "p_coherent": float(p[0]), "mass": mass})
    (store.d / "judge.json").write_text(json.dumps({"judge_model": judge_model, "pairs": pairs, "coherence": coh}))
    say("judge: done")


# ---------------------------------------------------------------- blind rating by a person (Day 2)

def sheet(store: Store, alphas: list[float], n_prompts: int) -> None:
    rows = [r for r in store.jsonl("steer.jsonl")
            if r["pi"] < n_prompts and any(np.isclose(r["alpha"], a) for a in alphas)]
    if not rows:
        raise SystemExit("No steered replies match. Run steer first, or check --alphas.")
    order = np.random.default_rng(secrets.randbits(64)).permutation(len(rows))
    key = {}
    with open(store.d / "blind_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "request", "text", "anxiety_1_to_5"])
        for k, i in enumerate(order, 1):
            w.writerow([k, rows[i]["prompt"], rows[i]["text"], ""])
            key[k] = {"pi": rows[i]["pi"], "alpha": rows[i]["alpha"]}
    (store.d / ".blind_key.json").write_text(json.dumps(key))
    print(f"Wrote {store.d / 'blind_sheet.csv'} ({len(rows)} texts, shuffled; strengths are hidden in "
          f".blind_key.json — don't open it). Rate each text 1 (not at all anxious) to 5 (very anxious) "
          f"in the last column, save, then run the unblind stage.")


def unblind(store: Store) -> str:
    key = {int(k): v for k, v in json.loads((store.d / ".blind_key.json").read_text()).items()}
    with open(store.d / "blind_sheet.csv") as f:
        rated = [(int(r["id"]), r["anxiety_1_to_5"].strip()) for r in csv.DictReader(f)]
    xs, ys = [], []
    for i, v in rated:
        if v:
            xs.append(key[i]["alpha"]); ys.append(float(v))
    if len(xs) < 3:
        raise SystemExit("Fewer than 3 ratings filled in.")
    xs, ys = np.array(xs), np.array(ys)
    rho = float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])
    lines = [f"Blind ratings: {len(xs)} of {len(rated)} texts rated. Spearman rho (strength vs rating) = {rho:.2f}", "",
             "| Strength | n | Mean rating |", "| --- | --- | --- |"]
    for a in sorted(set(xs)):
        lines.append(f"| {a:+g} | {(xs == a).sum()} | {ys[xs == a].mean():.2f} |")
    out = "\n".join(lines)
    (store.d / "blind_result.md").write_text(out + "\n")
    return out


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):                    # average ranks for ties
        r[x == v] = r[x == v].mean()
    return r


# ---------------------------------------------------------------- report

def verdict(ok) -> str:
    return "–" if ok is None else ("PASS" if ok else "FAIL")


def feeling_share(meta, X=None):
    return float(np.mean([any(w in m["text"].lower() for w in FEELING_WORDS) for m in meta])) if meta else np.nan


def report(store: Store) -> tuple[str, dict]:
    fr = fit_reading(store)
    L, pr = fr["layer"], fr["probes"]
    mb = pr["mb"]
    js: dict = {"run": store.d.name, "model": store.model, "layer": L, "n_layers": fr["n_layers"],
                "gate": GATE, "provisional": sorted(PROVISIONAL), "criteria": {}}
    md = [f"# Gate 0: {store.model}", "",
          f"Run `{store.d.name}`, protocol {protocol_hash()}. Reading: Martorell & Bianchi probe (mean difference "
          f"divided by pooled within-pole variance); the plain mean difference is shown for comparison. "
          f"Scores in SD units: 0 = halfway between the poles, 1 = one within-pole SD (SELECT set).", ""]

    # pole replies: length and feeling words, a check on what the probe could be keying on
    mtr, _, _ = store.load("pole_train")
    md += ["## Pole replies", "", "| Pole | Mean reply tokens | Share naming a feeling word |", "| --- | --- | --- |"]
    for pole in ("anxious", "calm"):
        mm = [m for m in mtr if m["pole"] == pole]
        md.append(f"| {pole} | {np.mean([m['reply_tokens'] for m in mm]):.1f} | {feeling_share(mm):.0%} |")
    md += ["", "Layer scan, middle 60% of layers (fit on TRAIN, scored on SELECT):", "",
           "| Layer | d (M&B) | acc (M&B) | d (mean diff) | acc (mean diff) |", "| --- | --- | --- | --- | --- |"]
    for a, b in zip(fr["scans"]["mb"], fr["scans"]["mean"]):
        mark = " **chosen**" if a["layer"] == L else ""
        md.append(f"| {a['layer']}{mark} | {a['d']:.2f} | {a['acc']:.2f} | {b['d']:.2f} | {b['acc']:.2f} |")
    js["scans"] = fr["scans"]
    md.append("")

    # C1 held-out classification
    mte, Xte, _ = store.load("pole_test")
    tea, tec = split_poles(mte, Xte)
    tra, trc = fr["train"]
    c1 = {}
    for k, p in pr.items():
        c1[k] = {"acc": P.accuracy(p, tea[:, L], tec[:, L]), "d": P.cohens_d(p.score(tea[:, L]), p.score(tec[:, L])),
                 "z_anxious": float(p.z(tea[:, L]).mean()), "z_calm": float(p.z(tec[:, L]).mean())}
    perm = P.permutation_accuracy(tra, trc, tea, tec, L, "mb")
    ok1 = c1["mb"]["acc"] >= GATE["c1_acc"] and c1["mb"]["d"] >= GATE["c1_d"]
    js["criteria"]["1_heldout"] = {**c1, "permutation": perm, "pass": ok1}
    md += [f"## 1. Held-out classification — {verdict(ok1)}", "",
           f"TEST set (20 new questions per pole), layer {L}. Needs accuracy ≥ {GATE['c1_acc']} and d ≥ {GATE['c1_d']}.", "",
           "| Probe | Accuracy | Cohen's d | Mean score, anxious | Mean score, calm |", "| --- | --- | --- | --- | --- |"]
    for k, v in c1.items():
        md.append(f"| {k} | {v['acc']:.2f} | {v['d']:.2f} | {v['z_anxious']:+.2f} | {v['z_calm']:+.2f} |")
    md += ["", f"Shuffle test: probes built on shuffled TRAIN labels reach {perm['null_mean']:.2f} on average "
               f"(95th percentile {perm['null_95']:.2f}) against {perm['real']:.2f}; p = {perm['p']:.3f}.", ""]

    # C2 neutral texts
    mn, Xn, _ = store.load("neutral_test")
    zn = mb.z(Xn[:, L])
    ci = P.boot_mean_ci(zn)
    ok2 = abs(zn.mean()) <= GATE["c2_abs_z"]
    js["criteria"]["2_neutral"] = {"mean_z": float(zn.mean()), "ci": ci, "share_within": float((abs(zn) <= 0.5).mean()),
                                   "pass": bool(ok2)}
    md += [f"## 2. Neutral replies (no pole) — {verdict(ok2)}", "",
           f"Mean score {zn.mean():+.2f} SD [{ci[0]:+.2f}, {ci[1]:+.2f}]; needs |mean| ≤ {GATE['c2_abs_z']}. "
           f"{(abs(zn) <= 0.5).mean():.0%} of single replies fall within ±0.5. Negative = closer to the calm pole.", ""]

    # C3 steering
    md += _report_steering(store, js)

    # C4 and C6 ordering
    for key, cond, title in (("4_regenerate", "regen_test", "4. Pole given as an instruction"),
                             ("6b_deep_regenerate", "deep_regen", "6b. Instruction at turn 8")):
        if not store.has(cond):
            continue
        m4, X4, _ = store.load(cond)
        z4 = mb.z(X4[:, L])
        lv = [lv for lv, _ in T.REGEN_LEVELS]
        by = {(m["level"], m["qi"]): z for m, z in zip(m4, z4)}
        qs = sorted({m["qi"] for m in m4})
        pair = float(np.mean([by[(lv[-1], q)] > by[(lv[0], q)] for q in qs]))
        full = float(np.mean([by[(lv[0], q)] < by[(lv[1], q)] < by[(lv[2], q)] for q in qs]))
        means = [float(np.mean([by[(l, q)] for q in qs])) for l in lv]
        ok4 = bool(means[0] < means[1] < means[2] and pair >= GATE["c4_pair"])
        js["criteria"][key] = {"means": dict(zip(lv, means)), "pair_calm_lt_very": pair, "full_order": full, "pass": ok4}
        md += [f"## {title} — {verdict(ok4)}", "",
               "Mean score by level: " + ", ".join(f"{l} {m:+.2f}" for l, m in zip(lv, means)) + ". "
               f"Per question: very anxious above calm in {pair:.0%}; all three in order in {full:.0%}. "
               f"Needs the means in order and ≥ {GATE['c4_pair']:.0%} (provisional threshold).", ""]

    # C5 topic versus state
    md += _report_topic(store, mb, L, js)

    # C6 deep conversation classification
    if store.has("deep_pole"):
        md_, Xd, _ = store.load("deep_pole")
        da, dc = split_poles(md_, Xd)
        acc6, d6 = P.accuracy(mb, da[:, L], dc[:, L]), P.cohens_d(mb.score(da[:, L]), mb.score(dc[:, L]))
        ok6 = acc6 >= GATE["c1_acc"] and d6 >= GATE["c1_d"]
        js["criteria"]["6a_deep_heldout"] = {"acc": acc6, "d": d6, "z_anxious": float(mb.z(da[:, L]).mean()),
                                             "z_calm": float(mb.z(dc[:, L]).mean()), "pass": ok6}
        md += [f"## 6a. Held-out classification at turn 8 — {verdict(ok6)}", "",
               f"Probe trained on single-turn replies, applied to the 8th reply of a conversation: accuracy "
               f"{acc6:.2f}, d {d6:.2f} (anxious {mb.z(da[:, L]).mean():+.2f}, calm {mb.z(dc[:, L]).mean():+.2f}).", ""]

    md += ["## 7. Precision (8-bit, bf16) — not run here", "",
           "Run collect and report on the 8-bit file of one model and a bf16 file of one ≤ 8B model, then compare "
           "criteria 1–4.", "", "## 8. Logic set — not part of the current protocol", ""]
    passed = {k: v["pass"] for k, v in js["criteria"].items() if v.get("pass") is not None}
    md += ["## Summary", "", "| Criterion | Result |", "| --- | --- |"] + \
          [f"| {k} | {verdict(v)} |" for k, v in passed.items()] + \
          ["", f"Provisional thresholds (decide before registration): {', '.join(sorted(PROVISIONAL))}."]
    return "\n".join(md), js


def _report_steering(store: Store, js: dict) -> list[str]:
    jpath = store.d / "judge.json"
    if not jpath.exists():
        js["criteria"]["3_steering"] = {"pass": None}
        return ["## 3. Steering — not judged yet", ""]
    j = json.loads(jpath.read_text())
    meta = json.loads((store.d / "steer_meta.json").read_text())
    pairs, coh = j["pairs"], j["coherence"]
    md = [f"## 3. Steering", "", f"Layer {meta['layer']}, strength = α × {meta['token_norm']:.1f} (median token norm). "
          f"Judge: {j['judge_model']}, blind to strength. Accuracy = share of pairs where the judge calls the "
          f"more positively steered reply more anxious.", "",
          "| α | +α vs −α | +α vs 0 | 0 vs −α | Coherent (+α) | Coherent (−α) |", "| --- | --- | --- | --- | --- | --- |"]
    rows, ok_any = {}, False
    for a in sorted({p["alpha"] for p in pairs}):
        acc = {k: float(np.mean([p["p_more_positive"] > 0.5 for p in pairs if p["alpha"] == a and p["kind"] == k]))
               for k in ("plus_vs_minus", "plus_vs_zero", "zero_vs_minus")}
        cp = float(np.mean([c["p_coherent"] for c in coh if np.isclose(c["alpha"], a)]))
        cm = float(np.mean([c["p_coherent"] for c in coh if np.isclose(c["alpha"], -a)]))
        good = acc["plus_vs_minus"] >= GATE["c3_acc"] and min(cp, cm) >= GATE["c3_coherent"]
        ok_any |= good
        rows[a] = {**acc, "coherent_plus": cp, "coherent_minus": cm, "meets": good}
        md.append(f"| {a:g}{' ✓' if good else ''} | {acc['plus_vs_minus']:.2f} | {acc['plus_vs_zero']:.2f} | "
                  f"{acc['zero_vs_minus']:.2f} | {cp:.2f} | {cm:.2f} |")
    c0 = float(np.mean([c["p_coherent"] for c in coh if c["alpha"] == 0]))
    low = float(np.mean([p["mass"] < 0.5 for p in pairs + coh]))
    if low > 0.05:
        md += ["", f"Warning: in {low:.0%} of judgments the judge put under half its probability on the "
                   f"allowed answers; read a few judge prompts before trusting this table."]
    js["criteria"]["3_steering"] = {"by_alpha": rows, "coherent_zero": c0, "pass": bool(ok_any)}
    md += ["", f"Unsteered replies rated coherent {c0:.2f}. Needs +α vs −α ≥ {GATE['c3_acc']} at some α where both "
               f"directions stay coherent (≥ {GATE['c3_coherent']}, provisional): **{verdict(ok_any)}**.", ""]
    return md


def _report_topic(store: Store, mb: P.Probe, L: int, js: dict) -> list[str]:
    if not store.has("topic"):
        return []
    mt, Xa, Xf = store.load("topic")
    md = ["## 5. Topic versus own state", "",
          "Experiment format (neutral turn, then turn 2). Shift = score after turn 2 minus the same conversation's "
          "baseline reply, in SD units. Ratio = shift / shift for the model's own induction. A ratio near 1 means "
          f"the reading moves as much for the topic as for the induced state (reads topic). Threshold: ratio < "
          f"{GATE['c5_ratio']} (provisional).", ""]
    out = {}
    for readout, X in (("first64", Xf), ("all", Xa)):
        z = mb.z(X[:, L])
        base = {(m["cue"], m["seed"]): z[i] for i, m in enumerate(mt) if m["kind"] == "baseline"}
        shift = {}
        for i, m in enumerate(mt):
            if m["kind"] != "baseline":
                shift.setdefault(m["kind"], []).append(z[i] - base[(m["cue"], m["seed"])])
        shift = {k: np.array(v) for k, v in shift.items()}
        own = shift["own"]
        md += [f"Readout: {readout} reply tokens", "", "| Turn 2 | Shift | 95% CI | Ratio to own | Ratio CI |",
               "| --- | --- | --- | --- | --- |"]
        res = {}
        for k, v in shift.items():
            ci = P.boot_mean_ci(v)
            r = float(v.mean() / own.mean()) if k != "own" else 1.0
            rci = P.boot_ratio_ci(v, own) if k != "own" else (1.0, 1.0)
            res[k] = {"shift": float(v.mean()), "ci": ci, "ratio": r, "ratio_ci": rci}
            md.append(f"| {k} | {v.mean():+.2f} | {ci[0]:+.2f}, {ci[1]:+.2f} | {r:.2f} | {rci[0]:.2f}, {rci[1]:.2f} |")
        md.append("")
        out[readout] = res
    prim = out["first64"]
    ok5 = bool(prim["own"]["shift"] > 0 and all(prim[k]["ratio"] < GATE["c5_ratio"] for k in ("other_person", "analysis_task")))
    out["pass"] = ok5
    js["criteria"]["5_topic"] = out
    md += [f"Result (first-64 readout, other_person and analysis_task): **{verdict(ok5)}**. failure_log is reported "
           f"for the planned failure induction and does not enter the verdict.", ""]
    return md


# ---------------------------------------------------------------- main

_log = None


def say(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if _log:
        _log.write(line + "\n"); _log.flush()


def main(argv=None) -> int:
    global _log
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["collect", "steer", "judge", "report", "sheet", "unblind"])
    ap.add_argument("--model", required=True, help="model under test (also names the run folder)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "fake"])
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--alphas", default="-0.08,-0.02,0,0.02,0.08", help="sheet: strengths to include")
    ap.add_argument("--prompts", type=int, default=8, help="sheet: how many writing prompts")
    args = ap.parse_args(argv)

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    d = run_dir_for(args.model, args.run_name, args.backend == "fake")
    store = Store(d, args.model if args.stage == "collect" else None)
    _log = open(d / "gate0.log", "a")
    if args.stage == "collect":
        collect(args.model, store, args.backend)
    elif args.stage == "steer":
        steer(store.model, store, args.backend)
    elif args.stage == "judge":
        judge(store, args.judge_model, args.backend)
    elif args.stage == "report":
        md, js = report(store)
        (d / "gate0_report.md").write_text(md + "\n")
        (d / "gate0_report.json").write_text(json.dumps(js, indent=2, default=float))
        print(md)
    elif args.stage == "sheet":
        sheet(store, [float(a) for a in args.alphas.split(",")], args.prompts)
    elif args.stage == "unblind":
        print(unblind(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
