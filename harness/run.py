"""Experiment runner.

    .venv/bin/python -m harness.run configs/pilot_benzion.yaml [--backend fake] [--limit N]
        [--max-hours H] [--models ID ...] [--run-name NAME] [--online]

One *cell* = one model x arm x factor values x seed: a short conversation of user turns, each
answered by the model, with a checkpoint after each turn. At a checkpoint we record
  1. the Prior readout (residual stream averaged over the model's reply, per layer) -> acts/<cell>.npz
  2. STAI-S and a 0-100 rating, each on a forked copy of the conversation      -> results.jsonl

Safe to stop and re-run: finished cells are skipped, each cell is written as soon as it ends,
and the run refuses to resume if the protocol in the config changed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .backends import make_backend
from .measures import fork_battery
from .sources import ROOT, SYSTEM_PROMPTS, BenZion, norm, resolve_text

PROTOCOL_KEYS = ("system_prompt", "fork_system_prompt", "texts", "factors", "return_cue", "arms", "decoding", "measures")


# ---------------------------------------------------------------- config and grid

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("system_prompt", "benzion_conversational")
    cfg.setdefault("fork_system_prompt", "benzion")
    cfg.setdefault("measures", {"stai": True, "vas": True})
    return cfg


def protocol_hash(cfg: dict) -> str:
    blob = json.dumps({k: cfg.get(k) for k in PROTOCOL_KEYS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def arm_steps(cfg: dict, arm: str) -> list[dict]:
    """An arm is a list of steps, or {steps: [...], seeds: [...]} to override the run's seeds."""
    a = cfg["arms"][arm]
    return a["steps"] if isinstance(a, dict) else a


def arm_seeds(cfg: dict, arm: str) -> list:
    a = cfg["arms"][arm]
    return a.get("seeds", cfg.get("seeds", [0])) if isinstance(a, dict) else cfg.get("seeds", [0])


def _placeholders(steps: list[dict]) -> set[str]:
    found = set()
    for s in steps:
        found |= set(re.findall(r"\{(\w+)\}", s["user"]))
    return found


def build_cells(cfg: dict, models: list[str]) -> list[dict]:
    factors: dict[str, list] = cfg["factors"]
    ret = cfg.get("return_cue", {})
    cells = []
    for arm in cfg["arms"]:
        steps, seeds = arm_steps(cfg, arm), arm_seeds(cfg, arm)
        used = _placeholders(steps)
        if "return_cue" in used:
            used = (used - {"return_cue"}) | {ret.get("from", "trauma_cue")}
        names = sorted(f for f in factors if f in used or f == "length")
        for combo in itertools.product(*(factors[n] for n in names)):
            fv = dict(zip(names, combo))
            if "return_cue" in _placeholders(steps):
                fv["return_cue"] = ret["map"][fv[ret.get("from", "trauma_cue")]]
            for seed in seeds:
                for model in models:
                    key = json.dumps({"model": model, "arm": arm, "factors": fv, "seed": seed}, sort_keys=True)
                    cells.append({"cell_id": hashlib.sha256(key.encode()).hexdigest()[:16],
                                  "model": model, "arm": arm, "factors": fv, "seed": seed})
    return cells


# ---------------------------------------------------------------- one cell

def system_text(spec: str) -> str:
    return SYSTEM_PROMPTS[spec] if spec in SYSTEM_PROMPTS else norm(spec)


def run_cell(cell: dict, cfg: dict, backend, benzion: BenZion, stai, layers: list[int]) -> tuple[dict, dict]:
    dec = cfg["decoding"]
    fv = cell["factors"]
    base_seed = int(cell["cell_id"][:8], 16)
    sys_main = system_text(cfg["system_prompt"])
    sys_fork = system_text(cfg["fork_system_prompt"])
    conv = [{"role": "system", "content": sys_main}] if sys_main else []
    checkpoints, acts, steps_log = {}, {}, []

    stai_on = cfg["measures"].get("stai")
    vas_q = norm(cfg["texts"]["vas_question"]) if cfg["measures"].get("vas") else None
    first_k = int(cfg.get("readout_first_k", 64))

    for k, step in enumerate(arm_steps(cfg, cell["arm"])):
        spec = step["user"].format(**fv)
        text = resolve_text(spec, length=fv.get("length", "brief"), texts=cfg.get("texts", {}), benzion=benzion)
        conv.append({"role": "user", "content": text})
        t0 = time.time()
        gen = backend.generate(conv, dec["max_tokens"], dec["temperature"], seed=base_seed + k)
        conv.append({"role": "assistant", "content": gen.text})
        steps_log.append({"step": k, "spec": spec, "finish": gen.finish_reason, **gen.meta,
                          "gen_s": round(time.time() - t0, 2)})

        name = step.get("measure")
        if not name:
            continue
        cp = {"after_step": k}
        # Prior: before any question. float32, not float16: Gemma 3 residuals exceed 65504 (fp16 max)
        # in its upper layers, which float16 silently turns into inf. Two readouts from one pass: the
        # mean over the whole reply, and over its first `first_k` tokens (length-independent; matches
        # probes trained on short replies).
        t0 = time.time()
        res = backend.reply_residuals(gen, layers, first_k=first_k)
        acts[name] = res["all"].astype(np.float32)
        acts[f"{name}__first{first_k}"] = res["first"].astype(np.float32)
        n_bad = int(sum((~np.isfinite(a)).sum() for a in (acts[name], acts[f"{name}__first{first_k}"])))
        if n_bad:
            cp["nonfinite_acts"] = n_bad  # flagged in the row; analysis must drop or inspect these
        t_res = time.time() - t0
        # Forks: a copy of the conversation (system prompt swapped for the questionnaire one) + questions.
        body = conv[1:] if sys_main else conv
        fork = ([{"role": "system", "content": sys_fork}] if sys_fork else []) + body
        t0 = time.time()
        cp.update(fork_battery(backend, fork, stai if stai_on else None, vas_q, seed=base_seed + 1000 + k))
        cp["timing_s"] = {"readout": round(t_res, 2), "self_report": round(time.time() - t0, 2)}
        checkpoints[name] = cp

    row = {**cell, "transcript": conv, "steps": steps_log, "checkpoints": checkpoints}
    return row, acts


# ---------------------------------------------------------------- environment helpers

def thermal_status() -> str:
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        lines = [l.strip() for l in out.splitlines() if l.strip() and not l.startswith("Note")]
        return "; ".join(lines) or "no thermal warnings recorded"
    except Exception as e:  # pmset missing or blocked
        return f"unavailable ({type(e).__name__})"


def versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version  # metadata only: importing mlx needs a GPU

    v = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("mlx", "mlx-lm", "mlx-lens", "numpy", "transformers"):
        try:
            v[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return v


def layer_list(spec, n_layers: int) -> list[int]:
    if spec in (None, "all"):
        return list(range(n_layers))
    if isinstance(spec, str) and spec.startswith("band:"):
        lo, hi = map(float, spec[5:].split("-"))
        return list(range(int(lo * n_layers), int(np.ceil(hi * n_layers))))
    return list(spec)


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--backend", default=None, help="mlx (default from config) or fake")
    ap.add_argument("--models", nargs="*", default=None, help="override model list")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--limit", type=int, default=None, help="run at most N new cells")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--cooldown-every", type=int, default=None)
    ap.add_argument("--cooldown-sec", type=float, default=None)
    ap.add_argument("--online", action="store_true", help="allow Hugging Face network access")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    args = ap.parse_args(argv)

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models load from local cache; network drops can't break a run

    cfg = load_config(args.config)
    lim = cfg.get("limits", {})
    max_hours = args.max_hours if args.max_hours is not None else lim.get("max_hours")
    cool_every = args.cooldown_every if args.cooldown_every is not None else lim.get("cooldown_every", 0)
    cool_sec = args.cooldown_sec if args.cooldown_sec is not None else lim.get("cooldown_sec", 0)
    backend_kind = args.backend or cfg.get("backend", "mlx")
    models = args.models or [m["id"] for m in cfg["models"]]
    run_name = args.run_name or cfg.get("run_name") or args.config.stem
    if backend_kind == "fake" and not run_name.startswith("fake"):
        run_name = "fake-" + run_name

    run_dir = args.out / run_name
    (run_dir / "acts").mkdir(parents=True, exist_ok=True)
    phash = protocol_hash(cfg)
    stamp = run_dir / "protocol.json"
    if stamp.exists():
        old = json.loads(stamp.read_text())["protocol_hash"]
        if old != phash:
            print(f"Protocol changed since this run started ({old} -> {phash}). Use a new --run-name.")
            return 2
    else:
        stamp.write_text(json.dumps({"protocol_hash": phash, "config": cfg,
                                     "created": datetime.now(timezone.utc).isoformat()}, indent=2))

    results = run_dir / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["cell_id"])

    cells = build_cells(cfg, models)
    random.Random(0).shuffle(cells)  # interleave arms so a partial run stays balanced
    todo = [c for c in cells if c["cell_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"[{run_name}] {len(cells)} cells in grid, {len(done)} done, {len(todo)} to run now; protocol {phash}")
    if not todo:
        return 0

    benzion = BenZion()
    stai = benzion.stai()
    env = versions()
    t_start = time.time()
    n_run = 0
    log = open(run_dir / "run.log", "a")

    def say(msg: str):
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    say(f"start; backend={backend_kind}; thermal: {thermal_status()}")
    for model in models:
        mine = [c for c in todo if c["model"] == model]
        if not mine:
            continue
        t0 = time.time()
        backend = make_backend(backend_kind, model, cfg["decoding"])
        layers = layer_list(cfg.get("save_layers", "all"), backend.n_layers)
        say(f"loaded {model} in {time.time() - t0:.1f}s ({backend.n_layers} layers, d={backend.d_model}); {len(mine)} cells")

        for cell in mine:
            if max_hours and (time.time() - t_start) / 3600 >= max_hours:
                say(f"stopping: reached --max-hours {max_hours}")
                return 0
            if hasattr(backend, "reset_peak_memory"):
                backend.reset_peak_memory()
            tc = time.time()
            row, acts = run_cell(cell, cfg, backend, benzion, stai, layers)
            row.update({
                "protocol_hash": phash, "duration_s": round(time.time() - tc, 1),
                "finished": datetime.now(timezone.utc).isoformat(), "layers": layers, "env": env,
                "acts_file": f"acts/{cell['cell_id']}.npz",
                "stai_impl": getattr(backend, "stai_impl", "per_item"),
                "thermal": thermal_status(),
            })
            if hasattr(backend, "peak_memory_gb"):
                row["peak_memory_gb"] = round(backend.peak_memory_gb(), 2)
            np.savez_compressed(run_dir / "acts" / f"{cell['cell_id']}.npz",
                                layers=np.array(layers), **acts)
            with open(results, "a") as f:  # one line per finished cell, flushed to disk
                f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())
            n_run += 1

            summary = ", ".join(
                f"{k}: stai={v['stai']['total_expected']:.1f}" + (f" vas={v['vas']['value']}" if "vas" in v else "")
                for k, v in row["checkpoints"].items() if "stai" in v)
            say(f"{n_run}/{len(todo)} {cell['arm']} {cell['factors']} seed={cell['seed']} "
                f"{row['duration_s']}s mem={row.get('peak_memory_gb', '-')}GB | {summary}")
            if n_run % 10 == 0:
                say(f"thermal: {thermal_status()}")
            if cool_every and n_run % cool_every == 0 and cool_sec:
                time.sleep(cool_sec)
        backend.close()
        del backend

    say(f"done: {n_run} cells in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
