# llm-state-recovery

Does calming a language model change what is going on inside it, or only what it says?

We put small open models through published negative-affect inductions, apply a published
relaxation script (guided imagery, from Ben-Zion et al. 2025), and measure two channels at each step:

1. **What the model says** — standard self-report scales, asked on a forked copy of the
   conversation so the question never contaminates the run.
2. **What the model holds internally** — a "distress" direction in the residual stream,
   read before any question is asked and validated before use (held-out classification,
   indifference to unrelated text, and steering that changes output).

Then we add one further negative event and check whether the internal state comes back.

All clinical terms are used in quotation marks. "Distress" here means a measurable induced
state in a model's activations; it makes no claim about experience.

## Why

- Wekenborg et al. (2026, *Lancet Digital Health*) and Ben-Zion et al. (2025, *npj Digital
  Medicine*) showed that induction prompts raise models' self-reported negative affect and
  relaxation and mindfulness prompts lower it, but measured only the self-report.
- Martorell & Bianchi (2026, arXiv 2603.18893) showed that models' self-reports do not track
  their internal affect probes.
- So it is open whether regulation reaches the internal state. Sofroniew et al. (2026,
  arXiv 2604.07729) describe emotion representations as "locally scoped"; Lee et al.
  (2026, arXiv 2504.08016) find internal states that persist and resist correction. The
  return test separates these.

**Credit.** The report–probe dissociation is Martorell & Bianchi's. The induction and
regulation protocols are Wekenborg's and Ben-Zion's. The probe method is Sofroniew et al.'s,
as adapted by Jeong (2026) and Martorell & Bianchi. What this project adds is reading a
validated internal channel through a validated induction-and-regulation protocol, then
testing whether the state returns.

## Status

Setup stage. Nothing here is a result yet.

- [x] Environment and folder layout
- [x] Smoke test on Gemma 3 4B (4-bit, MLX): loads in ~2 s, layer-17 capture `(1, tokens, 2560)`, 2.7 GB peak memory
- [ ] First probe (`probes/day1_probe.py`, written by hand)
- [x] Experiment runner (`harness/`): Ben-Zion protocol, forked self-report, Prior readout, resumable JSONL; tested with a stand-in backend
- [ ] Runner checked on a real model
- [ ] Validation gate on two pilot models, then pre-registration
- [ ] Main grid: six instruction-tuned models, 4B–14B

Progress, decisions and the dated log are in [TRACKING.md](TRACKING.md).

## Layout

```
configs/    model list and decoding settings
probes/     probe code (Day 1 skeleton first)
harness/    experiment runner: run.py (grid, resume, limits), backends.py (MLX + test stand-in),
            measures.py (forked STAI and 0-100 rating), sources.py (prompt loading), show.py (read transcripts)
scripts/    helpers: status.sh (progress), after_pilot.sh (checks), start_v2.sh (start the v2.0 run)
tests/      pipeline tests (no GPU needed): .venv/bin/python -m pytest -q tests
prompts/    prompt texts, with SOURCES.md giving URL, commit and license for each
analysis/   statistics and figures (to come)
runs/       JSONL outputs (not committed)
```

## Setup (Apple Silicon Mac)

```bash
bash setup_lite.sh
.venv/bin/python smoke_test.py
```

Call `.venv/bin/python` directly: if your shell aliases `python`, activating the
environment is not enough. The `mlx-community` models used here are not gated, so no
Hugging Face login is needed; logging in (`.venv/bin/hf auth login`, read-only token) only
raises download rate limits.

## Running the pilot

```bash
.venv/bin/python -m harness.run configs/pilot_benzion.yaml --models mlx-community/gemma-3-4b-it-4bit --limit 2
.venv/bin/python -m harness.show runs/pilot-benzion-v2
```

The protocol lives in `configs/pilot_benzion.yaml`. Each cell is one short conversation; after each
turn the runner saves the residual stream averaged over the model's reply (`runs/<run>/acts/`) and
asks the questionnaire on a copy of the conversation (`runs/<run>/results.jsonl`). Stopping and
re-running continues where it left off. Models load from the local cache only.

The 20 questionnaire items share the conversation, so the runner reads it once and answers each item
from a copy of that state (`stai_impl: prefix_reuse` in each row). Set `LSR_PREFIX_REUSE=0` to answer
each item with a full pass instead, and check that both give the same answers with
`.venv/bin/python -m harness.check_prefix_reuse runs/<run>`.

## Prompt sources

See [prompts/SOURCES.md](prompts/SOURCES.md). Third-party prompt text is copied into this
repository only where its license allows it.
