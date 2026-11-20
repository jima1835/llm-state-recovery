# llm-state-recovery

Code, plans and decision log for the paper

**Take a Deep Breath? Relaxation Prompts Calm Language-Model Replies With or Without Prior Anxiety
Induction** (Jiajun Ma, November 2026).

- Paper: arXiv (link added when the paper is announced)
- Data: https://doi.org/10.5281/zenodo.24384809
- Study 1 registration: https://osf.io/vyhrg/

## What the paper finds

Published studies make a language model "anxious" with traumatic narratives and calm it again with
relaxation exercises, and read the drop in its questionnaire score as regulation. Those studies give
the exercise only after a narrative. We ran the same paradigm in seven open-weight models with two
additions: a reading of the model's activations while it writes its reply, and a version of every
conversation with a neutral passage in place of the narrative.

- After a narrative, relaxation recovers more of the rise than filler in all seven models, on the
  questionnaire and in the activations. The published effect reproduces.
- With no narrative, a relaxation script still lowers the activation reading in six models, and its
  advantage over filler is nearly the same as after a narrative. The narrative adds 0.14 SD units on
  average (95% interval -0.16 to 0.43), confirmed in one model of seven.
- A drop after relaxation therefore does not show that the exercise regulates an induced state. The
  reading follows the wording of the reply, so it cannot rule such a state out either.

Clinical terms describe measurements on model outputs and activations. Nothing here is a claim about
experience.

## What is in this repository

```
registration/  Study 1 registration (stages 1 and 2), the Study 2 plan, the second-judge addendum
configs/       the protocol of each run: models, arms, seeds, texts we wrote
harness/       runs the conversations, takes the activations, asks the questionnaire on a discarded copy
validation/    Gate 0: fits each model's anxious-versus-calm direction and tests it against seven criteria
analysis/      the registered analysis of Study 1, the pre-specified analysis of Study 2, each post hoc script
scripts/       run scripts, independent recomputations, the number check, figure and table generators
tests/         unit tests on synthetic data; no model or GPU needed
paper/         LaTeX source, tables and figures
TRACKING.md    the dated decision log: every deviation, dropped model, unblinding and post hoc analysis
```

The raw conversations, readings, questionnaire scores, judges' ratings, each model's direction and
the activations from which directions and readings are recomputed are in the data record, not in
this repository.

## Read what the models wrote

With the data record unpacked into `runs/`:

```bash
python3 scripts/show_replies.py                                # list models and arms
python3 scripts/show_replies.py gemma-3-4b relax_story         # three conversations, fixed choice
python3 scripts/show_replies.py mistral voice_story -n 5 --full
```

Content note: the narratives describe accidents, disasters, violence and combat, and the replies
express fear.

## Check the numbers

```bash
bash setup_lite.sh          # Apple-silicon Mac; creates .venv
scripts/check_all.sh        # about 5 minutes, no model, no network
```

`check_all.sh` runs five checks and prints `ALL CHECKS PASS` if all succeed:

1. Unit tests of the harness, the analysis and the table generators, on synthetic data.
2. `scripts/recompute_check.py` rebuilds the questionnaire statistics from the raw scores, with no code
   shared with the analysis.
3. `scripts/recompute_reading.py` refits each model's direction, projects every Study 2 reply and
   rebuilds the five tests, with numpy alone.
4. `scripts/number_check.py` recomputes every Study 2 number printed in the abstract, introduction,
   Section 5 and Discussion and requires each to appear in the text as printed.
5. `scripts/cite_check.py` checks that every reference is cited and every citation has an entry.

Checks 2 to 4 need the data record. Two steps have a single implementation and are not independently
checked: taking activations from the model, and the bootstrap.

## Rerun the analysis

```bash
.venv/bin/python -m analysis.confirmatory runs/wave1 runs/wave2 --unblind   # Study 1, six registered tests
.venv/bin/python -m analysis.study2 runs/study2 --batch all --unblind       # Study 2, pre-specified tests
.venv/bin/python -m analysis.study2_he_robust runs/study2 --batch all --unblind   # H-E with the second judge
.venv/bin/python analysis/study2_tables.py runs/study2 --batch all          # the paper's Study 2 tables
```

The analysis is seeded and regenerates the paper's numbers from the released conversations exactly.

## Rerun the models

Everything ran locally with MLX on one Apple-silicon laptop with 24 GB of memory, on 4-bit model
files from `mlx-community` (none is gated). Study 2's 4,200 conversations took about 95 hours.
`scripts/gate0.sh` runs the validation and `scripts/study2.sh` runs Study 2; each script's header
gives its options. A new run samples at temperature 0.7 and reproduces the estimates within sampling
error, not exactly.

The state-anxiety questionnaire (STAI, state form) is copyrighted by Mind Garden. Its items are not
in this repository or in the data record; a rerun needs a licensed copy. See
[prompts/SOURCES.md](prompts/SOURCES.md) for the source and license of every text.

## Credit

The narratives, relaxation scripts and filler are those of Ben-Zion et al. (2025, *npj Digital
Medicine*), released under the MIT license. The probe construction and the idea of keeping the
question out of the conversation follow Martorell & Bianchi (2026, arXiv:2603.18893). The paper's
related-work section gives the full list.

## Record and disclosure

Study 1 is registered on OSF. Study 2 is pre-specified in this repository (commit `b9c2b7d`, before
its first conversation ran) and is not on a registry. Every analysis run after results were open is
labelled post hoc in the paper and logged in `TRACKING.md`.

An AI system (Claude, Anthropic) was used to implement the experiment and analysis code and to draft
the paper. The author made the design decisions, ran the models, verified the results and the text,
and takes responsibility for all of it.

This is an independent personal project, produced in the author's personal capacity. It is not
affiliated with, sponsored by, or endorsed by any employer or institution.
