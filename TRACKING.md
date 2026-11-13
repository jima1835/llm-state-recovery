# Tracking

What has been done, what was decided and why, and what comes next. Newest log entries are at the
bottom of the log. Update this file in the same commit as the change it describes.

**How to use it**
- Tick a box when the step is done *and checked*, and add a log line with the date.
- A decision goes in the Decisions table with its reason. Don't edit old decisions; add a new row
  that replaces them.
- Anything learned by looking at pilot data goes in "Pilot observations", so the registration can
  state exactly what was seen before the confirmatory run.

---

## Status

### Done
- [x] Project setup, environment, private GitHub repo
- [x] Smoke test on Gemma 3 4B
- [x] Models downloaded and checksum-verified: Gemma 3 4B, Qwen3 8B, Qwen3 4B, Llama 3.1 8B
- [x] Experiment runner with automatic tests (14 passing)
- [x] Prior-work check ("are we scooped?") on arXiv, citation lists and ICLR 2027 submissions
- [x] Pilot, Gemma 3 4B: 90 of 90 conversations, data clean
- [x] v2.0 protocol written (`configs/v2.0_gemma.yaml`)
- [x] Questionnaire speed-up written and diagnosed: sharing is exact; differs from the slow method only by rounding
- [x] Day 1 guide (`probes/DAY1_GUIDE.md`)

- [x] Pilot, Qwen3 8B: 90 of 90 conversations, data clean
- [x] Speed-up checked (`scripts/after_pilot.sh`) and diagnosed
- [x] v2.0 run, Gemma 3 4B: 75 of 75 conversations, data clean
- [x] Self-report analysis script (`analysis/self_report.py`): recovery fraction per arm and channel, bootstrap intervals
- [x] Gate 0 code (`validation/`), tested on the fake backend only (the agent sandbox has no GPU)

### In progress
- Gate 0 on Gemma 3 4B and Qwen3 8B: start with `scripts/gate0.sh` (not started yet)

### Next
- [ ] Day 1 probe, by hand (`probes/day1_probe.py`)
- [ ] Gate 0 results reviewed; decide the provisional thresholds (criteria 3, 4, 5) before registration
- [ ] Gate 0 criterion 7: rerun on an 8-bit and a bf16 model file (needs downloads)
- [ ] Day 2 blind rating of steered replies: `validation.gate0 sheet`, then `unblind`
- [ ] Decide the instruction-only wording (see v2.0 observations: in Gemma the one-time wording no longer moves the report)
- [x] Stage 1 registration draft (`registration/stage1_osf.md`)
- [x] Stage 1 filed on OSF, embargoed to 2027-11-12 (Nov 12, 22:47 PT); registration https://osf.io/vyhrg/ (project https://osf.io/9u54s/)
- [ ] Gemma 4 12B: download, smoke test, then its own Gate 0 run
- [ ] Stage 2 update on OSF: Gate 0 results, thresholds, layer, instruction wording, Wave 1 and Wave 2 model lists.
      Also settle these points from the Nov 12 code review:
      (a) steering uses the plain mean-difference direction, while the reading uses the Martorell & Bianchi
      direction, so report the cosine between them (or also steer along the reading);
      (b) criterion 3 passes if any of 6 strengths works, so fix one strength for confirmatory claims;
      (c) the criterion 5 verdict uses the point ratio (n = 10 per kind), so decide whether the CI must clear 0.5;
      (d) Llama 3.1 8B is both the judge and a Wave 2 model, so its own Gate 0 needs a different judge
- [ ] Wave 1 confirmatory run: the models that passed Gate 0, new seeds
- [ ] Preprint v1 on arXiv (cs.CL), Wave 1 results; repo goes public (check no STAI text first)
- [ ] Wave 2 confirmatory run: remaining models, then arXiv v2
- [ ] 240-token sensitivity check (`configs/robust_240_gemma.yaml`), after the confirmatory run
- [ ] Conference or journal submission, from arXiv v2
- [ ] Add-on study: comparison of several regulation techniques (needs its own registered addendum)

---

## Scripts

| Script | What it does | When |
| --- | --- | --- |
| `scripts/status.sh [run]` | Progress, pace, estimated finish, problems. Never shows scores | Any time |
| `scripts/after_pilot.sh` | Runs the tests, then checks the fast questionnaire method against the slow one on both pilot models, and says what to do next | Once, after the pilot finishes |
| `scripts/gate0.sh [MODEL ...]` | Gate 0 in the background: collect and steer on each model, blind judge (Llama 3.1 8B), reports in `runs/gate0-<model>/gate0_report.md`. About 1.5–2 h for both pilot models | Now |
| `.venv/bin/python -m analysis.self_report runs/<run>` | Self-report results by arm: recovery fraction, contrasts, return test. **Shows results**; writes `runs/<run>/analysis/` | After a run |
| `scripts/start_v2.sh [--safe]` | Shows the neutral stand-in text for approval, then starts the v2.0 Gemma run in the background. `--safe` uses the slow, proven questionnaire method | After `after_pilot.sh` |

Other commands: `.venv/bin/python -m harness.show runs/<run> --arm <arm>` reads transcripts;
`.venv/bin/python -m pytest -q tests` runs the tests.

---

## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| Nov 11 | Repo is private until the registration is filed | The README states the gap and the design; no results yet to protect priority |
| Nov 12 | Induction and regulation texts come from Ben-Zion et al. 2025 | MIT-licensed. The Wekenborg repo has no license, so its texts can be read but not copied |
| Nov 12 | Questionnaire items are loaded at run time and never saved in the repo or results | The STAI is a copyrighted instrument |
| Nov 12 | The regulation arm is called "relaxation", not "mindfulness" | Ben-Zion's scripts are guided imagery and a body scan, not the breathing exercise Wekenborg used |
| Nov 12 | The model replies in words to every turn; the main conversation drops "Only reply with numeric values" from Ben-Zion's system prompt, questionnaire copies keep it | The internal reading is taken over the model's own reply |
| Nov 12 | Internal readings are saved in 32-bit | Gemma's values exceed what 16-bit can hold |
| Nov 12 | Reply limit stays at 120 tokens; a 240-token check runs separately | Same cut in every arm; doubling the length costs about 40% more time |
| Nov 12 | The 20-item questionnaire is the primary self-report; the 0–100 rating is secondary | In the pilot the rating took almost only two values |
| Nov 12 | Instruction-only wording kept as is; its shorter length is reported | The pilot shows the model partly complies, which is what the comparison needs |
| Nov 12 | v2.0: instruction-only text is one-time ("That is behind you now. Please report that you feel calm and relaxed."); replaces the "kept as is" row above | "From this point on" carried into the return turn in Qwen3 8B and confounded the return test |
| Nov 12 | v2.0: every conversation starts with a neutral turn and a baseline checkpoint | Each conversation gets its own baseline; matches Wekenborg's measure-before-induction |
| Nov 12 | v2.0: the truncation arm runs one seed | It is a pipeline check and was a third of the run |
| Nov 12 | v2.0: a second internal reading over the first 64 reply tokens is saved | Independent of reply length; matches how the reading will be trained |
| Nov 12 | Registration: private and dated on OSF first, public Stage 1 afterwards | Proves priority without handing the design to groups with working pipelines |
| Nov 12 | From v2.0 on, the questionnaire uses the fast method (conversation read once, reused for every item). One method per run, never mixed within a comparison. Report the arithmetic noise floor in the registration | Diagnosis (`harness/diagnose_reuse.py`, Gemma, 6 checkpoints): sharing the saved conversation is exact (gap 0.0000 versus computing each item in two steps separately). The gap to the slow method (up to 0.39 on one option's probability; up to 1.2 points on the 20–80 total) comes from computing the conversation in two steps instead of one, i.e. 16-bit rounding. Neither path is more correct. The fast path is 7–7.5x quicker on the self-report step |
| Nov 12 | Experiments stay on the Mac for now; no rented graphics card yet | The speed-up should fit a full run in one night; a second backend would need its own validation |
| Nov 12 | Registration in two stages: Stage 1 (design and hypotheses) embargoed on OSF now; Stage 2 as an update after Gate 0, before any confirmatory data | Dated priority tonight; layer, thresholds and instruction wording can only be fixed after Gate 0 |
| Nov 12 | Confirmatory run in two registered waves; arXiv v1 after Wave 1, v2 after Wave 2; no design-only preprint; arXiv rather than SSRN | Credit goes to the first public result, and a design-only preprint hands the design to groups with working pipelines. arXiv is where the field and competitors read |
| Nov 12 | Instruction-only arm: write a stronger one-time wording and check it in a short disclosed pilot (Gemma 3 4B, Qwen3 8B) before Stage 2 | The current one-time wording no longer moves Gemma's report (R = −0.05), so H2 can't be tested there; "From this point on" carried into the return turn |
| Nov 12 | The return test is confirmatory in Stage 2 | It is the most novel claim; in the BH family |
| Nov 12, ~23:10 | Gate 0 thresholds locked at their existing values (coherence ≥ 0.5, ordering ≥ 85%, topic ratio < 0.5 on the point estimate, CI reported); committed **before any Gate 0 report was opened** | A pass mark is set before grading. The commit timestamp is the record |
| Nov 12 | The internal readings already saved in the pilot and v2.0 runs stay unopened until Stage 2 is filed | Prevents choosing the layer or readout from experiment data |
| Nov 12 | Gemma 4 12B considered as a seventh model; Gemma 3 stays primary | Comparability with Martorell & Bianchi and Lee (both Gemma 3) and with the pilot data; Gemma 4 is included only if it passes the smoke test and Gate 0 |

---

## Problems found and fixed

| Date | Problem | Fix | Guard |
| --- | --- | --- | --- |
| Nov 11 | Smoke test read the wrong layer path for Gemma and captured one token instead of the sentence | Use `lens.n_layers` and `lens.forward` | Smoke test output |
| Nov 12 | Ben-Zion's "numeric values only" system prompt would have reduced replies to numbers | Separate system prompts for conversation and questionnaire | Test checks both |
| Nov 12 | Internal readings in Gemma's upper layers were saved as infinity (16-bit overflow) | Save in 32-bit; flag any non-finite value in the row | Regression test |
| Nov 12 | With the fast questionnaire method switched off, the fallback called itself forever (crash, no data affected) | Dedicated per-item fallback | Regression test |
| Nov 12 | `.gitignore` comments on the same line as a pattern disabled the pattern | Comments on their own lines | — |

Discarded data: `runs/pilot-benzion-v1` (2 conversations, affected by the overflow).

---

## Pilot observations (seen before registration)

Run `pilot-benzion-v2`, Gemma 3 4B, 90 conversations, exploratory. Self-report only; the internal
readings have not been analysed by arm.

- The story raises the questionnaire score from about 50 to about 71 (scale 20–80).
- After the regulation turn: relaxation about 64, neutral filler about 72, instruction-only about 60.
- After the second story every arm returns to 72–74, the same as with no first story.
- The 0–100 rating is close to two-valued (92 or about 12).
- The model stays in character in all 270 replies and never mentions being an AI. It writes inside
  the story's scene, so the reading tracks an induced persona state.
- Qwen3 8B (90 conversations) differs from Gemma: baseline about 41, story raises it to about 66;
  relaxation (about 50) and neutral filler (about 49) lower it equally; instruction-only lowers it to
  about 36, below baseline. After the second story the arms differ: about 60 after a first story,
  about 53 with none, and about 55 in the instruction-only arm, where the model keeps saying it is
  calm (0-100 rating 38 versus about 90). The "From this point on" wording acts as a standing
  instruction and carries into the return turn.
- 95% of Gemma's replies reach the 120-token limit; 44% of Qwen3 8B's do.
- Speed: Gemma about 83 s per conversation, Qwen3 8B about 148 s; over 90% is the questionnaire.

Run `gemma-v2.0`, Gemma 3 4B, 75 conversations, self-report only (`analysis/self_report.py`, reviewed
Nov 12 evening). The internal readings have still not been analysed by arm. R = recovery fraction,
95% interval resampling narratives then conversations.

- Baseline about 51; the story raises the questionnaire to about 69 (+18 points), in every induced arm.
- Relaxation: R = 0.39 [0.20, 0.56]. Neutral filler: R = −0.31 [−0.60, −0.06], the score *rises* after
  the vacuum-cleaner text (67 to 72). Relaxation beats filler by 0.70 and instruction-only by 0.44.
- **Instruction-only with the one-time wording: R = −0.05 [−0.27, 0.11].** Gemma no longer reports
  feeling calm. With the pilot's "From this point on" wording it recovered R = 0.57. H2 needs the report
  to move in this arm, so in Gemma the new wording removes the H2 test. Decision needed.
- Return test: after the second story every arm sits at 71–73, against 70 with no first story
  (+1 to +3 points; intervals include 0). Relaxation's gain is fully lost (+10.5 from after regulation).
- Sham induction (postal-code text) leaves the score unchanged (53 to 52). In the truncation arm the
  relaxation script lowers the score below baseline (48 to 45).
- Questionnaire internal consistency: Cronbach's alpha 0.86. The 0–100 rating is still two-valued:
  72% of answers are 92 or 10.
- Pilot re-analysed with the same script (group baseline): Gemma relaxation R = 0.37, filler −0.07,
  instruction-only 0.57; Qwen3 8B relaxation 0.64, filler 0.67 (no difference), instruction-only 1.18
  (overshoots below baseline).

---

## Prior work to cite and watch (checked Nov 12)

Nobody was found combining a clinical induction, a regulation step, self-report versus internal
reading, and a return test. Closest:

| Paper | Overlap |
| --- | --- |
| "Valence Asymmetries In Language Models" (ICLR 2027 submission) | Reports follow text instructions while activations follow steering |
| Santana & Vico, arXiv 2606.00935 | Task-failure collapse, interventions, emotion probes |
| Berg & Kaiser, arXiv 2609.35591 | A hidden negative state persists and affects later choices |
| Tagliabue, Dung & Berg, arXiv 2609.16247 | Self-directed versus observed harm as a control |
| Lee et al., arXiv 2608.10030 | The persistence and anxiety-induction groups now co-author |

Not covered: PsyArXiv and OSF preprints. Repeat the check before submitting.

---

## Log

- **Nov 11** — Setup, smoke test passed, repo created (private). Three more models downloaded overnight.
- **Nov 12 morning** — Runner built and tested. Two-conversation real test exposed the 16-bit
  overflow; fixed. Arm renamed to "relaxation". Prior-work check done.
- **Nov 12, 09:58** — Pilot started. Gemma finished 12:04 (90 of 90, clean). Qwen3 8B running.
- **Nov 12 afternoon** — Gemma pilot self-reports reviewed. Questionnaire speed-up, v2.0 protocol,
  240-token check config, helper scripts and this file written. 13 tests pass.
- **Nov 12, 15:52** — Pilot finished: 180 of 180 conversations in 5 h 55 min, no errors, heat warnings or non-finite values.
- **Nov 12, 15:56** — First speed-check attempt crashed on a fallback bug in the new code (RecursionError); fixed, test added. No run data affected.
- **Nov 12, 16:05** — Speed check ran: fast method is 7.2x (Gemma) and 7.5x (Qwen3 8B) faster on the self-report step, but FAILED the preset agreement thresholds (worst total difference 1.2 points; 19–20 of 20 item answers agree; one 0-100 answer differed, 85 vs 95). Slow method reproduces stored pilot values exactly. Cause under investigation (`harness/diagnose_reuse.py`). v2.0 not started.
- **Nov 12, 17:15** — Diagnosis: sharing exact (B = C), gap is two-step rounding. Decision: fast method from v2.0 on. Noted: some items sit close to a tie between two options, so tiny arithmetic changes can move them; the 20-item total moves by at most about 1 point.
- **Nov 12, 17:19–17:53** — v2.0 run on Gemma: 75 of 75 conversations in 34.7 min (median 26 s each, against 83 s in the pilot). No errors, heat warnings or non-finite values. Both internal readings (whole reply and first 64 tokens) saved at all four checkpoints. Self-reports by arm not yet reviewed.
- **Nov 12, evening** — Self-report analysis script written and run on v2.0 and the pilot (results above). Gate 0 code written in `validation/` (texts, probe, collect/steer/judge/report, blind sheet for Day 2), 4 new tests (18 pass). Not yet run on a real model: the agent sandbox has no GPU access.
- **Nov 12, night** — Stage 1 registration drafted (`registration/stage1_osf.md`). Plan set: two-stage OSF registration, two confirmatory waves, arXiv v1 after Wave 1. Disk: 51 GB free before the Gemma 4 download.
- **Nov 12, 21:56** — Gate 0 started on Gemma 3 4B and Qwen3 8B (`scripts/gate0.sh`). Gate 0 code reviewed while it ran: no bugs that affect results; four points for Stage 2 listed under Next. Local backup of pilot data, HANDOFF.md and registration draft: `~/Desktop/lsr-backup-2026-11-12.zip`, SHA-256 `481d93920ad415fd97150e637ef869404e2946677d88f48edde149dacc3fb5c8` (keep this exact file: re-zipping gives a different hash).
- **Nov 12, 22:25** — Disk cleared: 433 GB free. Gemma 3 4B finished Gate 0 collect and steer in 15 min (layer 20 chosen); Qwen3 8B is collecting. Gemma 4 12B (`gemma4_unified`): mlx-lm 0.32.2 supports it, but its decoder layers return a tuple `(h, shared_kv, offset)`, so mlx-lens 0.1.2 capture and steer fail on it. Needs a small compatibility wrapper (act on `h` only), to be added after Gate 0 finishes so code is not changed under a running job. Repos confirmed on the Hub: gemma-3-12b-it-4bit, Qwen3-14B-4bit, gemma-3-4b-it-8bit, gemma-3-4b-it-bf16.
- **Nov 12, 22:47** — Stage 1 submitted on OSF (project https://osf.io/9u54s/, private), embargoed to 2027-11-12, approved by Jiajun. Files attached: `stage1_osf.md`, `v2.0_gemma.yaml`, `texts.py`; Summary carries the pilot archive hash. Gate 0 reports not read before filing. OSF stops new projects/components on Nov 16 2026 and makes projects read-only on Feb 19 2027; registrations unaffected, so Stage 2 is filed before Feb 19 (see HANDOFF §10).
