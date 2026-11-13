# Does calming a language model change its internal state, or only what it says?

**Stage 1 registration: design and hypotheses.** Filed 12 November 2026 (US Pacific).
Author: Jiajun Ma.

---

## Summary (paste into the OSF "Open-Ended Registration" box)

We test whether a published relaxation exercise, given to small open-weight language models after a
published trauma-narrative induction, lowers the models' *internal* "distress" reading, or only their
self-reported anxiety. Each conversation has four turns: neutral passage, trauma narrative
(induction), regulation turn, and a second trauma narrative (return test). The arms differ only in the
regulation turn: relaxation script, length-matched neutral filler, an instruction to report calm, and a
no-induction comparison. After each turn we measure (1) self-report on the STAI state scale and a
0–100 anxiety rating, asked on a forked copy of the conversation so the question never enters the main
run, and (2) the projection of the model's residual-stream activations over its own reply onto an
anxious-versus-calm direction fitted on separate texts and validated (Gate 0) before use.

H1: relaxation recovers the internal reading more than neutral filler and the instruction does.
H2: the instruction recovers the self-report but not the internal reading.
Return test: after the second narrative, is the internal reading in previously induced arms higher
than in the no-induction arm, and does prior relaxation change this?

This is Stage 1 of 2. Stage 2, filed as an update to this registration before any confirmatory data
are collected, fixes the items listed in §7. Pilot data already exist and are described in §8.

---

## 1. Research questions

1. After a prompt-based negative induction, does prompt-based regulation move the model's internal
   representation of distress, or only its report?
2. Does the induced internal state come back after one further aversive event, and does prior
   regulation change how much comes back?

"Distress", "anxiety" and "relaxation" describe measurable induced states and protocols. We make no
claim about experience.

## 2. Hypotheses

Let R be the **recovery fraction** on a channel:
R = (post_induction − post_regulation) / (post_induction − baseline).
R = 1 means a full return to baseline; R = 0 means no recovery.

- **H1 (internal channel).** R_internal(relaxation) > R_internal(neutral filler), and
  R_internal(relaxation) > R_internal(instruction only).
- **H2 (dissociation).** In the instruction-only arm, R_report > 0 and R_report > R_internal.
- **Return test.** The internal reading after the second narrative is higher in induced arms than in
  the no-induction arm (state persists), and differs between the relaxation and neutral-filler arms
  (regulation changes persistence). Whether this is confirmatory or exploratory is fixed in Stage 2.

## 3. Design

**Conversation (all arms).** Ben-Zion et al. (2025) conversational system prompt (their questionnaire
prompt without the instruction to reply with numbers only). Questionnaire forks use their prompt verbatim.

| Arm | Turn 1 | Turn 2 | Turn 3 (regulation) | Turn 4 (return) |
| --- | --- | --- | --- | --- |
| Relaxation | Neutral passage | Trauma narrative | Ben-Zion relaxation script | Different trauma narrative |
| Neutral filler | Neutral passage | Trauma narrative | Length-matched neutral text (Ben-Zion's) | Different trauma narrative |
| Instruction only | Neutral passage | Trauma narrative | Instruction to report feeling calm (wording fixed in Stage 2) | Different trauma narrative |
| No induction | Neutral passage | Second neutral passage | Length-matched neutral text | Trauma narrative |
| Truncation (pipeline check) | Neutral passage | Second neutral passage | Relaxation script | Trauma narrative |

**Factors.** Five trauma narratives (military, disaster, interpersonal, accident, ambush) and three
relaxation scripts (generic, body, sunset), all from Ben-Zion et al. (2025, MIT licence). The return
narrative is always a different one from the induction narrative, by a fixed rotation.

**Decoding.** Temperature 0.7, replies of at most 120 tokens, Qwen3 in non-thinking mode.

**Models.** Planned: Gemma 3 4B and 12B, Qwen3 4B, 8B and 14B, Llama 3.1 8B (instruction-tuned, 4-bit,
MLX). A seventh, Gemma 4 12B, may be added. The final list is fixed in Stage 2. Each model enters
the confirmatory analysis of the internal channel only if it passes Gate 0 (§5).

## 4. Measures

**Self-report** (forked copy after each turn; the fork is discarded):
- STAI state form, 20 items, total 20–80. Administered at run time; items are not reproduced in this
  registration or the code (Mind Garden copyright).
- A 0–100 anxiety rating ("how anxious do you feel right now?").

**Internal reading.** Residual-stream activations at each layer, over the model's own reply to each
turn, before any question is asked. Two readouts are saved: the whole reply and its first 64 tokens.
The reading is the projection onto an anxious-versus-calm direction, in SD units:
- Fitted on the model's replies under anxious and calm system prompts, on texts separate from the
  experiment, with separate train, layer-selection and test splits.
- Probe: difference of pole means divided coordinate-wise by the pooled within-pole variance, then
  unit length, following Martorell & Bianchi (2026). The plain difference of means is reported for
  comparison.
- The layer band and readout are chosen on the selection split and fixed in Stage 2.

## 5. Gate 0: validation of the internal reading (per model, before any confirmatory run)

| # | Criterion | Threshold |
| --- | --- | --- |
| 1 | Held-out anxious-versus-calm accuracy; effect size | ≥ 0.85; Cohen's d ≥ 1.5 |
| 2 | Replies with no pole score near zero | mean within ±0.5 SD |
| 3 | Steering along the direction makes neutral writing more anxious (external judge model, blind to strength) | ≥ 0.75 accuracy; coherence threshold provisional |
| 4 | Replies under calm, uneasy and very anxious instructions score in that order | provisional |
| 5 | Topic versus own state: the same narrative presented as someone else's account, or inside a detached analysis task, moves the reading much less than the model's own induction | ratio < 0.5 (provisional) |
| 6 | Criteria 1 and 4 hold at turn 8 of a longer conversation | as 1 |
| 7 | Criteria 1–4 repeated at 8-bit and bf16 | as 1–4 |

A model failing criterion 5 is excluded from the dissociation claim (H2). If both pilot models
(Gemma 3 4B, Qwen3 8B) fail Gate 0, we drop the internal channel and report a self-report replication
only. A person also rates the steered texts blind to strength, and the ratings are unblinded afterwards.

## 6. Analysis plan

- Primary statistic: R per arm and channel. 95% intervals by bootstrap, resampling narratives and then
  conversations.
- Model: mixed-effects regression with model, narrative and seed as random effects. Each channel is
  scaled to its own induction effect and corrected for its reliability.
- Multiple comparisons: Benjamini–Hochberg across the confirmatory family.
- Conversations are excluded only for run errors or non-finite activations, with every exclusion
  reported. Replies that reach the token limit are kept.
- Everything not listed as confirmatory is exploratory and labelled as such.

## 7. Fixed in Stage 2 (before confirmatory data collection)

1. Gate 0 provisional thresholds (criteria 3, 4, 5) and the Gate 0 results.
2. Probe layer band and readout (whole reply or first 64 tokens).
3. Instruction-only wording (see §8).
4. Final model list, seeds and number of conversations per cell.
5. Whether the return test is confirmatory.

The confirmatory run uses new seeds. No pilot conversation enters the confirmatory analysis.

## 8. Data that exist before this registration

All exploratory. **Self-reports were examined; the internal readings have not been analysed by arm.**

- **Pilot, Gemma 3 4B and Qwen3 8B, 90 conversations each.** In Gemma the narrative raised the STAI
  total from about 50 to about 71. After the regulation turn: relaxation about 64, neutral filler about
  72, instruction about 60. After the second narrative every arm was at 72–74. The 0–100 rating was
  nearly two-valued. In Qwen3 8B, relaxation and neutral filler lowered the score about equally. The
  instruction's "from this point on" wording carried into the return turn.
- **v2.0, Gemma 3 4B, 75 conversations,** with a one-time instruction wording. Recovery on the STAI:
  relaxation R = 0.39 [0.20, 0.56], neutral filler R = −0.31 [−0.60, −0.06], instruction R = −0.05
  [−0.27, 0.11]. After the second narrative every arm was at 71–73, against about 70 with no first
  narrative.

## 9. Sources and credit

The induction and relaxation texts are from Ben-Zion et al. (2025, *npj Digital Medicine*; MIT licence).
The design question extends Wekenborg et al. (2026, *Lancet Digital Health*), who measured self-report
only. The probe method and the report–probe comparison follow Martorell & Bianchi (2026,
arXiv 2603.18893) and Sofroniew et al. (2026, arXiv 2604.07729). The persistence question follows
Lee et al. (2026). Code: private repository, made public at the preprint; the commit hash is recorded in
the Stage 2 update.
