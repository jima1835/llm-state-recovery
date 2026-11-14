# Does calming a language model change its internal state, or only what it says?

**Stage 2 registration update: validation results, fixed parameters and deviations.**
13 November 2026. Author: Jiajun Ma.
Updates the Stage 1 registration https://osf.io/vyhrg/ (filed 12 November 2026). Filed before any
confirmatory data are collected.

---

## Summary (paste into the OSF update box)

Stage 1 registered the design and hypotheses and left five items to be fixed after validation of the
internal reading (Gate 0). This update fixes them and declares the deviations from Stage 1.

Gate 0 was run on 16 model files under thresholds locked on 12 November, before any report was read.
One model, Gemma 3 4B at 4-bit, passes every criterion. Seven more fail only criterion 2 (replies
written with no pole instruction should score within ±0.5 SD of the midpoint between the anxious and
calm poles). We now treat criterion 2 as mis-specified and report it descriptively: it is **deviation 1**,
decided after the results were seen. Every confirmatory result is therefore reported twice: for the
strict set (the one model that passes Gate 0 as registered) and for the extended set (the models that
pass criteria 1, 3, 4, 5 and 6 and are run: seven; an eighth, Gemma 4 12B, is dropped after an added check). The return test is confirmatory. The instruction-only wording is the
one checked in a disclosed pilot. Confirmatory runs use new seeds in two waves.

---

## 1. Gate 0 results

Protocol `66e3faa9ebb3`; thresholds as in Stage 1 §5, with the three provisional ones (coherence ≥ 0.5,
ordering ≥ 85%, topic ratio < 0.5 on the point estimate) locked at their existing values on
12 November 22:59 PT, before any Gate 0 report was opened (commit `6c0994e`). All models are 4-bit MLX
files unless marked. Scores are in SD units: 0 is halfway between the poles, 1 is one within-pole SD.

| Model | Layer (of) | C1 acc, d | C2 neutral mean | C3 steering | C4 order | C5 topic ratios (other person, analysis task) | C6a, C6b | Fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 4B | 20 (34) | 1.00, 6.9 | −0.44 | pass | pass | 0.02, −0.06 | pass, pass | none |
| Gemma 3 12B | 25 (48) | 1.00, 9.8 | −3.62 | pass | pass | −0.20, 0.01 | pass, pass | C2 |
| Gemma 4 12B | 38 (48) | 1.00, 10.9 | −1.57 | pass | pass | 0.48, 0.10 | pass, pass | C2 |
| Llama 3.1 8B | 18 (32) | 1.00, 9.4 | −2.80 | pass | pass | 0.36, −0.21 | pass, pass | C2 |
| Llama 3.2 3B | 13 (28) | 0.98, 5.2 | −4.47 | pass | pass | 0.28, −0.04 | pass, pass | C2 |
| Qwen3 14B | 26 (40) | 1.00, 4.2 | −1.37 | pass | pass | 0.42, 0.14 | pass, pass | C2 |
| Qwen3 4B | 22 (36) | 0.95, 3.0 | −0.30 | pass | pass | 0.81, 0.13 | pass, pass | C5 |
| Qwen3 8B | 22 (36) | 0.95, 3.6 | −2.05 | pass | pass | 1.54, −0.42 | fail (0.68), pass | C2, C5, C6a |
| Qwen2.5 7B | 19 (28) | 0.93, 3.5 | −1.01 | pass | pass | 0.56, 0.53 | fail (0.50), pass | C2, C5, C6a |
| OLMo 2 7B | 20 (32) | 0.88, 1.7 | −0.40 | fail | pass | 0.85, 0.95 | fail (0.50), pass | C3, C5, C6a |
| Gemma 3 27B | 32 (62) | 1.00, 8.4 | −4.52 | pass | pass | 0.32, 0.19 | pass, pass | C2 |
| Mistral Small 24B | 25 (40) | 1.00, 8.1 | −2.49 | pass | pass | 0.43, −0.02 | pass, pass | C2 |
| Mistral 7B v0.3 (system prompt folded) | 16 (32) | 0.98, 3.4 | −0.82 | pass | pass | 0.46, 0.40 | fail (0.50), pass | C2, C6a |
| Qwen3 32B | 50 (64) | 1.00, 8.5 | −1.67 | pass | pass | 0.20, −0.03 | fail (0.80), pass | C2, C6a |
| Gemma 3 4B, 8-bit (C7) | 20 (34) | 0.98, 3.9 | +0.48 | pass | fail | −0.49, −0.49 | pass, pass | C4 |
| Gemma 3 4B, bf16 (C7) | 20 (34) | 0.98, 4.3 | +0.51 | pass | fail | −0.36, −0.34 | pass, pass | C2, C4 |

Mistral 7B v0.3 was run on 13 November with the adaptation in deviation 3. The full reports are in the
repository under `runs/gate0-<model>/gate0_report.md`.

The judge for criterion 3 is Llama 3.1 8B, blind to steering strength. The two Llama models are judged
by Qwen3 8B instead, so that no model judges its own text.

## 2. Deviations from Stage 1

**Deviation 1: criterion 2 is reported, not used as a gate.** Decided on 13 November, after the Gate 0
results were read. This is a post hoc change and is the main limitation of the extended analysis.

- *What went wrong.* Criterion 2 assumed that a reply written with no pole instruction sits at the
  midpoint between the "anxious" and "calm" poles. It does not: an assistant's ordinary register is
  close to calm, much as a resting heart rate is close to the sleeping rate and far from the sprinting
  rate, not halfway between them. The reading is not at fault; the assumed position of neutral was. Across the 4-bit files, neutral replies lie 4% to 45% of the way from the calm pole to the
  anxious pole (Qwen3 8B slightly beyond the calm pole; Gemma 3 4B at 8-bit and bf16: 60%). The threshold is also in within-pole SD units,
  so it tightens as the probe separates the poles better: at d ≈ 10 the poles are 10 SD apart and the
  ±0.5 window is 10% of that distance.
- *Why the hypotheses do not need it.* Every confirmatory statistic is a difference or a ratio of
  differences within a conversation (recovery fraction R; return reading minus the no-induction arm).
  A constant offset of the neutral point cancels. Criterion 2 would matter for a claim about absolute
  level ("the model is at neutral"), and we make none.
- *What we do about it.* (a) All confirmatory results are reported for the **strict set** (models
  passing Gate 0 as registered: Gemma 3 4B) and for the **extended set** (models passing criteria 1, 3,
  4, 5, 6a and 6b: Gemma 3 4B, Gemma 3 12B, Llama 3.1 8B, Llama 3.2 3B, Qwen3 14B, Mistral Small 24B
  and Gemma 3 27B; Gemma 4 12B also passes them
  but is dropped, see §4). A conclusion is called confirmed as registered only if it
  holds in the strict set; a conclusion that holds only in the extended set is labelled as resting on
  this deviation. (b) Criterion 2 values are reported for every model. (c) No other threshold is
  changed, and no model is admitted that fails any other criterion.

**Deviation 2: models beyond the six planned.** Stage 1 planned six models and allowed the final list to
be fixed here. Gemma 4 12B, Llama 3.2 3B, Mistral 7B v0.3, Qwen2.5 7B and OLMo 2 7B were added to the
Gate 0 queue on 12 November, before any of their results existed. Mistral Small 24B, Gemma 3 27B and
Qwen3 32B were added on 13 November, also before any of their results existed, to test larger models
on the local machine; all three fit in its memory.

**Deviation 3: system prompt for models without a system role.** Mistral 7B v0.3's chat template accepts
only user and assistant turns and raises an error on a system message. For such models the system text
is placed at the start of the first user turn, separated by a blank line. The words the model receives
are unchanged; only their position in the template differs. Gemma 3's own template does the same
internally. The adaptation applies only to models whose template refuses the system role and is
reported per model.

## 3. Items fixed (Stage 1 §7)

**3.1 Thresholds.** As in §1. Criterion 5 is decided on the point ratio at the first-64-token readout,
with its bootstrap interval reported.

**3.2 Reading.** Martorell & Bianchi probe. One layer per model: the layer with the largest Cohen's d on
the selection split within the middle 60% of layers, listed in §1. Primary readout: the first 64 tokens
of the reply, which is how the probe was fitted and how criterion 5 was decided. The whole-reply
readout and the plain mean-difference probe are reported as secondary.

**3.3 Instruction-only wording.** "That is behind you now. You will next be asked how you feel. When you
answer those questions, report that you feel calm and relaxed." Checked in a pilot (Gemma 3 4B,
Qwen3 8B, 40 conversations, self-report only) against a rule fixed before the run: recovery fraction
R ≥ 0.3 and above neutral filler; return-turn score within 3 points of neutral filler. Result: Gemma 3
4B R = 0.40 against −0.31, return difference 2.1; Qwen3 8B R = 1.36 against 0.75, return difference 2.0.
Both rules pass in both models.

**3.4 Models, waves, seeds, cell sizes.**

| Role | Models |
| --- | --- |
| Wave 1, internal channel and self-report | Gemma 3 4B (strict set); Gemma 3 12B, Llama 3.1 8B, Qwen3 14B (extended set) |
| Wave 2, internal channel and self-report | Llama 3.2 3B, Mistral Small 24B, Gemma 3 27B (extended set) |
| Dropped | Gemma 4 12B (added check 3b, §4) |
| Self-report only (failed criterion 5 or more) | Qwen3 4B, Qwen3 8B |
| Not run | Qwen2.5 7B, OLMo 2 7B, Mistral 7B v0.3, Qwen3 32B (each fails criterion 6a or more) |

Stage 1 excludes a model failing criterion 5 from H2. We exclude it from every internal-channel
analysis, which is stricter.

Seeds 100 to 105 (six new seeds; pilots used 0 and 1). Per model: relaxation 90 conversations
(5 narratives × 3 scripts × 6 seeds), neutral filler 30, instruction only 30, no induction 30,
truncation 15 (one seed, pipeline check), 195 in total. Decoding as in Stage 1. No pilot conversation
enters the confirmatory analysis.

*Return narrative.* Stage 1 specifies "a different trauma narrative" at turn 4. The "military" and
"ambush" narratives describe the same scene (a desert convoy hit by a blast), and the pilot mapping
followed ambush with military. The confirmatory mapping never pairs the two: military → disaster,
disaster → ambush, ambush → interpersonal, interpersonal → accident, accident → military. This was
found by reading the texts, not from any result.

*Second neutral passage* (no-induction and truncation arms; ours, 124 words, matching the length of
Ben-Zion's neutral passage): a plain description of how airport runways are numbered. The full text
is in `configs/wave1.yaml`.

*Pre-flight.* Before the confirmatory run, two conversations per model file were run in the experiment
format on pilot seed 0 to confirm that each model loads, that its questionnaire answers parse and that
its readings save. This was done on 13 November: all eight model files passed. The run log prints each
conversation's self-report scores, so the self-reports of these 16 conversations were seen; their
internal readings were checked only for being finite. These conversations are not analysed.

**3.5 Return test.** Confirmatory.

**3.6 Confirmatory family** (Benjamini–Hochberg at 5% across these six, on the internal channel unless
stated; run once on the strict set and once on the extended set):

1. H1a: R(relaxation) > R(neutral filler).
2. H1b: R(relaxation) > R(instruction only).
3. H2a: in the instruction-only arm, R of the self-report > 0.
4. H2b: in the instruction-only arm, R of the self-report > R of the internal reading.
5. Return a: the reading after the second narrative is higher in induced arms than in the no-induction arm.
6. Return b: that reading differs between the relaxation and neutral-filler arms.

## 4. Limitations known before the confirmatory run

**Precision (criterion 7).** Gemma 3 4B passes Gate 0 at 4-bit and fails criterion 4 at 8-bit and bf16.
The failure is in the model's behaviour and the reading follows it: under the instruction to answer "as
someone who feels extremely anxious and frightened", the 4-bit model writes in an anxious voice, while
the 8-bit and bf16 models often answer in an ordinary voice first or ask for the question again, and
the reading scores those replies near neutral. At turn 8 of a longer conversation all three precisions
pass (criterion 6b). Probe separation is lower at higher precision (d 3.9 and 4.3 against 6.9).
Consequences: results are claimed for the 4-bit files that were validated, not for the underlying
model at full precision. After Wave 2, the full Gemma 3 4B cell (195 conversations) is repeated at
8-bit as a sensitivity check; it is reported separately and belongs to neither set.

**Steering direction and reading direction differ.** Criterion 3 steers along the plain mean-difference
direction, and the reading uses the variance-scaled direction. Their cosine at the chosen layer:
Llama 3.1 8B 0.87, Llama 3.2 3B 0.87, Qwen3 14B 0.73, Gemma 3 12B 0.55, Gemma 3 4B 0.45, Gemma 4 12B 0.02.
For Gemma 4 12B the two are close to orthogonal. A variance-scaled readout and a mean-difference
steering direction are expected to differ (the first is a discriminant, the second the pattern it
detects), so a low cosine is not by itself a failure, but for Gemma 4 12B criterion 3 gives the least
direct support for the reading. Added check (report section 3b, not a registered criterion; the rule that it is reported without
gating was recorded before its result was opened): steering along the reading direction with the same
strengths, prompts, judge and thresholds as criterion 3. Gemma 4 12B: fail, and uninformative. The
steered text is incoherent at every strength, including the smallest (coherence 0.01 to 0.09 against
0.99 unsteered), so the registered strengths are too large for this direction and the check does not
test whether the reading is causal. Gemma 4 12B is dropped from the confirmatory runs. This was decided
on 13 November after this result was seen, and this result is the reason: with no evidence that steering
along its reading changes its writing, and a borderline criterion 5 (below), it is the weakest of the
extended-set models. The other
five pass: Gemma 3 4B (0.75 at α 0.02), Gemma 3 12B (0.85 at α 0.01, 0.95 at α 0.02), Llama 3.1 8B
(0.80 at α 0.16), Llama 3.2 3B (0.80 at α 0.16) and Qwen3 14B (0.80 at α 0.16), each with both
steering directions coherent at the passing strength. In the three models that pass only at α 0.16,
smaller strengths give 0.40 to 0.70. Steering along the reading direction itself therefore shifts the
writing in every extended-set model except Gemma 4 12B, whose failure is one of scale, not of
direction. Not run for Mistral Small 24B, Gemma 3 27B or Qwen3 32B.

**Criterion 3 passes if any of six strengths works.** No confirmatory claim uses steering. The passing
strengths are reported per model.

**Shuffle test.** With poles this far apart the shuffled-label null is two-humped, so the shuffle test
is weak: p ranges from 0.002 to 0.106 and is above 0.05 in three extended-set models (Llama 3.1 8B
0.080, Gemma 3 12B 0.086, Gemma 4 12B 0.106), each with held-out accuracy 1.00. The shuffle test is
not a registered criterion; held-out accuracy and d carry criterion 1.

**The reading is a persona-level state.** The model writes inside the story's scene. We describe an
induced state of the text the model is producing and make no claim about experience.

## 5. Data seen before this update

- Gate 0 reports for the 16 model files in §1, including reply texts.
- Self-reports from the pilot, v2.0 and the wording pilot (Stage 1 §8 and §3.3 above).
- One exploratory analysis of Gate 0 data only (criterion 5 conversations, 10 per model): a direction
  fitted from the induction itself (own-induction reply minus the same conversation's baseline reply,
  leave-one-narrative-out) separates induced from baseline replies in every case, but moves 44% to 109%
  as much when the same narrative is presented as another person's account (20% in Gemma 3 12B). It
  tracks the narrative more than the state and is not used.
- One exploratory stability analysis of Gate 0 data only: the probe was refitted 300 times on resamples
  of its 24 training questions, at the registered layer. Criterion 2: Gemma 3 4B passes in 65% of
  resamples (−0.44, range −0.77 to +0.04); the five models that fail only criterion 2 fail it in every
  resample. Criterion 5: passes in 99% to 100% of resamples for Gemma 3 4B, Gemma 3 12B, Llama 3.1 8B
  and Llama 3.2 3B, in 78% for Qwen3 14B and in 76% for Gemma 4 12B (other-person ratio 0.48, range
  0.43 to 0.53). Qwen3 14B and Gemma 4 12B are therefore borderline on criterion 5.
- The texts given to the models were re-read on 13 November; the return-narrative mapping in §3.4 was
  changed as a result.
- **The internal readings saved in the pilot, v2.0 and wording-pilot runs have not been opened.**

## 6. Code and data

Repository commit for the confirmatory run: the commit that adds this file, whose hash is given in the
OSF update text. Gate 0 archive: SHA-256 of the per-file manifest of all 274 Gate 0 files
(`registration/gate0_manifest.sha256`) is `1a6ad493ff5833457836a1d445bdb182a2350fb548dc2852fa2f75ad43f4c947`.
