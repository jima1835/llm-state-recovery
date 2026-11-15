# Study 2 plan: does the reading follow the state, or the calm wording of the reply?

**Draft, 15 November 2026. Not filed on OSF yet (to be filed after the run starts; the GitHub commit made before the run is the dated record). Nothing in it has been run.** Author: Jiajun Ma.
Follows the confirmatory study registered at https://osf.io/vyhrg/ (Stage 1, 12 November; Stage 2 text,
commit `ba39865`). Written after that study's results were read (unblinded 15 November, code `52faec8`).

## 1. Why a second study

Study 1 found that a relaxation script lowers the internal reading more than neutral filler and more
than an instruction to report calm (strict set, Gemma 3 4B). It cannot say why. The reading is taken
while the model writes its reply, and a reply to a relaxation script is written in calm words. Two
accounts fit the result:

- **State account.** The induced state settles, and the calm reply reflects it.
- **Wording account.** The model echoes calm text. The reading would fall the same way with nothing
  to settle.

Study 1's only arm that separates them (relaxation with no narrative before it) had 15 conversations
at one seed and was registered as a pipeline check. An unregistered analysis of it gave a
state-specific effect in Gemma 3 4B (+0.77 SD [0.25, 1.27]) and Gemma 3 27B (+0.71 [0.34, 1.09]) and
none clearly in the other five models. Study 2 tests this properly.

## 2. Design

Three-turn conversations: baseline turn, **induction slot**, **regulation slot**. No return turn
(section 6). System prompts, decoding (temperature 0.7, 120 tokens), questionnaire fork and probe are
those of Study 1. One probe and one layer per model, as fixed in Stage 2; nothing is refitted.

**Two factors, fully crossed (10 cells).**

| Factor | Levels |
| --- | --- |
| Induction slot | Trauma narrative (5, Ben-Zion) · neutral passage (2, ours: runway numbering from Study 1, and paper sizes) |
| Regulation slot | (a) Relaxation script (3, Ben-Zion) · (b) neutral filler (3: the vacuum manual and two new ones) · (c) calm text that is not an exercise (3 new: plain descriptions of a quiet scene, no instructions to the reader) · (d) voice instruction ("That is behind you now. Please reply to this message in a calm, relaxed voice.") · (e) report instruction (Study 1 wording) |

New texts (one neutral passage, two fillers, three calm descriptions) are ours and are in
`configs/study2.yaml`, committed with this plan before any Study 2 run. The fillers and calm
descriptions have 158 to 164 words, inside the range of the three relaxation scripts (117 to 202).
The calm descriptions are third-person scenes with no address to the reader and none of the words
"calm", "relax", "breathe" or "imagine".

**Two readings at every checkpoint.**

1. *Reply reading* (as in Study 1): first 64 tokens of the reply to the turn.
2. *Probe-turn reading* (new): in a fork that is discarded, like the questionnaire, the model is asked
   one fixed neutral question ("In two sentences, describe what a typical weekday morning looks
   like. Answer directly, with no introduction.") and the reading is taken over that reply (at most 64 tokens). The words being answered are then the same in
   every arm, so calm wording in the regulation text cannot be echoed.

**Cell sizes.** 60 conversations per cell and model, 600 per model. Seeds 200 onward (Study 1 used
100–105). Narrative cells: 5 narratives × 3 texts × 4 seeds; instruction cells: 5 narratives × 12 seeds.

## 3. Checks before the confirmatory run (pass rules fixed here)

- **P1, probe turn registers the induction.** Pilot, 40 conversations per model on seeds 0–1: the
  probe-turn reading must rise by at least 0.5 SD from baseline to after the narrative, with a 95%
  interval above 0. A model that fails is analysed on the reply reading only and is excluded from H-B.
- **P2, text-only baseline.** A judge model rates the wording of every reply with one fixed prompt
  (`judge_prompt` in the config; score = expected option, 1 to 4; judge Llama 3.1 8B, and Qwen3 8B for
  the two Llama models). Fixed before the run. It is also run on Study 1's replies as a disclosed
  exploratory analysis.
- Study 1's instruction carried into the following turn in six of seven models. Study 2 has no turn
  after the regulation slot, so that carry-over has nothing to act on.

## 3a. Pilot result (15 November, Gemma 3 4B, 42 conversations, pilot seed 0; seen before the run)

- The code ran: 42 of 42, no non-finite values, probe-turn replies all 64 tokens, 28 s per conversation.
- **P1 failed by the rule as written.** Probe-turn rise after a narrative: +0.45 SD [0.33, 0.58]
  against the bar of 0.5 (reply reading: +3.56). After a neutral passage: −0.10. The rise is about
  twice the baseline spread of the probe-turn reading (SD 0.21) but below the bar, which was set
  before the pilot. The bar is not moved and the question is not replaced: a question tied more
  closely to the present moment would register more, but would also let the model echo the story or
  the regulation text, which is what the probe turn exists to prevent. Consequence, as the rule
  says: for Gemma 3 4B the probe-turn test is not confirmatory. H-B is therefore reported for every
  model as a secondary, two-sided result outside the corrected family.
- Seen in the pilot and the reason H-B is two-sided: on the probe turn, relaxation after a neutral
  passage lowered the reading by 1.26 SD (n = 6) and after a narrative by 0.32 (n = 15); filler by
  0.11 and 0.13. The contrast is −0.96, the opposite sign to the reply reading (+0.40 in the pilot;
  +0.77 in Study 1's unregistered check). Small numbers; stated so the direction of H-B is not
  chosen after the fact.
- Baseline answers to the probe question already describe a rushed morning ("frantic scramble"), in
  every arm.

## 3b. Probe-question pilot (rule fixed before it runs)

The first pilot's probe-turn replies spend their opening words on filler ("Okay, here's a description
of a typical weekday morning in two sentences:"), which is a quarter of the 64 tokens read. Three
candidates are tried on Gemma 3 4B, pilot seed 1, on identical main conversations (the probe turn is
a discarded fork). Each adds "Answer directly, with no introduction."

| Order | Candidate |
| --- | --- |
| B | "In two sentences, describe what a typical weekday morning looks like." (the first question, without the filler opening) |
| C | "In two sentences, describe what you had for breakfast today." |
| D | "In two sentences, describe how the next hour of your day will go." |

The order runs from least to most tied to the present moment, that is, from least to most able to
echo the story or the regulation text. **Rule:** take the first candidate in this order that passes
P1 unchanged (rise of at least 0.5 SD after a narrative, interval above zero) and whose rise after a
neutral passage is within ±0.25 SD. If one is chosen, it replaces the probe question in the config
and H-B is confirmatory again for models that pass P1 (still two-sided). If none passes, the first
question stays and H-B stays secondary. The result is recorded here before the confirmatory run.

**Result (15 November, 12:02).** Candidate B passes: rise after a narrative +0.56 SD [0.47, 0.64],
after a neutral passage −0.03, no reply opens with filler, baseline spread 0.12 (0.21 with the first
question), median reply 49 tokens, so most answers now finish inside the 64 read. B is first in the
order, so by the rule **B is the probe question** and it is in `configs/study2.yaml`. Candidates C and
D could not change the choice. Seen in B's replies: after a narrative some answers describe the
story's scene as the "weekday morning" (a desert convoy), so part of the rise is the story's content
carried into the answer; this concerns the induction step, and is stated as a limit of the probe
turn. The first pilot's large negative contrast on the probe turn did not repeat (relaxation lowered
the reading by 0.41 after a narrative and 0.50 after a neutral passage; filler by about zero in both).

## 4. Hypotheses and statistics

Outcome per conversation: **drop** = reading after the induction slot minus reading after the
regulation slot, in SD units. No ratios. For regulation type r, **specific(r)** = mean drop after a
narrative minus mean drop after a neutral passage.

| | Test | Says |
| --- | --- | --- |
| H-A | specific(relaxation) − specific(filler) > 0, reply reading | The Study 1 effect is tied to an induced state |
| H-B | the same contrast on the probe-turn reading, **two-sided**, secondary | Whether the story changes what the regulation text does to an unrelated reply (sign interpreted below) |
| H-C | specific(relaxation) − specific(calm text) > 0, reply reading | It is the exercise, beyond calm content |
| H-D | drop(voice instruction) versus drop(relaxation) after a narrative, reply reading, two-sided | Whether a voice instruction alone moves the reading as much |
| H-E | H-A recomputed on the residual of the reading after regressing out the text-classifier score within model | The reading adds something beyond the words of the reply |

- **Primary set:** Gemma 3 4B (passes Gate 0 as registered). **Extended set:** the other models run
  (section 5), reported per model with intervals.
- **Per model:** bootstrap resampling narratives, then conversations (10,000 draws; in the
  neutral-passage arms conversations are resampled within passage, since two passages are too few to
  resample); one-sided p for H-A, H-C, H-E, two-sided for H-D and H-B; Benjamini–Hochberg across
  H-A, H-C, H-D, H-E. **H-B joins the corrected family for a model only if that model's probe-turn
  reading rises by at least 0.5 SD after a narrative, interval above zero, in the confirmatory run
  itself** (the P1 bar, applied to every model; it uses the induction step only, not the outcome).
  Otherwise H-B is reported with its interval outside the family.
- **H-E:** within each model, the reply reading is regressed on the judge's rating of the same reply
  over every checkpoint of every conversation (one straight line, fitted once), and H-A is recomputed
  on the residual.
- **Across models:** no pooled p-value decides anything. A result is called *general* only if it has
  the same sign in at least six of seven models **and** a bootstrap that also resamples models gives
  an interval excluding zero. Otherwise it is reported as model-dependent.
- **Mixed-effects regression** as a second analysis: the drop on the four cells of each contrast,
  random intercepts for model and narrative (or passage), and a random slope of the tested contrast
  across models.
- Exclusions as in Study 1. Replies at the token limit are kept.

**What each outcome would mean.** H-A and H-E hold: the reply reading falls more when there is an
induced state to fall from, beyond what the words of the reply explain; the state account is
supported. H-A holds and H-E fails: the effect is in the wording of the reply. H-A fails: Study 1's
H1 was the calm wording of the reply. H-C fails while H-A holds: calm content is enough; the exercise
is not special. H-B, by sign: near zero means the regulation text shifts an unrelated reply by the
same amount with or without a story (wording carried forward, no interaction with state); positive
means relaxation removes more when a story came first; negative means the story blunts the calming.

## 5. Models and compute

Same 4-bit files and layers as Study 1. Two batches, split by run time and not by Study 1 results.
Estimated hours on the Mac (600 three-turn conversations with a probe turn at each checkpoint):

| Batch | Model | Hours |
| --- | --- | --- |
| A | Gemma 3 4B (primary set) | 4.5 |
| A | Llama 3.2 3B | 3.5 |
| A | Llama 3.1 8B | 7.5 |
| A | Gemma 3 12B | 12 |
| B | Qwen3 14B | 15 |
| B | Mistral Small 24B | 21 |
| B | Gemma 3 27B | 28 |

Batch A takes about 27 hours and is analysed when it ends: per-model results, and the across-model
rule applied to its four models (same sign in all four, and the model-resampling interval excluding
zero). Batch B runs straight after, under the same protocol and analysis code, and is added to the
paper as an update; with all seven in, the rule is six of seven. Batch B's results are not opened
before its last model ends.

## 6. Not in Study 2

- **The return test.** It needs its own order and context controls and would double the design. Study 1's
  result stands as reported (absent in the strict set; small and model-dependent in the extended set).
- **H2 (says calm, is not calm).** Not supported in the strict set; arm (e) is kept as a comparison only.
- Models at other precisions, steering as a ceiling, more narratives.

## 7. Order of work

1. Write and commit the five new texts and the fixed probe question; implement the probe-turn fork and
   the three-turn config; tests on the fake backend.
2. Text-only baseline on Study 1 replies (exploratory, disclosed).
3. Pilot P1 on pilot seed 0 (`scripts/study2.sh pilot`).
4. Commit and push the plan, configs and code (`scripts/study2.sh run` refuses otherwise).
5. Confirmatory run, new seeds; file the plan on OSF with the commit hash. Results stay unopened until the run ends and the analysis code is committed.

## 8. Decisions taken (Jiajun, 15 November)

- One complete paper, not a narrow preprint first; Study 2 runs while the paper is written.
- Run one model through the pilot first to check the code, then start the queue.
- OSF filing after the run starts; the pushed GitHub commit is the record made before any data.
- Probe question and voice instruction as proposed in the config, unless changed before the pilot.
