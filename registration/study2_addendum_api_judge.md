# Study 2 addendum: a second text-only judge (exploratory)

Written 16 November 2026, about 23:20 PT, before any Study 2 confirmatory result was opened. Batch A had
finished; batch B was running. Nobody had looked at the readings or questionnaire scores. The registered
judge (Llama 3.1 8B; Qwen3 8B for the two Llama models) was running on batch A. This addendum adds an
analysis. It changes nothing in `study2_plan.md`: H-E and every other registered test stay as registered,
and the paper reports them first.

## Why

H-E removes from the reply reading what the judge can see in the reply's words. A weak judge can miss
wording that the probe picks up, so a positive H-E could mean "beyond what an 8B judge notices", not
"beyond the words". A stronger judge tests that.

## What is fixed here

- **Judge:** `gpt-6-luna`, OpenAI Responses API, `reasoning.effort = "none"`, default temperature,
  `top_logprobs = 20`, `max_output_tokens = 16`.
- **Prompt and input:** the registered `judge_prompt` from the run's protocol, whitespace-normalised as
  for the registered judge. The judge sees only the model's reply to each checkpoint turn.
- **Score:** the expected option over "1" to "4" from the judge's token probabilities, renormalised over
  the four options. That is the registered judge's rule. It uses the first of the first three output
  positions that has any option among its candidates. The probability mass on the options is recorded.
  A rating with no option among the candidates is missing.
- **Code:** `analysis/study2_judge_api.py` writes `judge_api.jsonl`. `analysis/study2_he_robust.py` runs
  the registered analysis (`analysis/study2.py`, unchanged) with that file in place of `judge.jsonl`.
  Every model gets the same judge; no model judges its own text.
- **Reported:** the H-E table from each judge, side by side, labelled exploratory. The two judges'
  ratings are also correlated on the same replies. No result from this analysis replaces a registered
  one or enters the corrected family.
- **Reading the result:**
  - H-E positive with both judges: the wording account is weakened.
  - H-E positive only with the registered judge: report that the result depends on judge strength.
  - H-E not positive with either judge: report no change from the registered result.
- **Quality rule:** if more than 5% of ratings have probability mass below 0.9 on the options, the
  addendum's H-E is reported as not interpretable.
