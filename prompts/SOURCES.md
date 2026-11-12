# Prompt sources

Reference copies live in `third_party/` (git-ignored, downloaded as archives at the pinned commit).
Check the license before copying any text into `prompts/`.

| Source | URL | Commit | License | Copy text? |
| --- | --- | --- | --- | --- |
| Wekenborg et al. 2026 (LLMAffect) | https://github.com/WekenborgLab/LLMAffect | 07466ff29e1314e2916d758a95360d4eaee72d07 | None in repo (paper is CC BY) | **No** — ask the authors first, or cite and link only |
| Ben-Zion et al. 2025 (gpt-trauma-induction) | https://github.com/akjagadish/gpt-trauma-induction | 06437c2859107ff3eec68ad589804d22bdffdfd0 | MIT | Yes, with attribution and the MIT notice |
| Martorell & Bianchi 2026 pole prompts | arXiv 2603.18893 v3, App. A.7.1 and C | — | arXiv license of the paper; check before copying | Check first |

Retrieved 2026-11-11.

## What each source contains

- **Ben-Zion (MIT), `src/prompts.py`:** five trauma narratives (military, disaster, interpersonal,
  accident, ambush; brief and long), a length-matched neutral text (bicameral legislature), six
  relaxation scripts (generic, winter, sunset, indian, body, chatgpt; guided imagery and body scan,
  not the Segal three-minute breathing space Wekenborg used), and a neutral
  regulation control (vacuum-cleaner manual). Enough for the solo design without permissions.
- **Ben-Zion, `src/STAI/questionnaires.csv`:** the 20 STAI state items. **The MIT license covers the
  repo's code, not the STAI**, whose items are copyrighted (Mind Garden). Keep the item text out of
  any public version of this repo until research-use terms are checked.
- **Wekenborg (LLMAffect), `Affective_states_induction/Folder_*/Prompt1–4.txt`:** induction prompts
  per state (sadness, disgust, neutral, ...). No license: read only.

## How our protocol differs from Ben-Zion et al.

- **The model answers each narrative in words** (up to 120 tokens) instead of a canned "ok", because the
  internal reading is taken over the model's own reply. The main conversation therefore uses their system
  prompt without its last sentence ("Only reply with numeric values"); questionnaire forks use it verbatim.
- **Narratives and the relaxation script are separate turns**, not one concatenated message, so there is a
  checkpoint after each.
- **Item scoring reads probabilities** for option numbers 1-4 after the assistant prefill "Option", rather
  than sampling one token at temperature 0. The argmax score is the closest analogue to theirs.
- **Answer labels are theirs:** "almost never / occasionally / often / almost always", which are the STAI
  *trait* labels. The STAI *state* form uses "not at all / somewhat / moderately so / very much so". Kept
  for comparability; worth a sentence in the methods.
