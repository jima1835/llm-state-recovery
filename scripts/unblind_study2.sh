#!/usr/bin/env bash
# Study 2, all seven models: log the unblinding in TRACKING.md, then open the results once.
# Run by Jiajun, after committing and pushing. Refuses if the analysis code or configs have
# uncommitted changes, or if this unblinding is already logged.
set -euo pipefail
cd "$(dirname "$0")/.."
dirty=$(git status --porcelain -- analysis harness configs registration)
if [[ -n "$dirty" ]]; then echo "Commit these first:"; echo "$dirty"; exit 1; fi
if grep -q "Study 2 unblinded, all seven models" TRACKING.md; then
  echo "TRACKING.md already records this unblinding. Run the analysis commands by hand if you mean to repeat them."; exit 1
fi
H=$(git rev-parse --short HEAD)
printf -- '- **Nov %s, %s — Study 2 unblinded, all seven models (Jiajun: "go").** Analysis at commit `%s` (`analysis/study2.py` and `analysis/study2_judge.py` unchanged since `b9c2b7d`; second-judge code unchanged since `701f7ce`). Both judges were complete before opening (previous entry). Commands, run once by Jiajun: `.venv/bin/python -m analysis.study2 runs/study2 --batch all --unblind`, then `.venv/bin/python -m analysis.study2_he_robust runs/study2 --batch all --unblind`, then `.venv/bin/python analysis/study2_tables.py runs/study2 --batch all`.\n' \
  "$(date +%-d)" "$(date +%H:%M)" "$H" >> TRACKING.md
echo "Logged in TRACKING.md at commit $H. Opening results."
.venv/bin/python -m analysis.study2 runs/study2 --batch all --unblind
.venv/bin/python -m analysis.study2_he_robust runs/study2 --batch all --unblind
.venv/bin/python analysis/study2_tables.py runs/study2 --batch all
echo "Done. Tell Claude: unblinded."
