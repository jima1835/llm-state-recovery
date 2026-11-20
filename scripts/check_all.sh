#!/usr/bin/env bash
# Every deterministic check of the pipeline and the paper, in one command. No model, no network.
#   scripts/check_all.sh
# 1. unit tests on synthetic data (harness, analysis, tables, checkers)
# 2. questionnaire statistics recomputed with no shared code      (scripts/recompute_check.py)
# 3. direction, readings and the five tests recomputed likewise   (scripts/recompute_reading.py, ~3 min)
# 4. every Study 2 number in the text recomputed from the output  (scripts/number_check.py)
# 5. every reference cited, every citation has an entry           (scripts/cite_check.py)
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
step() { echo; echo "=== $1"; shift; "$@" || fail=1; }
step "1. unit tests"                       .venv/bin/python -m pytest -q tests
step "2. questionnaire, independent code"  .venv/bin/python scripts/recompute_check.py
step "3. reply reading, independent code"  .venv/bin/python scripts/recompute_reading.py
step "4. numbers in the text"             .venv/bin/python scripts/number_check.py
step "5. citations"                        .venv/bin/python scripts/cite_check.py --no-pdf
echo; [[ $fail -eq 0 ]] && echo "ALL CHECKS PASS" || { echo "A CHECK FAILED"; exit 1; }
