#!/usr/bin/env bash
# Nov 13 queue: the Mistral 7B rerun, then two larger models to see what this 24 GB Mac can carry.
# Gate 0 only; none of it is the confirmatory experiment.
#   1. Gate 0 Mistral 7B v0.3 (failed overnight: its chat template refuses a system message; the backend
#      now folds the system prompt into the first user turn for such templates)        about 35 min
#   2. Mistral Small 24B, 13 GB of weights: download, fit check, Gate 0                 about 1.7 h + download
#   3. Gemma 3 27B, 17 GB of weights: download, fit check, Gate 0                       about 2 h + download
#      Step 3 is the stretch: it may not fit. The fit check stops it within minutes if the model is being
#      paged to disk (under 3 tokens per second), before the long run starts. Close other apps first.
# Each step runs on its own: a failure is logged and the queue moves on. Re-running skips finished work.
#   scripts/bigger.sh               start in the background (keeps the Mac awake)
#   scripts/bigger.sh MODEL ...     the same for other models (fit check, then Gate 0, for each)
#   tail -f runs/bigger.log         progress
#   grep -E "START|DONE|FAILED|fit check" runs/bigger.log    one line per step
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--fg" ]]; then
  if pgrep -f "harness.run|validation.gate0" >/dev/null 2>&1; then
    echo "An experiment or Gate 0 run is still going. Wait for it first."; exit 1
  fi
  nohup caffeinate -is "$0" --fg "$@" >> runs/bigger.log 2>&1 &
  echo "Started (process $!)."
  echo "  Progress:    tail -f runs/bigger.log"
  echo "  Step status: grep -E 'START|DONE|FAILED|fit check' runs/bigger.log"
  echo "  Stop it:     kill $!   (then pkill -f 'harness.run|validation.gate0|fit_check')"
  exit 0
fi
shift

PY=.venv/bin/python
HF=.venv/bin/hf
JUDGE=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step() {  # step NAME CMD...
  local name=$1 rc; shift
  echo "$(stamp) START  $name"
  "$@"; rc=$?
  if [[ $rc -eq 0 ]]; then echo "$(stamp) DONE   $name"; else echo "$(stamp) FAILED $name (exit $rc)"; fi
}
gate0() {  # gate0 MODEL
  local m=$1
  $HF download "$m" >/dev/null &&
  $PY -m validation.gate0 collect --model "$m" &&
  $PY -m validation.gate0 steer --model "$m" &&
  $PY -m validation.gate0 judge --model "$m" --judge-model "$JUDGE" &&
  $PY -m validation.gate0 report --model "$m" > /dev/null &&
  sed -n '/^## Summary/,$p' "runs/gate0-${m##*/}/gate0_report.md"
}
big() {  # big MODEL: download, check that it runs at a usable speed, then Gate 0
  local m=$1
  $HF download "$m" >/dev/null && $PY scripts/fit_check.py "$m" && gate0 "$m"
}

echo "$(stamp) ===== queue started ====="
if [[ $# -gt 0 ]]; then
  for m in "$@"; do step "fit check + gate0 ${m##*/}" big "$m"; done
else
  step "gate0 Mistral 7B v0.3"                gate0 mlx-community/Mistral-7B-Instruct-v0.3-4bit
  step "fit check + gate0 Mistral Small 24B"  big mlx-community/Mistral-Small-24B-Instruct-2501-4bit
  step "fit check + gate0 Gemma 3 27B"        big mlx-community/gemma-3-27b-it-4bit
fi
echo "$(stamp) ===== queue finished ====="
grep -E "START|DONE|FAILED|fit check" runs/bigger.log | tail -20
