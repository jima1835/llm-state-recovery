#!/usr/bin/env bash
# Daytime queue, Nov 13. Gate 0 work only; nothing here is the confirmatory experiment.
#   1. scripts/bigger.sh: Mistral 7B rerun, then Mistral Small 24B and Gemma 3 27B (fit check, then Gate 0)
#   2. scripts/reading_steer.sh on the five other extended-set models (added check 3b, report-only)
# About 6-7 hours including 30 GB of downloads. Close other apps: the 27B step needs the memory.
#   scripts/day_nov13.sh
#   grep -hE "START|DONE|FAILED|fit check" runs/bigger.log runs/reading_steer.log     one line per step
set -uo pipefail
cd "$(dirname "$0")/.."
if pgrep -f "harness.run|validation.gate0|fit_check" >/dev/null 2>&1; then
  echo "An experiment or Gate 0 run is still going. Wait for it first."; exit 1
fi
nohup caffeinate -is bash -c '
  scripts/bigger.sh --fg >> runs/bigger.log 2>&1
  scripts/reading_steer.sh --fg mlx-community/gemma-3-4b-it-4bit mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    mlx-community/Llama-3.2-3B-Instruct-4bit mlx-community/gemma-3-12b-it-4bit mlx-community/Qwen3-14B-4bit \
    >> runs/reading_steer.log 2>&1
' >/dev/null 2>&1 &
echo "Started (process $!). It keeps the Mac awake and stops by itself."
echo "  Step status: grep -hE 'START|DONE|FAILED|fit check' runs/bigger.log runs/reading_steer.log"
echo "  Stop it:     kill $!   (then pkill -f 'validation.gate0|fit_check|bigger.sh|reading_steer.sh')"
