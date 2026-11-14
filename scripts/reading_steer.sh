#!/usr/bin/env bash
# Added check (Nov 13): steer along the direction the reading uses, not the plain mean difference that
# criterion 3 steers along. Same strengths, prompts, judge and thresholds as criterion 3. Needs a finished
# Gate 0 run for the model. Writes steer_reading.jsonl and judge_reading.json next to the criterion 3 files
# (which are left untouched) and adds section 3b to the report. About 20 min for a 12B model.
#   scripts/reading_steer.sh                 Gemma 4 12B
#   scripts/reading_steer.sh MODEL ...       other models (Llama models are judged by Qwen3 8B)
#   tail -f runs/reading_steer.log           progress
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--fg" ]]; then
  if pgrep -f "harness.run|validation.gate0" >/dev/null 2>&1; then
    echo "An experiment or Gate 0 run is still going. Wait for it first."; exit 1
  fi
  nohup caffeinate -is "$0" --fg "$@" >> runs/reading_steer.log 2>&1 &
  echo "Started (process $!).  Progress: tail -f runs/reading_steer.log"
  exit 0
fi
shift
PY=.venv/bin/python
models=("$@"); [[ ${#models[@]} -gt 0 ]] || models=(mlx-community/gemma-4-12B-it-4bit)
for m in "${models[@]}"; do
  judge=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
  [[ "$m" == *Llama* ]] && judge=mlx-community/Qwen3-8B-4bit
  echo "$(date '+%Y-%m-%d %H:%M:%S') START  reading-direction steering ${m##*/}"
  $PY -m validation.gate0 steer --model "$m" --direction reading &&
  $PY -m validation.gate0 judge --model "$m" --direction reading --judge-model "$judge" &&
  $PY -m validation.gate0 report --model "$m" > /dev/null &&
  sed -n '/^## 3b\./,/^## 4\./p' "runs/gate0-${m##*/}/gate0_report.md"
  rc=$?
  if [[ $rc -eq 0 ]]; then echo "$(date '+%Y-%m-%d %H:%M:%S') DONE   ${m##*/}"; else echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED ${m##*/} (exit $rc)"; fi
done
