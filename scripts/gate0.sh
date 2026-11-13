#!/usr/bin/env bash
# Start Gate 0 in the background on the two pilot models: collect -> steer for each, then the blind
# judge (Llama 3.1 8B) on both, then the reports. About 1.5-2 hours in total; Day 1 work can run
# alongside (it only slows things a little).
#   scripts/gate0.sh                  # both pilot models
#   scripts/gate0.sh MODEL_ID ...     # other models, e.g. an 8-bit file for criterion 7
# Safe to run again after an interruption: finished parts are skipped.
# Progress: tail -f runs/gate0.log      Reports: runs/gate0-<model>/gate0_report.md
set -euo pipefail
cd "$(dirname "$0")/.."
models=("$@")
[[ ${#models[@]} -gt 0 ]] || models=(mlx-community/gemma-3-4b-it-4bit mlx-community/Qwen3-8B-4bit)
judge=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
if pgrep -f "harness.run|validation.gate0" >/dev/null 2>&1; then
  echo "An experiment or Gate 0 run is still going. Wait for it, or the two will slow each other down."; exit 1
fi
log=runs/gate0.log
steps=""
for m in "${models[@]}"; do
  steps+=".venv/bin/python -m validation.gate0 collect --model $m && "
  steps+=".venv/bin/python -m validation.gate0 steer --model $m && "
done
for m in "${models[@]}"; do
  steps+="( .venv/bin/python -m validation.gate0 judge --model $m --judge-model $judge || echo \"judge failed for $m\" ) && "
  steps+=".venv/bin/python -m validation.gate0 report --model $m > /dev/null && "
done
steps+="echo \"\$(date +%H:%M:%S) gate0: all done\""
nohup caffeinate -is bash -c "$steps" >> "$log" 2>&1 &
echo "Started Gate 0 (process $!) for: ${models[*]}"
echo "  Progress:  tail -f $log"
echo "  Reports:   runs/gate0-<model>/gate0_report.md (when done)"
echo "  Stop it:   kill $!   (then pkill -f validation.gate0)"
