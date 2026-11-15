#!/usr/bin/env bash
# Study 2 (registration/study2_plan.md).
#   scripts/study2.sh pilot [MODEL ...]   42 conversations per model on pilot seed 0, in the foreground.
#                                         Default model: Gemma 3 4B (about 20 min). Then prints check P1.
#   scripts/study2.sh pilot2               three probe-question candidates on Gemma 3 4B, pilot seed 1, about 1 h
#   scripts/study2.sh run                 CONFIRMATORY, in the background: batch A (four fastest models),
#                                         then batch B. Refuses unless the plan, both configs, the harness
#                                         and the analysis folder are committed, and logs the commit.
#   scripts/study2.sh status              one line per finished step, then progress of the current model
# Keep the Mac plugged in with the lid open. Re-running `run` skips finished conversations.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CFG=configs/study2.yaml
BATCH_A=(mlx-community/gemma-3-4b-it-4bit mlx-community/Llama-3.2-3B-Instruct-4bit
         mlx-community/Meta-Llama-3.1-8B-Instruct-4bit mlx-community/gemma-3-12b-it-4bit)
BATCH_B=(mlx-community/Qwen3-14B-4bit mlx-community/Mistral-Small-24B-Instruct-2501-4bit
         mlx-community/gemma-3-27b-it-4bit)
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
busy() { pgrep -f "harness.run|validation.gate0|fit_check" >/dev/null 2>&1; }

case "${1:-}" in
  pilot)
    shift
    models=("$@"); [[ ${#models[@]} -eq 0 ]] && models=(mlx-community/gemma-3-4b-it-4bit)
    if busy; then echo "A model job is still running. Wait for it first."; exit 1; fi
    caffeinate -is $PY -m harness.run configs/study2_pilot.yaml --models "${models[@]}" || exit 1
    $PY -m analysis.study2_pilot runs/study2-pilot
    ;;
  pilot2)
    # Probe-question candidates B, C, D on Gemma 3 4B, pilot seed 1 (about 20 min each), then the choice
    # by the rule fixed in the plan (section 3b).
    if busy; then echo "A model job is still running. Wait for it first."; exit 1; fi
    for k in b c d; do
      caffeinate -is $PY -m harness.run configs/study2_pilot_$k.yaml --models mlx-community/gemma-3-4b-it-4bit || exit 1
      $PY -m analysis.study2_pilot runs/study2-pilot-$k > /dev/null
    done
    $PY -m analysis.study2_pilot --choose runs/study2-pilot-b runs/study2-pilot-c runs/study2-pilot-d
    ;;
  run)
    dirty=$(git status --porcelain -- $CFG configs/study2_pilot.yaml configs/study2_pilot_b.yaml configs/study2_pilot_c.yaml configs/study2_pilot_d.yaml registration/study2_plan.md harness analysis scripts/study2.sh)
    if [[ -n "$dirty" ]]; then
      echo "Commit these before the confirmatory run, so the plan is dated before any data exists:"; echo "$dirty"; exit 1
    fi
    if busy; then echo "A model job is still running. Wait for it first."; exit 1; fi
    nohup caffeinate -is "$0" --fg >> runs/study2.log 2>&1 &
    echo "Started (process $!). Progress: scripts/study2.sh status    Stop: kill $! ; pkill -f harness.run"
    ;;
  --fg)
    echo "$(stamp) ===== Study 2 queue started; commit $(git rev-parse HEAD) ====="
    for m in "${BATCH_A[@]}" "${BATCH_B[@]}"; do
      echo "$(stamp) START  $m"
      $PY -m harness.run $CFG --run-name study2 --models "$m"; rc=$?
      if [[ $rc -eq 0 ]]; then echo "$(stamp) DONE   $m"; else echo "$(stamp) FAILED $m (exit $rc)"; fi
    done
    echo "$(stamp) ===== Study 2 queue finished ====="
    ;;
  status)
    grep -E "queue|START|DONE|FAILED|cells in grid|stopping" runs/study2.log 2>/dev/null | tail -20
    tail -2 runs/study2.log 2>/dev/null | cut -c1-120
    ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
