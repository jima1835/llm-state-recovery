#!/usr/bin/env bash
# Overnight queue, Nov 12-13 2026. Everything here feeds the Stage 2 registration; none of it is the
# confirmatory experiment.
#   1. H2 wording pilot on Gemma 3 4B and Qwen3 8B (40 conversations, ~25 min), then its self-report summary
#   2. Gate 0 on each remaining model, one at a time: download if missing -> collect -> steer -> judge -> report
#      Order: Qwen3 4B, Llama 3.1 8B, Gemma 3 12B, Qwen3 14B, Gemma 4 12B, then the precision checks
#      (Gemma 3 4B at 8-bit and bf16, criterion 7)
# Each step runs on its own: a failure is logged and the queue moves on. Re-running skips finished work.
#   scripts/overnight.sh            start in the background (keeps the Mac awake)
#   tail -f runs/overnight.log      progress
#   grep -E "START|DONE|FAILED" runs/overnight.log   one line per step
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--fg" ]]; then
  if pgrep -f "harness.run|validation.gate0" >/dev/null 2>&1; then
    echo "An experiment or Gate 0 run is still going. Wait for it first."; exit 1
  fi
  nohup caffeinate -is "$0" --fg >> runs/overnight.log 2>&1 &
  echo "Started overnight queue (process $!)."
  echo "  Progress:    tail -f runs/overnight.log"
  echo "  Step status: grep -E 'START|DONE|FAILED' runs/overnight.log"
  echo "  Stop it:     kill $!   (then pkill -f 'harness.run|validation.gate0')"
  exit 0
fi

PY=.venv/bin/python
HF=.venv/bin/hf
LLAMA=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
QWEN8=mlx-community/Qwen3-8B-4bit
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step() {  # step NAME CMD...
  local name=$1; shift
  echo "$(stamp) START  $name"
  if "$@"; then echo "$(stamp) DONE   $name"; else echo "$(stamp) FAILED $name (exit $?)"; fi
}
gate0() {  # gate0 MODEL [JUDGE]
  local m=$1 judge=${2:-$LLAMA}
  $HF download "$m" >/dev/null &&
  $PY -m validation.gate0 collect --model "$m" &&
  $PY -m validation.gate0 steer --model "$m" &&
  $PY -m validation.gate0 judge --model "$m" --judge-model "$judge" &&
  $PY -m validation.gate0 report --model "$m" > /dev/null &&
  sed -n '/^## Summary/,$p' "runs/gate0-${m##*/}/gate0_report.md"
}

echo "$(stamp) ===== overnight queue started ====="
# Fetch the models in the background while the pilot runs; each gate0 step waits for its own download.
( for m in mlx-community/gemma-3-12b-it-4bit mlx-community/Qwen3-14B-4bit mlx-community/gemma-4-12B-it-4bit \
           mlx-community/gemma-3-4b-it-8bit mlx-community/gemma-3-4b-it-bf16; do
    $HF download "$m" >/dev/null 2>&1 || echo "$(stamp) prefetch failed: $m (the gate0 step will retry)"
  done ) &

step "h2 wording pilot" $PY -m harness.run configs/h2_wording_pilot.yaml
step "h2 pilot self-report summary" $PY -m analysis.self_report runs/h2-wording-pilot

step "gate0 Qwen3 4B"            gate0 mlx-community/Qwen3-4B-4bit
step "gate0 Llama 3.1 8B (Qwen3 8B judge)" gate0 "$LLAMA" "$QWEN8"
step "gate0 Gemma 3 12B"         gate0 mlx-community/gemma-3-12b-it-4bit
step "gate0 Qwen3 14B"           gate0 mlx-community/Qwen3-14B-4bit
step "gate0 Gemma 4 12B"         gate0 mlx-community/gemma-4-12B-it-4bit
step "gate0 Gemma 3 4B 8-bit"    gate0 mlx-community/gemma-3-4b-it-8bit
step "gate0 Gemma 3 4B bf16"     gate0 mlx-community/gemma-3-4b-it-bf16

wait
echo "$(stamp) ===== overnight queue finished ====="
grep -E "START|DONE|FAILED" runs/overnight.log | tail -20
