#!/usr/bin/env bash
# Overnight queue, Nov 12-13 2026. Everything here feeds the Stage 2 registration; none of it is the
# confirmatory experiment.
#   1. H2 wording pilot on Gemma 3 4B and Qwen3 8B (40 conversations, ~25 min), then its self-report summary
#   2. Gate 0 on each remaining model, one at a time: download if missing -> collect -> steer -> judge -> report
#      Order: the planned six, then Gemma 4 12B, Llama 3.2 3B and Mistral 7B, then the precision checks
#      (Gemma 3 4B at 8-bit and bf16, criterion 7), then any optional extras switched on in section D.
#      Extras are fetched by their own gate0 step, not by the background prefetch.
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
  local name=$1 rc; shift
  echo "$(stamp) START  $name"
  "$@"; rc=$?
  if [[ $rc -eq 0 ]]; then echo "$(stamp) DONE   $name"; else echo "$(stamp) FAILED $name (exit $rc)"; fi
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
           mlx-community/Llama-3.2-3B-Instruct-4bit mlx-community/Mistral-7B-Instruct-v0.3-4bit \
           mlx-community/gemma-3-4b-it-8bit mlx-community/gemma-3-4b-it-bf16; do
    $HF download "$m" >/dev/null 2>&1 || echo "$(stamp) prefetch failed: $m (the gate0 step will retry)"
  done ) &

step "h2 wording pilot" $PY -m harness.run configs/h2_wording_pilot.yaml
step "h2 pilot self-report summary" $PY -m analysis.self_report runs/h2-wording-pilot

# Priority order: what Stage 2 needs first. Llama-family models are judged by Qwen3 8B, not by Llama.
# A. The planned six (two done Nov 12)
step "gate0 Qwen3 4B"            gate0 mlx-community/Qwen3-4B-4bit
step "gate0 Llama 3.1 8B (Qwen3 8B judge)" gate0 "$LLAMA" "$QWEN8"
step "gate0 Gemma 3 12B"         gate0 mlx-community/gemma-3-12b-it-4bit
step "gate0 Qwen3 14B"           gate0 mlx-community/Qwen3-14B-4bit
# B. Added candidates (approved Nov 12, before any of their results)
step "gate0 Gemma 4 12B"         gate0 mlx-community/gemma-4-12B-it-4bit
step "gate0 Llama 3.2 3B (Qwen3 8B judge)" gate0 mlx-community/Llama-3.2-3B-Instruct-4bit "$QWEN8"
step "gate0 Mistral 7B v0.3"     gate0 mlx-community/Mistral-7B-Instruct-v0.3-4bit
# C. Criterion 7: the same model (Gemma 3 4B) at higher precision
step "gate0 Gemma 3 4B 8-bit"    gate0 mlx-community/gemma-3-4b-it-8bit
step "gate0 Gemma 3 4B bf16"     gate0 mlx-community/gemma-3-4b-it-bf16
# D. Optional extras: remove the leading "# " to switch one on (only before starting the queue)
 step "gate0 Qwen2.5 7B"          gate0 mlx-community/Qwen2.5-7B-Instruct-4bit
# step "gate0 Llama 3.2 1B (Qwen3 8B judge)" gate0 mlx-community/Llama-3.2-1B-Instruct-4bit "$QWEN8"
# step "gate0 Gemma 3 1B"          gate0 mlx-community/gemma-3-1b-it-4bit
# step "gate0 Phi-4-mini"          gate0 mlx-community/Phi-4-mini-instruct-4bit
 step "gate0 OLMo 2 7B"           gate0 mlx-community/OLMo-2-1124-7B-Instruct-4bit

wait
echo "$(stamp) ===== overnight queue finished ====="
grep -E "START|DONE|FAILED" runs/overnight.log | tail -20
