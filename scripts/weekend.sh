#!/usr/bin/env bash
# Weekend queue, Nov 13-15: Wave 1, then Wave 2. CONFIRMATORY: start it only after the Stage 2 update is
# filed on OSF. Every step uses one config file (configs/wave1.yaml); only the model list changes, so
# the protocol is identical.
#   Wave 1  Gemma 3 4B, Llama 3.1 8B, Gemma 3 12B, Qwen3 14B                 about 12.5 h
#   Wave 2  Llama 3.2 3B, Mistral Small 24B, Gemma 3 27B                     about 22-28 h
#   Last    Gemma 3 4B at 8-bit (sensitivity check)                          about 2 h
# Wave 1 goes first: it holds the strict-set model, so if anything breaks later the main result is in.
# Keep the Mac plugged in with the lid open: caffeinate cannot stop sleep when the lid is closed.
# Re-running skips finished conversations, so after a stop or a restart just start it again.
#   scripts/weekend.sh filed     start in the background
#   grep -E "START|DONE|FAILED|cells in grid|stopping" runs/weekend.log      one line per step
#   tail -f runs/weekend.log                                                 progress
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/wave1.yaml
WAVE1=(mlx-community/gemma-3-4b-it-4bit mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
       mlx-community/gemma-3-12b-it-4bit mlx-community/Qwen3-14B-4bit)
WAVE2=(mlx-community/Llama-3.2-3B-Instruct-4bit
       mlx-community/Mistral-Small-24B-Instruct-2501-4bit
       mlx-community/gemma-3-27b-it-4bit)

if [[ "${1:-}" != "--fg" ]]; then
  if [[ "${1:-}" != "filed" ]]; then
    echo "This is the confirmatory run. Start it only after Stage 2 is filed on OSF:  scripts/weekend.sh filed"; exit 1
  fi
  [[ -f $CFG ]] || { echo "$CFG not found: it is written when Stage 2 is finalised."; exit 1; }
  if pgrep -f "harness.run|validation.gate0|fit_check" >/dev/null 2>&1; then
    echo "A model job is still running. Wait for it first."; exit 1
  fi
  nohup caffeinate -is "$0" --fg >> runs/weekend.log 2>&1 &
  echo "Started (process $!)."
  echo "  Step status: grep -E 'START|DONE|FAILED|cells in grid|stopping' runs/weekend.log"
  echo "  Stop it:     kill $!   (then pkill -f harness.run)"
  exit 0
fi

PY=.venv/bin/python
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step() {  # step NAME CMD...
  local name=$1 rc; shift
  echo "$(stamp) START  $name"
  "$@"; rc=$?
  if [[ $rc -eq 0 ]]; then echo "$(stamp) DONE   $name"; else echo "$(stamp) FAILED $name (exit $rc)"; fi
}
echo "$(stamp) ===== weekend queue started ====="
step "wave 1" $PY -m harness.run "$CFG" --run-name wave1 --models "${WAVE1[@]}"
step "wave 2" $PY -m harness.run "$CFG" --run-name wave2 --models "${WAVE2[@]}" --max-hours 40
# Sensitivity check, not in either set: the strict-set model at 8-bit (it fails Gate 0 criterion 4 there).
step "8-bit repeat, Gemma 3 4B" $PY -m harness.run "$CFG" --run-name wave2-8bit --models mlx-community/gemma-3-4b-it-8bit
echo "$(stamp) ===== weekend queue finished ====="
grep -E "START|DONE|FAILED|cells in grid|stopping" runs/weekend.log | tail -12
