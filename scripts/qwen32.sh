#!/usr/bin/env bash
# Qwen3 32B (18.4 GB of weights, 64 layers): the largest model worth trying on a 24 GB Mac.
# Safe to start while the day queue is still running: it downloads now, then waits for every other
# model job to end before loading the model, so two models are never in memory at once.
# Then: fit check, then Gate 0 (about 2.5 h). The fit check needs at least 8 tokens/s (Gemma 3 27B ran at
# 14.2); slower means the model is spilling out of memory, and the step stops within minutes.
# Close other apps before it starts.
#   scripts/qwen32.sh
#   grep -E "START|DONE|FAILED|fit check|waiting" runs/bigger.log | tail
set -uo pipefail
cd "$(dirname "$0")/.."
M=mlx-community/Qwen3-32B-4bit
if [[ "${1:-}" != "--fg" ]]; then
  nohup caffeinate -is "$0" --fg >> runs/bigger.log 2>&1 &
  echo "Started (process $!): downloads Qwen3 32B now, starts Gate 0 when the current queue ends."
  echo "  Step status: grep -E 'START|DONE|FAILED|fit check|waiting' runs/bigger.log | tail"
  echo "  Stop it:     kill $!   (then pkill -f 'validation.gate0|fit_check')"
  exit 0
fi
echo "$(date '+%Y-%m-%d %H:%M:%S') qwen32: downloading $M"
.venv/bin/hf download "$M" >/dev/null || { echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED qwen32 download"; exit 1; }
echo "$(date '+%Y-%m-%d %H:%M:%S') qwen32: download done; waiting for other model jobs to end"
while pgrep -f "day_nov13|bigger.sh --fg|reading_steer.sh --fg|validation.gate0|fit_check|harness.run" >/dev/null; do
  sleep 60
done
FIT_FLOOR=8 scripts/bigger.sh --fg "$M"
