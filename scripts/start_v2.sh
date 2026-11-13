#!/usr/bin/env bash
# Start the v2.0 Gemma run in the background (75 conversations).
#   scripts/start_v2.sh          # fast questionnaire method (use after after_pilot.sh says PASS)
#   scripts/start_v2.sh --safe   # slow, proven method (about 3x longer)
# Safe to run again after an interruption: finished conversations are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=configs/v2.0_gemma.yaml; log=runs/gemma-v2.0.log
if pgrep -f "harness.run" >/dev/null 2>&1; then
  echo "Another experiment run is still going. Wait for it (scripts/status.sh), or the two will slow each other down."; exit 1
fi
if [[ ! -f runs/gemma-v2.0/protocol.json ]]; then
  echo "Before the first start, read the neutral stand-in text (it cannot change once the run begins):"
  echo
  .venv/bin/python -c "import yaml,textwrap; print(textwrap.fill(' '.join(yaml.safe_load(open('$cfg'))['texts']['neutral_2'].split()), 100))"
  echo
  read -r -p "Approve this text and start the run? [y/N] " ok
  [[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "Not started. Edit texts.neutral_2 in $cfg, then run this again."; exit 1; }
fi
if [[ "${1:-}" == "--safe" ]]; then export LSR_PREFIX_REUSE=0; echo "Using the slow, proven questionnaire method."; fi
nohup caffeinate -is .venv/bin/python -m harness.run "$cfg" >> "$log" 2>&1 &
echo "Started (process $!). It keeps the Mac awake and stops by itself."
echo "  Progress:  scripts/status.sh gemma-v2.0"
echo "  Stop it:   kill $!"
