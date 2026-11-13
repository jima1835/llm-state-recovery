#!/usr/bin/env bash
# Run once the pilot has finished (about 15 minutes). Does three things:
#   1. runs the automatic tests
#   2. checks that the fast questionnaire method gives the same answers as the slow one, on both models
#   3. tells you what to do next
set -uo pipefail
cd "$(dirname "$0")/.."
if ! grep -q "done:" runs/pilot-v2-night1.log 2>/dev/null; then
  echo "The pilot has not finished yet. Check with: scripts/status.sh pilot-benzion-v2"; exit 1
fi
echo "== 1/2 Automatic tests"
.venv/bin/python -m pytest -q tests || { echo "TESTS FAILED. Stop here and paste the output to Claude."; exit 1; }
echo "== 2/2 Fast method versus slow method (2 conversations per model)"
out="runs/prefix-check-$(date +%F-%H%M).log"
HF_HUB_OFFLINE=1 .venv/bin/python -u -m harness.check_prefix_reuse runs/pilot-benzion-v2 -n 2 2>&1 | tee "$out"
status=${PIPESTATUS[0]}
echo
if [[ $status -eq 0 ]]; then
  echo "PASS. The fast method is safe. Next:"
  echo "  git add -A && git commit -m \"v2.0 protocol: baseline turn, shared fork prefix, first-64 readout\" && git push"
  echo "  scripts/start_v2.sh"
else
  echo "FAIL or error. The fast method is NOT confirmed. Paste $out to Claude."
  echo "You can still run v2.0 with the slow, proven method:  scripts/start_v2.sh --safe"
fi
exit $status
