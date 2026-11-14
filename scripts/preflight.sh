#!/usr/bin/env bash
# Pre-flight for the weekend run: two conversations per model in the exact experiment format, on pilot
# seed 0 (never a confirmatory seed), to confirm every model loads, answers the questionnaire in a
# readable form and saves its readings. Six of the eight models have only run Gate 0 so far.
# NOT confirmatory data: nothing here enters the analysis, and only validity is looked at (did the
# answers parse), not the scores by arm. About 15 minutes.
#   scripts/preflight.sh            run in the foreground and print a table
set -uo pipefail
cd "$(dirname "$0")/.."
if pgrep -f "harness.run|validation.gate0|fit_check" >/dev/null 2>&1; then
  echo "A model job is still running. Wait for it first."; exit 1
fi
PY=.venv/bin/python
CFG=runs/preflight.yaml
# Same file as the confirmatory config, with every seed list replaced by the pilot seed.
sed -E -e 's/^seeds: .*/seeds: [0]/' -e 's/^    seeds: \[100\]/    seeds: [0]/' -e 's/^run_name: .*/run_name: preflight/' \
  configs/wave1.yaml > "$CFG"
grep -E "^ *seeds:" "$CFG" | grep -q "10[0-5]" && { echo "A confirmatory seed is still in $CFG. Stopping."; exit 1; }
for m in mlx-community/gemma-3-4b-it-4bit mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
         mlx-community/gemma-3-12b-it-4bit mlx-community/Qwen3-14B-4bit \
         mlx-community/Llama-3.2-3B-Instruct-4bit mlx-community/Mistral-Small-24B-Instruct-2501-4bit \
         mlx-community/gemma-3-27b-it-4bit mlx-community/gemma-3-4b-it-8bit; do
  echo "== ${m##*/}"
  caffeinate -is $PY -m harness.run "$CFG" --run-name preflight --models "$m" --limit 2 --cooldown-sec 0 2>&1 \
    | grep -E "loaded|Traceback|Error|error|stopping" | grep -v "thermal"
done
$PY - <<'EOF'
import json, collections
from pathlib import Path
import numpy as np
run = Path("runs/preflight")
rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[r["model"].split("/")[-1]].append(r)
print(f"\n{'model':40s} convs  seeds  lowest answer mass  0-100 answers read  readings finite  s/conv")
ok_all = True
for m, rs in by.items():
    mass = min(c["stai"]["min_mass"] for r in rs for c in r["checkpoints"].values())
    vas = [c["vas"]["value"] for r in rs for c in r["checkpoints"].values()]
    finite = all(np.isfinite(a).all() for r in rs for a in np.load(run / r["acts_file"]).values())
    seeds = sorted({r["seed"] for r in rs})
    ok = mass >= 0.5 and all(v is not None for v in vas) and finite and seeds == [0]
    ok_all &= ok
    print(f"{m:40s} {len(rs):5d}  {seeds}  {mass:18.2f}  {sum(v is not None for v in vas):8d} of {len(vas):<6d}  "
          f"{str(finite):15s}  {np.mean([r['duration_s'] for r in rs]):5.0f}  {'ok' if ok else 'CHECK'}")
print("\nPRE-FLIGHT", "PASSED" if ok_all and len(by) == 8 else "NEEDS A LOOK", f"({len(by)} of 8 models ran)")
EOF
