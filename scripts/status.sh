#!/usr/bin/env bash
# Where is a run? Shows progress, pace, estimated finish and any problems.
# It never prints questionnaire scores, so checking progress does not reveal results by arm.
#   scripts/status.sh              # most recently active run
#   scripts/status.sh gemma-v2.0   # a named run (folder under runs/)
set -euo pipefail
cd "$(dirname "$0")/.."
run="${1:-$(ls -t runs/*/run.log 2>/dev/null | head -1 | xargs -n1 dirname | xargs -n1 basename)}"
[[ -n "$run" && -d "runs/$run" ]] || { echo "No run found. Usage: scripts/status.sh <run-name>"; exit 1; }
.venv/bin/python - "$run" <<'PY'
import json, statistics, sys, time
from datetime import datetime, timedelta
from pathlib import Path
run = Path("runs") / sys.argv[1]
log = (run / "run.log").read_text().splitlines() if (run / "run.log").exists() else []
res = run / "results.jsonl"
rows = [json.loads(l) for l in res.read_text().splitlines() if l.strip()] if res.exists() else []
cfg = json.loads((run / "protocol.json").read_text())["config"]
sys.path.insert(0, ".")
from harness.run import build_cells
total = len(build_cells(cfg, [m["id"] for m in cfg["models"]]))
print(f"Run: {run.name}   {len(rows)} of {total} conversations done")
for model in dict.fromkeys(r["model"] for r in rows):
    mine = [r for r in rows if r["model"] == model]
    print(f"  {model}: {len(mine)} done, median {statistics.median(r['duration_s'] for r in mine):.0f}s each, "
          f"peak memory {max(r.get('peak_memory_gb', 0) for r in mine)} GB")
finished = any("done:" in l or "stopping:" in l for l in log[-3:])
idle = time.time() - res.stat().st_mtime if res.exists() else 0
if finished:
    print("State: FINISHED -", log[-1])
elif rows:
    pace = statistics.median(r["duration_s"] for r in rows[-10:])
    eta = datetime.now() + timedelta(seconds=pace * (total - len(rows)))
    state = "RUNNING" if idle < 600 else f"POSSIBLY STALLED (no new result for {idle/60:.0f} min)"
    print(f"State: {state}; at the recent pace it finishes around {eta:%H:%M}")
bad = sum("nonfinite_acts" in cp for r in rows for cp in r["checkpoints"].values())
errs = [l for l in log if any(w in l for w in ("Traceback", "Error", "Killed"))]
hot = [l for l in log if "thermal:" in l and "no thermal warnings" not in l and "start;" not in l]
print(f"Problems: {bad} checkpoints with infinite values, {len(errs)} error lines, {len(hot)} heat warnings")
for l in (errs + hot)[-3:]:
    print("   ", l[:200])
PY
