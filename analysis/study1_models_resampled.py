"""Post hoc (18 Nov): Study 1 extended-set intervals that also resample the seven models.
Read-only. Uses analysis.confirmatory unchanged. Study 1 was unblinded on 15 Nov."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from analysis import confirmatory as C

data, info = C.load_runs([Path("runs/wave1"), Path("runs/wave2")], Path(sys.argv[1]), C.EXTENDED)
models = list(C.EXTENDED)
per = {m: C.model_stats({arm: a["X"] for arm, a in data[m].items()}) for m in models}
rng = np.random.default_rng(20261118)
N = 4000
draws = {t: [] for t in ("T1", "T2", "T3", "T4", "T5", "T6")}
for b in range(N):
    ms = [models[i] for i in rng.integers(0, len(models), len(models))]
    d = C.bootstrap(data, ms, 1, "narrative", seed=int(rng.integers(0, 2**31)))
    for t in draws:
        draws[t].append(d[t][0])
print("test  mean-of-7  models-resampled 95%  share<=0  per-model")
for t, _, side in C.TESTS:
    x = np.array(draws[t]); x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    pm = " ".join(f"{per[m][t]:+.2f}" for m in models)
    print(f"{t}  {np.mean([per[m][t] for m in models]):+.2f}  [{lo:+.2f}, {hi:+.2f}]  {np.mean(x<=0):.3f}  {pm}")
print("models:", " | ".join(C.short(m) for m in models))
print("\nQuestionnaire total (20-80), mean by checkpoint: baseline / after narrative / after regulation / after 2nd narrative")
for m in models:
    for arm in ("relaxation", "neutral_filler"):
        X = data[m][arm]["X"]["stai"]
        print(f"{C.short(m):45s} {arm:15s} " + " / ".join(f"{v:.1f}" for v in np.nanmean(X, axis=0)))
