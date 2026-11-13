"""The anxious-versus-calm reading: fitting and scoring. Numpy only (no model needed).

Two probes are fitted at every layer from the TRAIN replies:
    mb    Martorell & Bianchi: difference of pole means divided coordinate-wise by the pooled
          within-pole variance, then unit length. Needed for Gemma, whose variance sits in a few
          huge coordinates that would otherwise dominate the projection.
    mean  the plain difference of means, unit length (the Day 1 method), for comparison.
A reply's score is its residual vector dotted with the probe. Scores are reported in "SD units":
(score - midpoint of the two training pole means) / pooled within-pole SD of the scores on the
SELECT set. On that scale the poles sit at about +-d/2, and 0 is halfway between them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VAR_FLOOR = 1e-6  # fraction of the mean variance added to every coordinate (only guards against zeros)


@dataclass
class Probe:
    layer: int
    kind: str
    w: np.ndarray          # unit vector, [d]
    mid: float             # midpoint of the training pole means of the score
    scale: float = 1.0     # pooled within-pole SD of the score on the SELECT set (set by calibrate)

    def score(self, X: np.ndarray) -> np.ndarray:
        """X: [n, d] residuals at self.layer -> raw scores [n]."""
        return np.asarray(X, dtype=np.float64) @ self.w

    def z(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) - self.mid) / self.scale


def fit(Xa: np.ndarray, Xc: np.ndarray, layer: int, kind: str = "mb") -> Probe:
    """Xa, Xc: [n, d] anxious and calm training residuals at one layer."""
    Xa, Xc = np.asarray(Xa, np.float64), np.asarray(Xc, np.float64)
    diff = Xa.mean(0) - Xc.mean(0)
    if kind == "mb":
        var = ((len(Xa) - 1) * Xa.var(0, ddof=1) + (len(Xc) - 1) * Xc.var(0, ddof=1)) / (len(Xa) + len(Xc) - 2)
        w = diff / (var + VAR_FLOOR * var.mean())
    elif kind == "mean":
        w = diff
    else:
        raise ValueError(kind)
    w = w / np.linalg.norm(w)
    mid = float((Xa @ w).mean() + (Xc @ w).mean()) / 2
    return Probe(layer, kind, w, mid)


def pooled_sd(sa: np.ndarray, sc: np.ndarray) -> float:
    return float(np.sqrt(((len(sa) - 1) * sa.var(ddof=1) + (len(sc) - 1) * sc.var(ddof=1)) / (len(sa) + len(sc) - 2)))


def cohens_d(sa: np.ndarray, sc: np.ndarray) -> float:
    sd = pooled_sd(sa, sc)
    return float((sa.mean() - sc.mean()) / sd) if sd > 0 else np.nan


def accuracy(p: Probe, Xa: np.ndarray, Xc: np.ndarray) -> float:
    """Share of replies on the right side of the training midpoint."""
    return float(np.concatenate([p.score(Xa) > p.mid, p.score(Xc) <= p.mid]).mean())


def calibrate(p: Probe, Xa_sel: np.ndarray, Xc_sel: np.ndarray) -> Probe:
    p.scale = pooled_sd(p.score(Xa_sel), p.score(Xc_sel))
    return p


def band(n_layers: int, lo: float = 0.2, hi: float = 0.8) -> list[int]:
    """The middle 60% of layers."""
    return list(range(int(lo * n_layers), int(np.ceil(hi * n_layers))))


def layer_scan(train_a, train_c, sel_a, sel_c, layers, kind="mb") -> list[dict]:
    """Per layer: fit on TRAIN, Cohen's d and accuracy on SELECT. Arrays are [n, L, d]."""
    out = []
    for l in layers:
        p = fit(train_a[:, l], train_c[:, l], l, kind)
        sa, sc = p.score(sel_a[:, l]), p.score(sel_c[:, l])
        out.append({"layer": l, "d": cohens_d(sa, sc), "acc": accuracy(p, sel_a[:, l], sel_c[:, l])})
    return out


def permutation_accuracy(train_a, train_c, test_a, test_c, layer, kind="mb", n_perm=500, seed=0) -> dict:
    """Shuffle test: refit with the TRAIN pole labels shuffled; how often does a shuffled probe sort
    the TEST replies as well as the real one? The sign is fixed so that higher means anxious on TRAIN
    labels, which shuffled probes cannot exploit."""
    rng = np.random.default_rng(seed)
    real = accuracy(fit(train_a[:, layer], train_c[:, layer], layer, kind), test_a[:, layer], test_c[:, layer])
    X = np.concatenate([train_a[:, layer], train_c[:, layer]])
    n = len(train_a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(X))
        p = fit(X[idx[:n]], X[idx[n:]], layer, kind)
        null[i] = accuracy(p, test_a[:, layer], test_c[:, layer])
    return {"real": real, "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
            "p": float((1 + (null >= real).sum()) / (1 + n_perm))}


def boot_mean_ci(x: np.ndarray, n_boot=5000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return tuple(np.percentile(m, [2.5, 97.5]).tolist())


def boot_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot=5000, seed=0) -> tuple[float, float]:
    """CI for mean(num) / mean(den), resampling the two groups independently."""
    rng = np.random.default_rng(seed)
    r = np.array([num[rng.integers(0, len(num), len(num))].mean() / den[rng.integers(0, len(den), len(den))].mean()
                  for _ in range(n_boot)])
    return tuple(np.percentile(r, [2.5, 97.5]).tolist())
