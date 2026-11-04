"""AUROC helpers.

`agrawal_bootstrap_auc` replicates the AUC reported by Agrawal et al. (code/metrics.ipynb calling
roc-utils 0.2.2 `plot_roc_bootstrap(n_bootstrap=100, random_state=42, auto_flip=True)`):
  * a fresh np.random.RandomState(seed) per call; each resample uses rng.randint(0, n, n)
  * resamples that lack one class are skipped (the draw still consumes the RNG)
  * reported value = mean of the per-resample AUCs; if the mean ROC curve has AUC < 0.5 the
    predictor is treated as inverted and every resample AUC becomes 1 - AUC (auto-flip)
  * roc-utils rounds thresholds to 8 decimals, so near-identical scores count as ties

Auto-flip is for reproducing V2 only. Our own analyses fix score orientation in advance
(protocol §3) and must call it with auto_flip=False.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def point_auc(scores, y) -> float:
    return float(roc_auc_score(np.asarray(y), np.round(np.asarray(scores, dtype=float), 8)))


def cluster_bootstrap_auc(scores: dict, y, clusters, n_boot: int = 2000, seed: int = 7) -> dict:
    """Point AUROC and 95% percentile CI per detector, resampling clusters (topics) with replacement.

    No auto-flip: orientation is fixed in advance (protocol §3). The same resamples are used for every
    detector, so differences between detectors are paired; `deltas` holds the per-resample AUCs for that.
    """
    y = np.asarray(y)
    clusters = np.asarray(clusters)
    s = {k: np.round(np.asarray(v, dtype=float), 8) for k, v in scores.items()}
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in s}
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) < 2:
            continue
        for k, v in s.items():
            boots[k].append(roc_auc_score(y[idx], v[idx]))
    out = {}
    for k, v in s.items():
        b = np.array(boots[k])
        out[k] = {"auc": float(roc_auc_score(y, v)), "lo": float(np.percentile(b, 2.5)),
                  "hi": float(np.percentile(b, 97.5)), "boot": b}
    return out


def paired_delta(result: dict, a: str, b: str) -> dict:
    d = result[a]["boot"] - result[b]["boot"]
    return {"delta": result[a]["auc"] - result[b]["auc"], "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5))}


def agrawal_bootstrap_auc(scores, y, n_bootstrap: int = 100, seed: int = 42, auto_flip: bool = True) -> dict:
    s = np.round(np.asarray(scores, dtype=float), 8)
    y = np.asarray(y)
    if len(s) != len(y):
        raise ValueError("scores and labels differ in length")
    rng = np.random.RandomState(seed)
    n = len(y)
    fpr_grid = np.insert(np.linspace(0, 1, 101), 0, 0)  # roc-utils resolution=101 + leading 0
    aucs, tprs = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
        fpr, tpr, _ = roc_curve(yb, sb, drop_intermediate=False)
        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[[0, -1]] = tpr[[0, -1]]
        tprs.append(tpr_i)
    aucs = np.array(aucs)
    mean_curve_auc = float(np.trapezoid(np.mean(tprs, axis=0), fpr_grid))
    flipped = bool(auto_flip and mean_curve_auc < 0.5)
    if flipped:
        aucs = 1 - aucs
    return {"auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
            "n_resamples_used": int(len(aucs)), "flipped": flipped}
