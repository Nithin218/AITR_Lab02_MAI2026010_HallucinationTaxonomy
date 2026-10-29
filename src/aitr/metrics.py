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
