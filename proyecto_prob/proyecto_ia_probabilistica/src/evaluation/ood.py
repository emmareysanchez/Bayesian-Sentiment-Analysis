"""Out-of-distribution detection metrics.

Treat uncertainty as a score. Higher uncertainty on OOD than ID is what we want.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc_id_vs_ood(score_id: np.ndarray, score_ood: np.ndarray) -> float:
    """AUROC treating OOD=positive class."""
    score_id = np.asarray(score_id).ravel()
    score_ood = np.asarray(score_ood).ravel()
    y = np.concatenate([np.zeros_like(score_id), np.ones_like(score_ood)])
    s = np.concatenate([score_id, score_ood])
    return float(roc_auc_score(y, s))


def aupr_id_vs_ood(score_id: np.ndarray, score_ood: np.ndarray) -> float:
    """Average Precision with OOD as positive class."""
    score_id = np.asarray(score_id).ravel()
    score_ood = np.asarray(score_ood).ravel()
    y = np.concatenate([np.zeros_like(score_id), np.ones_like(score_ood)])
    s = np.concatenate([score_id, score_ood])
    return float(average_precision_score(y, s))


def fpr_at_tpr(score_id: np.ndarray, score_ood: np.ndarray, target_tpr: float = 0.95) -> float:
    """FPR (false-positive rate on ID) at TPR=target (detection rate on OOD)."""
    score_id = np.asarray(score_id).ravel()
    score_ood = np.asarray(score_ood).ravel()
    # Threshold such that target_tpr fraction of OOD is flagged
    th = np.quantile(score_ood, 1 - target_tpr)
    fpr = float((score_id >= th).mean())
    return fpr


class MahalanobisOOD:
    """Class-conditional Mahalanobis distance OOD detector (Lee et al. 2018).

    Fit on training features + labels. Score = min over classes of
    (x - μ_c)^T Σ^{-1} (x - μ_c), then negated so higher = more OOD.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.class_means_: Optional[np.ndarray] = None   # [C, D]
        self.precision_: Optional[np.ndarray] = None      # [D, D]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisOOD":
        classes = np.unique(y)
        means = np.stack([X[y == c].mean(axis=0) for c in classes])   # [C, D]
        # Pooled within-class covariance
        centered = np.vstack([X[y == c] - means[i] for i, c in enumerate(classes)])
        cov = np.cov(centered, rowvar=False) + self.eps * np.eye(X.shape[1])
        self.class_means_ = means
        self.precision_ = np.linalg.inv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample Mahalanobis score (higher = more OOD)."""
        dists = []
        for mu in self.class_means_:
            d = X - mu                                       # [N, D]
            maha = np.einsum("nd,de,ne->n", d, self.precision_, d)
            dists.append(maha)
        min_dist = np.stack(dists, axis=1).min(axis=1)      # [N]
        return min_dist                                      # higher = farther from any class


def ood_report(
    scores_id: Dict[str, np.ndarray],
    scores_ood: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Args:
        scores_id  : {score_name: array[N_id]}
        scores_ood : {score_name: array[N_ood]} — same score names as scores_id
    """
    out = {}
    for name in scores_id:
        if name not in scores_ood:
            continue
        out[name] = {
            "auroc": auroc_id_vs_ood(scores_id[name], scores_ood[name]),
            "aupr_ood": aupr_id_vs_ood(scores_id[name], scores_ood[name]),
            "fpr_at_tpr_95": fpr_at_tpr(scores_id[name], scores_ood[name], 0.95),
        }
    return out
