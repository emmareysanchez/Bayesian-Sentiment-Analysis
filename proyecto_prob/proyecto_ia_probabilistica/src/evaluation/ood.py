"""Out-of-distribution detection metrics.

Treat uncertainty as a score. Higher uncertainty on OOD than ID is what we want.
"""
from __future__ import annotations

from typing import Dict

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
