"""Business value analysis: optimal rejection threshold under a cost model.

Cost model
----------
For each sample, the system either:
    - classifies automatically:
        correct  -> cost 0
        FP (predicting positive when true is negative)  -> cost_fp
        FN (predicting negative when true is positive)  -> cost_fn
    - rejects to a human reviewer:
        cost cost_review regardless of outcome

Expected total cost as a function of uncertainty threshold τ:
    C(τ) = Σ_i [  1[u_i <= τ] * classification_cost_i
                + 1[u_i >  τ] * cost_review ]

We compute C(τ) on a grid of τ values and locate τ* = argmin C(τ).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def classification_costs(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> np.ndarray:
    """Per-sample cost if we let the model decide (no rejection)."""
    preds = np.asarray(predictions).ravel()
    y = np.asarray(true_labels).ravel()
    cost = np.zeros(len(y), dtype=np.float64)
    # FP: predicted positive, true negative
    cost[(preds == 1) & (y == 0)] = cost_fp
    # FN: predicted negative, true positive
    cost[(preds == 0) & (y == 1)] = cost_fn
    return cost


def expected_cost_curve(
    uncertainty: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    cost_review: float,
    n_thresholds: int = 200,
) -> Dict[str, np.ndarray]:
    """Sweep thresholds and compute expected total cost.

    Returns dict with:
        'thresholds', 'total_cost', 'avg_cost', 'rejection_rate'
    """
    u = np.asarray(uncertainty).ravel().astype(np.float64)
    auto_cost = classification_costs(predictions, true_labels, cost_fp, cost_fn)
    N = len(u)

    # Use quantiles of u as candidate thresholds (plus 0 and max)
    qs = np.linspace(0.0, 1.0, n_thresholds)
    thresholds = np.quantile(u, qs)
    thresholds = np.concatenate([[u.min() - 1e-9], thresholds, [u.max() + 1e-9]])
    thresholds = np.unique(thresholds)

    total_cost = np.zeros_like(thresholds)
    rejection_rate = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        reject = u > t
        classify = ~reject
        total_cost[i] = auto_cost[classify].sum() + cost_review * reject.sum()
        rejection_rate[i] = reject.mean()

    return {
        "thresholds": thresholds,
        "total_cost": total_cost,
        "avg_cost": total_cost / N,
        "rejection_rate": rejection_rate,
    }


def optimal_threshold(
    uncertainty: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    cost_review: float,
) -> Dict[str, float]:
    """Find τ* minimizing expected total cost."""
    curve = expected_cost_curve(
        uncertainty, predictions, true_labels, cost_fp, cost_fn, cost_review
    )
    i_star = int(np.argmin(curve["total_cost"]))
    tau_star = float(curve["thresholds"][i_star])

    # Baselines
    auto_cost = classification_costs(predictions, true_labels, cost_fp, cost_fn)
    always_classify = float(auto_cost.sum())
    always_review = float(cost_review * len(predictions))

    savings = always_classify - float(curve["total_cost"][i_star])
    savings_pct = 100.0 * savings / max(always_classify, 1e-9)

    return {
        "tau_star": tau_star,
        "total_cost_at_star": float(curve["total_cost"][i_star]),
        "avg_cost_at_star": float(curve["avg_cost"][i_star]),
        "rejection_rate_at_star": float(curve["rejection_rate"][i_star]),
        "baseline_always_classify": always_classify,
        "baseline_always_review": always_review,
        "savings_vs_always_classify": savings,
        "savings_pct_vs_always_classify": savings_pct,
    }
