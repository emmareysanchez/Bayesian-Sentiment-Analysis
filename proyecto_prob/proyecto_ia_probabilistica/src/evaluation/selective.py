"""Selective prediction: risk-coverage curves and AURC.

A selective predictor rejects samples whose uncertainty score exceeds a
threshold. For a given threshold τ:
    coverage(τ) = fraction of samples the model decides to classify
    risk(τ)     = error rate on those classified samples

AURC (Area Under the Risk-Coverage curve) summarizes the trade-off: lower is
better. Random uncertainty gives a flat curve; a perfect rejector gives zero
risk at full coverage up to the point where all errors are rejected.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

# NumPy 2.0 renamed np.trapz to np.trapezoid. Keep a compat alias.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def risk_coverage_curve(
    uncertainty: torch.Tensor | np.ndarray,
    correct: torch.Tensor | np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute the risk-coverage curve.

    Args:
        uncertainty : [N] score; HIGHER means MORE uncertain.
        correct     : [N] bool/int, 1 if prediction is correct.

    Returns:
        dict with keys 'coverage', 'risk', 'threshold' — each [N+1] arrays
        sorted by coverage ascending.
    """
    u = np.asarray(uncertainty).ravel().astype(np.float64)
    c = np.asarray(correct).ravel().astype(np.float64)
    N = len(u)

    # Sort by uncertainty ascending -> we keep the most confident first
    order = np.argsort(u, kind="stable")
    c_sorted = c[order]
    u_sorted = u[order]

    # For k samples kept (k=1..N): coverage = k/N, risk = 1 - mean(correct among top-k confident)
    kept_correct = np.cumsum(c_sorted)
    ks = np.arange(1, N + 1)
    coverage = ks / N
    risk = 1.0 - (kept_correct / ks)

    # Prepend coverage=0 (abstain all): risk convention = 0
    coverage = np.concatenate([[0.0], coverage])
    risk = np.concatenate([[0.0], risk])
    threshold = np.concatenate([[u_sorted[0] if N > 0 else 0.0], u_sorted])

    return {"coverage": coverage, "risk": risk, "threshold": threshold}


def aurc(uncertainty, correct) -> float:
    """Area Under the Risk-Coverage curve (trapezoidal)."""
    rc = risk_coverage_curve(uncertainty, correct)
    return float(_trapz(rc["risk"], rc["coverage"]))


def error_at_coverage(uncertainty, correct, target_coverage: float) -> float:
    """Risk (error rate) at a given coverage level."""
    rc = risk_coverage_curve(uncertainty, correct)
    idx = np.searchsorted(rc["coverage"], target_coverage, side="right") - 1
    idx = max(0, min(idx, len(rc["risk"]) - 1))
    return float(rc["risk"][idx])


def coverage_at_risk(uncertainty, correct, target_risk: float) -> float:
    """Maximum coverage achievable while keeping risk below target."""
    rc = risk_coverage_curve(uncertainty, correct)
    ok = rc["risk"] <= target_risk
    if not ok.any():
        return 0.0
    return float(rc["coverage"][ok].max())


def compare_scores(
    uncertainty_dict: Dict[str, np.ndarray],
    correct: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute AURC + error@{80,90,95}% coverage for each score."""
    out = {}
    for name, unc in uncertainty_dict.items():
        out[name] = {
            "aurc": aurc(unc, correct),
            "err_at_cov_80": error_at_coverage(unc, correct, 0.80),
            "err_at_cov_90": error_at_coverage(unc, correct, 0.90),
            "err_at_cov_95": error_at_coverage(unc, correct, 0.95),
            "cov_at_risk_5pct": coverage_at_risk(unc, correct, 0.05),
        }
    return out
