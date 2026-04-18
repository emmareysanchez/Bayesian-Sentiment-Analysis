"""Uncertainty decomposition (Gal 2016; Depeweg et al. 2018)."""
from __future__ import annotations

from typing import Dict

import torch


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy along last dim."""
    return -(p * p.clamp_min(eps).log()).sum(dim=-1)


def decompose_mc(mc_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Decompose predictive uncertainty from MC samples.

    Args:
        mc_probs : [S, B, C] Monte Carlo probability samples.

    Returns dict with per-example [B] tensors:
        predictive_entropy : H[ E_q[p(y|x,w)] ]
        aleatoric_entropy  : E_q[ H[p(y|x,w)] ]
        mutual_info        : predictive - aleatoric  (epistemic)
    """
    mean_probs = mc_probs.mean(dim=0)              # [B, C]
    pred_ent = entropy(mean_probs)                 # [B]
    ale_ent = entropy(mc_probs).mean(dim=0)        # [B]
    mi = pred_ent - ale_ent
    return {
        "mean_probs": mean_probs,
        "predictive_entropy": pred_ent,
        "aleatoric_entropy": ale_ent,
        "mutual_info": mi,
    }


def mean_decomposition(mc_probs: torch.Tensor) -> Dict[str, float]:
    d = decompose_mc(mc_probs)
    return {
        "predictive_entropy": float(d["predictive_entropy"].mean().item()),
        "aleatoric_entropy": float(d["aleatoric_entropy"].mean().item()),
        "mutual_info": float(d["mutual_info"].mean().item()),
    }
