"""Basic probabilistic classification metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def accuracy(probs: torch.Tensor, y: torch.Tensor) -> float:
    return float((probs.argmax(dim=1) == y).float().mean().item())


def nll(probs: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    idx = torch.arange(y.shape[0])
    return float(-torch.log(probs[idx, y].clamp_min(eps)).mean().item())


def brier_binary(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Binary Brier score on class-1 probability."""
    p1 = probs[:, 1]
    return float(((p1 - y.float()) ** 2).mean().item())


def ece(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    conf, pred = probs.max(dim=1)
    acc = pred.eq(y)
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    N = y.shape[0]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            bin_acc = acc[mask].float().mean()
            bin_conf = conf[mask].mean()
            total += (mask.sum().item() / N) * abs(bin_acc.item() - bin_conf.item())
    return float(total)


def mce(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """Maximum Calibration Error."""
    conf, pred = probs.max(dim=1)
    acc = pred.eq(y)
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            bin_acc = acc[mask].float().mean().item()
            bin_conf = conf[mask].mean().item()
            worst = max(worst, abs(bin_acc - bin_conf))
    return float(worst)


def reliability_bins(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> Dict[str, np.ndarray]:
    """Return per-bin data for reliability diagrams."""
    conf, pred = probs.max(dim=1)
    acc = pred.eq(y).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1)

    centers, accs, confs, sizes = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            centers.append(((lo + hi) / 2).item())
            accs.append(acc[mask].mean().item())
            confs.append(conf[mask].mean().item())
            sizes.append(int(mask.sum().item()))
    return {
        "bin_center": np.array(centers),
        "bin_accuracy": np.array(accs),
        "bin_confidence": np.array(confs),
        "bin_size": np.array(sizes),
    }


def all_metrics(probs: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    return {
        "accuracy": accuracy(probs, y),
        "nll": nll(probs, y),
        "brier": brier_binary(probs, y),
        "ece": ece(probs, y),
        "mce": mce(probs, y),
    }
