"""
src/models/deterministic.py
Deterministic baseline classifier for sentiment analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


class DeterministicClassifier(nn.Module):
    """Simple deterministic MLP baseline with the same architecture as the BNN."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EvalMetrics:
    accuracy: float
    nll: float

    def to_dict(self) -> Dict[str, float]:
        return {"accuracy": self.accuracy, "nll": self.nll}


def train_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device = "cpu",
) -> float:
    """Train for one epoch and return average cross-entropy per sample."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.long().to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.shape[0]
        total_samples += x_batch.shape[0]

    return total_loss / total_samples


@torch.no_grad()
def evaluate_deterministic(
    model: nn.Module,
    data_loader,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Evaluate accuracy and NLL using predicted class probabilities."""
    model.eval()

    all_probs = []
    all_targets = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.long().to(device)

        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu())
        all_targets.append(y_batch.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    preds = all_probs.argmax(dim=1)
    accuracy = (preds == all_targets).float().mean().item()

    eps = 1e-8
    true_class_probs = all_probs[torch.arange(len(all_targets)), all_targets]
    nll = -torch.log(true_class_probs.clamp(min=eps)).mean().item()

    return EvalMetrics(accuracy=accuracy, nll=nll).to_dict()
