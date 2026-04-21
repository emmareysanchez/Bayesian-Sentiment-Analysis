"""Deterministic MLP baseline."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DeterministicConfig:
    input_dim: int
    hidden_dim: int = 128
    dropout_p: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 25
    patience: int = 5


class DeterministicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
