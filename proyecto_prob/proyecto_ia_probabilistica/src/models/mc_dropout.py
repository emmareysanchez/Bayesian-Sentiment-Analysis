"""MC-Dropout baseline (keeps dropout active at inference for uncertainty)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MCDropoutConfig:
    input_dim: int
    hidden_dim: int = 128
    dropout_p: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 25
    patience: int = 5
    mc_samples: int = 50


class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_p: float = 0.2):
        super().__init__()
        assert dropout_p > 0.0, "MC-Dropout requires dropout_p > 0"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 2),
        )
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """Return [S, B, 2] logits with dropout active."""
        # Force train mode on dropout only
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        out = []
        for _ in range(mc_samples):
            out.append(self.net(x))
        return torch.stack(out, dim=0)
