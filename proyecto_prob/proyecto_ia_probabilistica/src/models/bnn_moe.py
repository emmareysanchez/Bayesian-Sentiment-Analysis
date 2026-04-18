"""
Bayesian Mixture-of-Experts BNN with topic gating and heteroscedastic head.

Architecture
============

Inputs:
    x  ∈ R^D_bert   (BERT CLS)
    θ  ∈ Δ^{K-1}    (topic mixture from sLDA; optional — if None, gating is uniform)

Shared feature extractor:
    φ(x)  =  ReLU( W_shared · x + b_shared )          W_shared ∈ R^{H×D_bert}

K Bayesian experts (one per topic):
    W_k ~ N(0, σ²_W / H · I )                         W_k ∈ R^{2 × H}
    b_k ~ N(0, σ²_b I)                                b_k ∈ R^2
    logits_k(x)  =  W_k φ(x) + b_k

Mixing via topic gate (deterministic, from sLDA):
    μ(x, θ)      =  Σ_k  θ_k · logits_k(x)

Heteroscedastic head (optional, flag `use_heteroscedastic`):
    W_σ ~ N(0, σ²_het I),  b_σ ~ N(0, σ²_het I)
    log σ²(x)    =  W_σ φ(x) + b_σ                     ∈ R^2
    logits_final =  μ(x, θ) + σ(x) ⊙ ε,  ε ~ N(0, I)     (reparameterized)

Likelihood:
    y ~ Categorical( softmax(logits_final) )

Priors: W_shared is NOT Bayesian (kept deterministic to cut parameter count;
all uncertainty comes from the experts and the heteroscedastic head).

Notes
-----
- All Bayesian weights are sampled via PyroSample → AutoNormal guide (mean-field).
- The gate θ is supplied by sLDA and is treated as observed side-information.
  If not supplied, we default to uniform θ = 1/K (equivalent to averaging experts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample


@dataclass
class BayesianMoEConfig:
    input_dim: int = 768
    hidden_dim: int = 128
    n_experts: int = 10
    prior_std_w: float = 1.0
    prior_std_b: float = 1.0
    prior_std_het: float = 1.0
    use_heteroscedastic: bool = False
    fan_in_scaled_prior: bool = True
    hetero_samples: int = 8          # MC samples for the Gaussian noise on logits


class BayesianMoE(PyroModule):
    def __init__(self, cfg: BayesianMoEConfig):
        super().__init__()
        self.cfg = cfg

        # --- Shared deterministic feature extractor ---
        self.shared = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
        )

        # --- K Bayesian experts ---
        # We use one PyroModule per expert to keep site names clean.
        self.experts = PyroModule[nn.ModuleList]([
            self._build_expert(k, cfg) for k in range(cfg.n_experts)
        ])

        # --- Heteroscedastic head ---
        if cfg.use_heteroscedastic:
            self.hetero = self._build_hetero(cfg)
        else:
            self.hetero = None

    def _build_expert(self, k: int, cfg: BayesianMoEConfig) -> PyroModule:
        """Build the k-th Bayesian linear expert h -> logits (2)."""
        expert = PyroModule[nn.Linear](cfg.hidden_dim, 2)
        std_w = cfg.prior_std_w / math.sqrt(max(cfg.hidden_dim, 1)) if cfg.fan_in_scaled_prior else cfg.prior_std_w
        expert.register_buffer(f"w_loc", torch.tensor(0.0))
        expert.register_buffer(f"w_scale", torch.tensor(std_w))
        expert.register_buffer(f"b_loc", torch.tensor(0.0))
        expert.register_buffer(f"b_scale", torch.tensor(cfg.prior_std_b))
        H = cfg.hidden_dim
        expert.weight = PyroSample(
            lambda m: dist.Normal(m.w_loc, m.w_scale).expand([2, H]).to_event(2)
        )
        expert.bias = PyroSample(
            lambda m: dist.Normal(m.b_loc, m.b_scale).expand([2]).to_event(1)
        )
        return expert

    def _build_hetero(self, cfg: BayesianMoEConfig) -> PyroModule:
        """Heteroscedastic head producing log σ² per class."""
        het = PyroModule[nn.Linear](cfg.hidden_dim, 2)
        het.register_buffer("w_loc", torch.tensor(0.0))
        het.register_buffer("w_scale", torch.tensor(cfg.prior_std_het))
        het.register_buffer("b_loc", torch.tensor(0.0))
        het.register_buffer("b_scale", torch.tensor(cfg.prior_std_het))
        H = cfg.hidden_dim
        het.weight = PyroSample(
            lambda m: dist.Normal(m.w_loc, m.w_scale).expand([2, H]).to_event(2)
        )
        het.bias = PyroSample(
            lambda m: dist.Normal(m.b_loc, m.b_scale).expand([2]).to_event(1)
        )
        return het

    # -----------------------------
    # Forward
    # -----------------------------

    def forward(
        self,
        x: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x     : [B, D_bert]
            theta : [B, K] topic gate (rows sum to 1). If None, uniform.
            y     : [B] labels for SVI; None at inference.
        Returns:
            logits : [B, 2]   (mean logits under the mixture and hetero noise)
        """
        B = x.shape[0]
        K = self.cfg.n_experts

        # Shared features
        h = self.shared(x)                                # [B, H]

        # Gate
        if theta is None:
            theta = torch.full((B, K), 1.0 / K, device=x.device)

        # Experts
        # stack of expert logits: [B, K, 2]
        expert_logits = torch.stack([exp(h) for exp in self.experts], dim=1)

        # Weighted mixture over K experts
        # gate: [B, K] -> [B, K, 1]
        mu = (theta.unsqueeze(-1) * expert_logits).sum(dim=1)   # [B, 2]

        # Heteroscedastic noise (optional)
        if self.hetero is not None:
            log_var = self.hetero(h).clamp(min=-6.0, max=2.0)    # [B, 2]
            sigma = torch.exp(0.5 * log_var)
            # Sample T noise draws and average in logit space (reparameterisation with variance reduction)
            T = self.cfg.hetero_samples
            eps = torch.randn(T, B, 2, device=x.device)
            noisy = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps    # [T, B, 2]
            logits = noisy.mean(0)                                 # [B, 2]
        else:
            logits = mu

        if y is not None:
            with pyro.plate("data", B):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        return logits
