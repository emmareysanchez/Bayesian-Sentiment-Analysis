"""Generic SVI trainer for any PyroModule.

Supports:
  - DataLoaders that yield (x, y) or (x, y, theta)
  - AutoNormal or AutoDiagonalNormal guides
  - Early stopping on validation NLL
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pyro
import pyro.infer
import torch
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import Adam


@dataclass
class SVITrainerConfig:
    lr: float = 1e-3
    epochs: int = 25
    patience: int = 5
    mc_samples_eval: int = 100
    min_delta: float = 1e-4


def _unpack_batch(batch, device):
    """Accept (x, y) or (x, y, theta) and move to device."""
    if len(batch) == 2:
        x, y = batch
        return x.to(device), y.long().to(device), None
    elif len(batch) == 3:
        x, y, theta = batch
        return x.to(device), y.long().to(device), theta.to(device)
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}")


class SVITrainer:
    """Wraps an SVI training loop with optional theta-conditioning."""

    def __init__(
        self,
        model: Any,
        guide: Any,
        cfg: SVITrainerConfig,
        device: torch.device,
    ):
        self.model = model
        self.guide = guide
        self.cfg = cfg
        self.device = device

        self.svi = SVI(
            model=model,
            guide=guide,
            optim=Adam({"lr": cfg.lr}),
            loss=TraceMeanField_ELBO(),
        )

    def _step(self, x, y, theta):
        if theta is None:
            return self.svi.step(x, y)
        return self.svi.step(x, theta, y)

    def fit(
        self,
        train_loader,
        val_loader=None,
        val_metric_fn: Optional[Callable] = None,
        verbose: bool = True,
    ):
        """
        Args:
            val_metric_fn : callable(model, guide, val_loader, device) -> dict
                            must return a key 'nll' used for early stopping.
        """
        # Initialize guide on one batch (necessary for AutoNormal to place params)
        first = next(iter(train_loader))
        x0, y0, theta0 = _unpack_batch(first, self.device)
        if theta0 is None:
            self.guide(x0, y=y0)
        else:
            self.guide(x0, theta=theta0, y=y0)

        best_val = float("inf")
        bad = 0
        best_store = None
        history = {"train_elbo": [], "val_nll": [], "val_accuracy": []}

        for epoch in range(self.cfg.epochs):
            self.model.train()
            total = 0.0
            N = 0
            for batch in train_loader:
                x, y, theta = _unpack_batch(batch, self.device)
                loss = self._step(x, y, theta)
                total += loss
                N += x.shape[0]
            train_elbo = total / max(N, 1)
            history["train_elbo"].append(train_elbo)

            if val_loader is not None and val_metric_fn is not None:
                metrics = val_metric_fn(self.model, self.guide, val_loader, self.device)
                v_nll = metrics["nll"]
                v_acc = metrics.get("accuracy", float("nan"))
                history["val_nll"].append(v_nll)
                history["val_accuracy"].append(v_acc)

                if v_nll < best_val - self.cfg.min_delta:
                    best_val = v_nll
                    bad = 0
                    best_store = {k: v.detach().cpu().clone() for k, v in pyro.get_param_store().items()}
                else:
                    bad += 1

                if verbose:
                    print(f"  epoch {epoch+1:02d}  train_elbo={train_elbo:.4f}  val_nll={v_nll:.4f}  val_acc={v_acc:.4f}")

                if bad >= self.cfg.patience:
                    if verbose:
                        print("  early stopping")
                    break
            elif verbose:
                print(f"  epoch {epoch+1:02d}  train_elbo={train_elbo:.4f}")

        if best_store is not None:
            pyro.clear_param_store()
            for k, v in best_store.items():
                pyro.get_param_store()[k] = v.clone().to(self.device)

        return history
