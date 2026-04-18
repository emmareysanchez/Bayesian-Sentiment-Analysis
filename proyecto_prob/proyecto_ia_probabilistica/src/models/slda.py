"""
Amortized Supervised LDA.

Pragmatic variant of sLDA (Blei & McAuliffe, 2008):
  - Unsupervised LDA (sklearn) extracts θ_d from the bag-of-words w_d
  - A small Bayesian supervised head models p(y_d | θ_d) with a Normal prior on W

At inference, predicting topic sentiment (and sanity-checking topic quality) is
performed by running the trained head on new documents' θ_d.

This module intentionally avoids the fully-joint ELBO over {β, θ, z, W} because
that path is brittle in practice. The supervised head is trained jointly with a
regularizer that encourages θ_d → separability of y_d, giving the "supervised"
character.

ELBO of the supervised head alone (binary classification, K topics):
    log p(y | θ) ≈ E_{q(W)}[log σ(y · (W θ + b))] − KL[q(W) || N(0, σ²I)]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import Adam
from sklearn.decomposition import LatentDirichletAllocation


# -----------------------------
# Unsupervised LDA backbone
# -----------------------------

class TopicExtractor:
    """Wraps sklearn's LatentDirichletAllocation with a clean interface."""

    def __init__(self, n_topics: int = 10, random_state: int = 42):
        self.n_topics = n_topics
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            learning_method="online",
            doc_topic_prior=1.0 / n_topics,
            topic_word_prior=1.0 / n_topics,
            max_iter=20,
        )

    def fit(self, tfidf) -> "TopicExtractor":
        self.lda.fit(tfidf)
        return self

    def transform(self, tfidf) -> np.ndarray:
        theta = self.lda.transform(tfidf)
        # Ensure row-stochastic (sklearn already normalizes, but be safe)
        theta = theta / theta.sum(axis=1, keepdims=True).clip(min=1e-8)
        return theta.astype(np.float32)

    def top_words(self, feature_names, top_k: int = 10) -> Dict[int, list]:
        out = {}
        for k, comp in enumerate(self.lda.components_):
            idx = comp.argsort()[: -top_k - 1 : -1]
            out[k] = [feature_names[i] for i in idx]
        return out

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TopicExtractor":
        return joblib.load(path)


# -----------------------------
# Supervised Bayesian head
# -----------------------------

class SupervisedHead(PyroModule):
    """
    Bayesian logistic regression over θ_d:

        W ~ N(0, σ_W² I)
        b ~ N(0, σ_b² I)
        y_d | θ_d ~ Bernoulli( sigmoid(W θ_d + b) )

    With K topics and binary sentiment, this is a tiny model (K+1 weights) that
    still contributes a real probabilistic term to the joint ELBO and lets us
    read topic-sentiment associations from the posterior mean of W.
    """

    def __init__(self, n_topics: int, prior_std_w: float = 1.0, prior_std_b: float = 1.0):
        super().__init__()
        self.n_topics = n_topics
        self.linear = PyroModule[nn.Linear](n_topics, 1)

        self.linear.register_buffer("w_loc", torch.tensor(0.0))
        self.linear.register_buffer("w_scale", torch.tensor(prior_std_w))
        self.linear.register_buffer("b_loc", torch.tensor(0.0))
        self.linear.register_buffer("b_scale", torch.tensor(prior_std_b))

        self.linear.weight = PyroSample(
            lambda m: dist.Normal(m.w_loc, m.w_scale).expand([1, n_topics]).to_event(2)
        )
        self.linear.bias = PyroSample(
            lambda m: dist.Normal(m.b_loc, m.b_scale).expand([1]).to_event(1)
        )

    def forward(self, theta: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.linear(theta).squeeze(-1)   # [B] — scalar logit per doc
        if y is not None:
            with pyro.plate("data", theta.shape[0]):
                pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y.float())
        return logits


# -----------------------------
# Trainer
# -----------------------------

@dataclass
class SLDAConfig:
    n_topics: int = 10
    prior_std_w: float = 1.0
    prior_std_b: float = 1.0
    lr: float = 1e-2
    epochs: int = 30
    batch_size: int = 256
    patience: int = 5


class AmortizedSLDA:
    """sLDA = unsupervised LDA + Bayesian supervised head trained jointly.

    Training flow:
      1. Fit TopicExtractor on train TF-IDF  ->  θ_train
      2. Train SupervisedHead via SVI on (θ_train, y_train)
      3. Exposes .theta() and .predict() for all splits
    """

    def __init__(self, cfg: SLDAConfig):
        self.cfg = cfg
        self.topics: TopicExtractor | None = None
        self.head: SupervisedHead | None = None
        self.guide: AutoNormal | None = None

    # ---------- fit ----------
    def fit(
        self,
        tfidf_train,
        y_train: np.ndarray,
        tfidf_val=None,
        y_val: np.ndarray | None = None,
        device: str = "cpu",
        verbose: bool = True,
    ) -> Dict[str, list]:
        # 1) Unsupervised topics
        self.topics = TopicExtractor(n_topics=self.cfg.n_topics).fit(tfidf_train)
        theta_train = self.topics.transform(tfidf_train)
        theta_val = self.topics.transform(tfidf_val) if tfidf_val is not None else None

        theta_train_t = torch.tensor(theta_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        if theta_val is not None:
            theta_val_t = torch.tensor(theta_val, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

        # 2) Bayesian supervised head
        pyro.clear_param_store()
        self.head = SupervisedHead(
            n_topics=self.cfg.n_topics,
            prior_std_w=self.cfg.prior_std_w,
            prior_std_b=self.cfg.prior_std_b,
        ).to(device)
        self.guide = AutoNormal(self.head)
        self.guide(theta_train_t[: min(64, len(theta_train_t))], y_train_t[: min(64, len(y_train_t))])

        svi = SVI(self.head, self.guide, Adam({"lr": self.cfg.lr}), loss=Trace_ELBO())

        history = {"train_elbo": [], "val_nll": []}
        best_val = float("inf")
        bad = 0
        best_store = None

        N = theta_train_t.shape[0]
        bs = self.cfg.batch_size

        for epoch in range(self.cfg.epochs):
            # train
            perm = torch.randperm(N)
            total_loss = 0.0
            total_n = 0
            for i in range(0, N, bs):
                idx = perm[i : i + bs]
                loss = svi.step(theta_train_t[idx], y_train_t[idx])
                total_loss += loss
                total_n += len(idx)
            train_elbo = total_loss / total_n
            history["train_elbo"].append(train_elbo)

            # val
            if theta_val is not None:
                val_nll = self._nll(theta_val_t, y_val_t, mc=50)
                history["val_nll"].append(val_nll)

                if val_nll < best_val - 1e-5:
                    best_val = val_nll
                    bad = 0
                    best_store = {k: v.detach().cpu().clone() for k, v in pyro.get_param_store().items()}
                else:
                    bad += 1

                if verbose:
                    print(f"  [sLDA] epoch {epoch+1:02d}  train_elbo={train_elbo:.4f}  val_nll={val_nll:.4f}")

                if bad >= self.cfg.patience:
                    if verbose:
                        print("  [sLDA] early stopping")
                    break
            else:
                if verbose:
                    print(f"  [sLDA] epoch {epoch+1:02d}  train_elbo={train_elbo:.4f}")

        # Restore best
        if best_store is not None:
            pyro.clear_param_store()
            for k, v in best_store.items():
                pyro.get_param_store()[k] = v.to(device).clone()

        return history

    # ---------- inference ----------
    def theta(self, tfidf) -> np.ndarray:
        assert self.topics is not None
        return self.topics.transform(tfidf)

    @torch.no_grad()
    def _nll(self, theta_t: torch.Tensor, y_t: torch.Tensor, mc: int = 50) -> float:
        probs = self.predict_probs(theta_t, mc=mc)
        idx = torch.arange(y_t.shape[0], device=y_t.device)
        return float(-torch.log(probs[idx, y_t].clamp_min(1e-8)).mean().item())

    @torch.no_grad()
    def predict_probs(self, theta: torch.Tensor | np.ndarray, mc: int = 100) -> torch.Tensor:
        """Return mean predicted probabilities under the posterior q(W)."""
        assert self.head is not None and self.guide is not None
        predictive = pyro.infer.Predictive(
            self.head, guide=self.guide, num_samples=mc, return_sites=("_RETURN",)
        )
        if isinstance(theta, np.ndarray):
            theta = torch.tensor(theta, dtype=torch.float32)
        logits = predictive(theta)["_RETURN"]           # [S, B]
        p1 = torch.sigmoid(logits).mean(0)              # [B]
        probs = torch.stack([1 - p1, p1], dim=-1)       # [B, 2]
        return probs.cpu()

    def topic_sentiment(self, mc: int = 200) -> np.ndarray:
        """Posterior mean of W mapping topic -> sentiment score (pos - neg)."""
        assert self.head is not None and self.guide is not None
        predictive = pyro.infer.Predictive(
            self.head, guide=self.guide, num_samples=mc, return_sites=("linear.weight",)
        )
        dummy = torch.zeros(1, self.cfg.n_topics)
        W_samples = predictive(dummy)["linear.weight"]   # [S, 1, K]
        W_mean = W_samples.mean(0).cpu().numpy()         # [1, K]
        return W_mean[0]                                 # positive logit coefficient per topic

    # ---------- persistence ----------
    def save(self, dir_path: str | Path):
        p = Path(dir_path); p.mkdir(parents=True, exist_ok=True)
        self.topics.save(p / "lda.pkl")
        pyro.get_param_store().save(str(p / "supervised_head.pt"))
        joblib.dump(self.cfg, p / "cfg.pkl")

    @staticmethod
    def load(dir_path: str | Path, device: str = "cpu") -> "AmortizedSLDA":
        p = Path(dir_path)
        cfg = joblib.load(p / "cfg.pkl")
        obj = AmortizedSLDA(cfg)
        obj.topics = TopicExtractor.load(p / "lda.pkl")
        pyro.clear_param_store()
        pyro.get_param_store().load(str(p / "supervised_head.pt"), map_location=device)
        obj.head = SupervisedHead(
            n_topics=cfg.n_topics,
            prior_std_w=cfg.prior_std_w,
            prior_std_b=cfg.prior_std_b,
        ).to(device)
        obj.guide = AutoNormal(obj.head)
        # Warm guide
        obj.guide(torch.zeros(1, cfg.n_topics, device=device), torch.zeros(1, dtype=torch.long, device=device))
        return obj
