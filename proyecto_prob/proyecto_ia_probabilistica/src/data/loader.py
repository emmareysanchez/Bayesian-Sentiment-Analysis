"""Data loaders for all model variants."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

PROCESSED_DIR = Path("data/processed")
OOD_DIR = Path("data/ood")


# -----------------------------
# Raw loaders
# -----------------------------

def load_raw(processed_dir: str | Path = PROCESSED_DIR) -> Dict[str, np.ndarray]:
    processed_dir = Path(processed_dir)
    with open(processed_dir / "splits.json") as f:
        splits = json.load(f)
    return {
        "bert": np.load(processed_dir / "bert_embeddings.npy"),
        "labels": np.load(processed_dir / "labels.npy"),
        "true_labels": np.load(processed_dir / "true_labels.npy"),
        "splits": splits,
    }


def load_tfidf(processed_dir: str | Path = PROCESSED_DIR) -> Tuple[sp.csr_matrix, object]:
    processed_dir = Path(processed_dir)
    X = sp.load_npz(processed_dir / "tfidf_matrix.npz")
    with open(processed_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return X, vec


def load_ood(ood_dir: str | Path = OOD_DIR) -> Dict[str, np.ndarray]:
    ood_dir = Path(ood_dir)
    with open(ood_dir / "ood_metadata.json") as f:
        meta = json.load(f)
    return {
        "bert": np.load(ood_dir / "ood_embeddings.npy"),
        "tfidf": sp.load_npz(ood_dir / "ood_tfidf.npz"),
        "sources": meta["sources"],
    }


# -----------------------------
# Split helper
# -----------------------------

def split_tfidf(X: sp.csr_matrix, splits: Dict[str, list]) -> Dict[str, sp.csr_matrix]:
    return {k: X[idx] for k, idx in splits.items()}


def split_array(A: np.ndarray, splits: Dict[str, list]) -> Dict[str, np.ndarray]:
    return {k: A[idx] for k, idx in splits.items()}


# -----------------------------
# Feature composition
# -----------------------------

def compose_features(
    bert: np.ndarray,
    theta: Optional[np.ndarray] = None,
    mode: str = "bert_only",
) -> np.ndarray:
    """Compose final feature matrix according to mode.

    Modes:
      - 'bert_only'   : just BERT (N, 768)
      - 'bert_lda'    : concat BERT + theta  (N, 768+K)
      - 'lda_only'    : just theta (N, K)
    """
    if mode == "bert_only":
        return bert.astype(np.float32)
    if mode == "lda_only":
        if theta is None:
            raise ValueError("theta required for lda_only")
        return theta.astype(np.float32)
    if mode == "bert_lda":
        if theta is None:
            raise ValueError("theta required for bert_lda")
        return np.concatenate([bert, theta], axis=1).astype(np.float32)
    raise ValueError(f"Unknown mode {mode}")


# -----------------------------
# PyTorch DataLoaders
# -----------------------------

def make_loader(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    batch_size: int,
    shuffle: bool,
    theta: Optional[np.ndarray | torch.Tensor] = None,
) -> DataLoader:
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    if theta is None:
        ds = TensorDataset(x, y)
    else:
        t = torch.as_tensor(theta, dtype=torch.float32)
        ds = TensorDataset(x, y, t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# -----------------------------
# Standardizer
# -----------------------------

class Standardizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Standardizer":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return cls(mean, std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
