"""
src/data/loader.py

PyTorch Dataset classes for feeding preprocessed data
into the LDA and BNN training loops.

Usage:
    from src.data.loader import load_splits, BNNDataset

    train_ds, val_ds, test_ds = load_splits()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
"""

import json
import pickle
import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset, DataLoader

PROCESSED_DIR = "data/processed"


# ---------------------------------------------------------------------------
# Load preprocessed arrays
# ---------------------------------------------------------------------------

def load_splits(
    processed_dir: str = PROCESSED_DIR,
) -> tuple["BNNDataset", "BNNDataset", "BNNDataset"]:
    """
    Loads preprocessed embeddings, labels and splits from disk.
    Returns train, val, test BNNDataset objects ready for DataLoader.
    """
    bert_embeddings = np.load(f"{processed_dir}/bert_embeddings.npy")
    labels          = np.load(f"{processed_dir}/labels.npy")

    with open(f"{processed_dir}/splits.json") as f:
        splits = json.load(f)

    # Filter out OOD samples (label == -1) from in-distribution sets
    def make_dataset(indices):
        idx    = [i for i in indices if labels[i] != -1]
        X      = bert_embeddings[idx]
        y      = labels[idx]
        return BNNDataset(X, y)

    train_ds = make_dataset(splits["train"])
    val_ds   = make_dataset(splits["val"])
    test_ds  = make_dataset(splits["test"])

    return train_ds, val_ds, test_ds


def load_tfidf_splits(
    processed_dir: str = PROCESSED_DIR,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Loads TF-IDF matrix and returns train/val/test sparse splits.
    Used by the LDA module.

    Returns:
        X_train, X_val, X_test : sparse csr_matrix
        labels                  : full label array (includes OOD as -1)
    """
    tfidf = sp.load_npz(f"{processed_dir}/tfidf_matrix.npz")
    labels = np.load(f"{processed_dir}/labels.npy")

    with open(f"{processed_dir}/splits.json") as f:
        splits = json.load(f)

    train_idx = [i for i in splits["train"] if labels[i] != -1]
    val_idx   = [i for i in splits["val"]   if labels[i] != -1]
    test_idx  = [i for i in splits["test"]  if labels[i] != -1]

    return (
        tfidf[train_idx],
        tfidf[val_idx],
        tfidf[test_idx],
        labels,
    )


def load_vectorizer(processed_dir: str = PROCESSED_DIR) -> object:
    """Loads the fitted TF-IDF vectorizer (needed at inference time)."""
    with open(f"{processed_dir}/tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)


def load_ood_embeddings(processed_dir: str = PROCESSED_DIR) -> np.ndarray:
    """
    Returns BERT embeddings for OOD samples only.
    Used in the evaluation notebook to verify the BNN assigns
    high epistemic uncertainty to out-of-distribution inputs.
    """
    bert_embeddings = np.load(f"{processed_dir}/bert_embeddings.npy")
    labels          = np.load(f"{processed_dir}/labels.npy")

    with open(f"{processed_dir}/splits.json") as f:
        splits = json.load(f)

    ood_idx = splits["ood"]
    return bert_embeddings[ood_idx]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BNNDataset(Dataset):
    """
    Simple Dataset wrapping BERT embeddings and binary labels.

    Args:
        embeddings : float32 array of shape (N, 768)
        labels     : int array of shape (N,)  — 0 or 1
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class BNNDatasetWithTopics(Dataset):
    """
    Extended Dataset that concatenates BERT embeddings with
    LDA topic vectors θ before feeding into the BNN.

    This is the main input representation used in the full pipeline:
        input = concat([bert_cls (768-dim), theta (K-dim)])

    Args:
        embeddings  : float32 array (N, 768)
        topic_vecs  : float32 array (N, K)   — soft Dirichlet assignments
        labels      : int array (N,)
    """

    def __init__(
        self,
        embeddings:  np.ndarray,
        topic_vecs:  np.ndarray,
        labels:      np.ndarray,
    ):
        self.X = torch.tensor(
            np.concatenate([embeddings, topic_vecs], axis=1),
            dtype=torch.float32,
        )
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    batch_size: int = 32,
    processed_dir: str = PROCESSED_DIR,
    topic_vecs: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function that returns ready-to-use DataLoaders.

    Args:
        batch_size   : mini-batch size
        processed_dir: path to processed data
        topic_vecs   : if provided (N x K array), uses BNNDatasetWithTopics
                       so that LDA soft assignments are concatenated to BERT embeddings

    Returns:
        train_loader, val_loader, test_loader
    """
    bert_embeddings = np.load(f"{processed_dir}/bert_embeddings.npy")
    labels          = np.load(f"{processed_dir}/labels.npy")

    with open(f"{processed_dir}/splits.json") as f:
        splits = json.load(f)

    def _get(indices):
        idx = [i for i in indices if labels[i] != -1]
        X   = bert_embeddings[idx]
        y   = labels[idx]
        if topic_vecs is not None:
            T = topic_vecs[idx]
            return BNNDatasetWithTopics(X, T, y)
        return BNNDataset(X, y)

    train_ds = _get(splits["train"])
    val_ds   = _get(splits["val"])
    test_ds  = _get(splits["test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
