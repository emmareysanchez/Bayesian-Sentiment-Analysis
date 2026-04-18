"""
Preprocessing pipeline — extended version.

Produces:
  data/processed/
    bert_embeddings.npy   (N_id, 768)   in-distribution samples
    tfidf_matrix.npz      sparse (N_id, V)
    tfidf_vectorizer.pkl
    labels.npy            (N_id,)  values in {0, 1, ..., -2}   -2 = label-noise flipped
    true_labels.npy       (N_id,)  ground-truth labels (no flip)
    splits.json           {train, val, test} indices on in-distribution
  data/ood/
    ood_embeddings.npy    (N_ood, 768)
    ood_metadata.json     source per sample (amazon / agnews / multilingual)

Sources of realistic noise:
  - Label noise: flip 10% of TRAIN labels (short reviews flipped preferentially)
  - Missing: truncate 5% of reviews to first 20 tokens
  - Typos: corrupt 3% of tokens in 8% of reviews
  - Class imbalance: subsample positive class to 25% of negative class in train
  - OOD: Amazon books, AG-News, multilingual reviews
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
OOD_DIR = Path("data/ood")

BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_BATCH_SIZE = 64
BERT_MAX_LENGTH = 256

TFIDF_CONFIG = dict(
    max_features=10_000,
    stop_words="english",
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

# -----------------------------
# Cleaning
# -----------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Noise
# -----------------------------

def inject_label_noise(
    labels: np.ndarray, texts: List[str], rate: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip ~rate of labels, biased toward short reviews.

    Returns (noisy_labels, flip_mask).
    """
    N = len(labels)
    lengths = np.array([len(t.split()) for t in texts])
    # Shorter -> higher flip probability
    inv = 1.0 / (lengths + 5.0)
    prob = inv / inv.sum()
    n_flip = int(N * rate)
    flip_idx = rng.choice(N, size=n_flip, replace=False, p=prob)
    noisy = labels.copy()
    noisy[flip_idx] = 1 - noisy[flip_idx]
    mask = np.zeros(N, dtype=bool)
    mask[flip_idx] = True
    log.info(f"  Flipped {n_flip}/{N} labels ({rate*100:.1f}%)")
    return noisy, mask


def inject_missing(texts: List[str], rate: float, rng: np.random.Generator) -> List[str]:
    """Truncate `rate` of reviews to first 20 tokens."""
    N = len(texts)
    n = int(N * rate)
    idx = rng.choice(N, size=n, replace=False)
    texts = list(texts)
    for i in idx:
        tokens = texts[i].split()[:20]
        texts[i] = " ".join(tokens) if tokens else ""
    log.info(f"  Truncated {n}/{N} reviews to 20 tokens")
    return texts


def inject_typos(texts: List[str], doc_rate: float, token_rate: float, rng: np.random.Generator) -> List[str]:
    """Corrupt token_rate of tokens in doc_rate of docs with random char swaps."""
    N = len(texts)
    n_docs = int(N * doc_rate)
    doc_idx = rng.choice(N, size=n_docs, replace=False)
    texts = list(texts)
    for i in doc_idx:
        toks = texts[i].split()
        if not toks:
            continue
        n_tok = max(1, int(len(toks) * token_rate))
        tok_idx = rng.choice(len(toks), size=n_tok, replace=False)
        for j in tok_idx:
            t = list(toks[j])
            if len(t) > 1:
                k = rng.integers(0, len(t) - 1)
                t[k], t[k + 1] = t[k + 1], t[k]
                toks[j] = "".join(t)
        texts[i] = " ".join(toks)
    log.info(f"  Added typos to {n_docs}/{N} docs")
    return texts


def apply_class_imbalance(
    texts: List[str],
    labels: np.ndarray,
    minority_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Subsample the positive class so that |pos| = minority_ratio * |neg|."""
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_keep = int(len(neg_idx) * minority_ratio)
    keep_pos = rng.choice(pos_idx, size=min(n_keep, len(pos_idx)), replace=False)
    keep = np.concatenate([neg_idx, keep_pos])
    rng.shuffle(keep)
    new_texts = [texts[i] for i in keep]
    new_labels = labels[keep]
    log.info(f"  Imbalance: kept {len(keep_pos)} pos / {len(neg_idx)} neg")
    return new_texts, new_labels, keep


# -----------------------------
# BERT embeddings
# -----------------------------

class _TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def build_bert_embeddings(texts: List[str]) -> np.ndarray:
    from transformers import DistilBertModel, DistilBertTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  BERT device: {device}")

    tok = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = DistilBertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()

    safe_texts = [t if t.strip() else "[UNK]" for t in texts]
    ds = _TextDataset(safe_texts, tok, BERT_MAX_LENGTH)
    dl = DataLoader(ds, batch_size=BERT_BATCH_SIZE, shuffle=False)

    out = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="  BERT"):
            batch = {k: v.to(device) for k, v in batch.items()}
            h = model(**batch).last_hidden_state[:, 0, :]
            out.append(h.cpu().float().numpy())
    return np.concatenate(out, axis=0)


# -----------------------------
# OOD sources
# -----------------------------

def load_ood_samples(size_per_source: int) -> Tuple[List[str], List[str]]:
    """Return (texts, sources) with realistic OOD data."""
    from datasets import load_dataset

    ood_texts: List[str] = []
    ood_sources: List[str] = []

    # AG News — news headlines/articles (domain shift: news vs movies)
    try:
        log.info("  Loading AG News (OOD)...")
        ag = load_dataset("ag_news", split="test")
        idx = np.random.RandomState(0).choice(len(ag), size=min(size_per_source, len(ag)), replace=False)
        for i in idx:
            ood_texts.append(ag[int(i)]["text"])
            ood_sources.append("agnews")
    except Exception as e:
        log.warning(f"  Could not load AG News: {e}")

    # Amazon product reviews — same format, different domain
    try:
        log.info("  Loading Amazon polarity (OOD)...")
        amz = load_dataset("amazon_polarity", split="test")
        idx = np.random.RandomState(1).choice(len(amz), size=min(size_per_source, len(amz)), replace=False)
        for i in idx:
            ood_texts.append(amz[int(i)]["content"])
            ood_sources.append("amazon")
    except Exception as e:
        log.warning(f"  Could not load Amazon: {e}")

    # Multilingual — fallback to hard-coded Spanish/French if dataset fails
    try:
        log.info("  Loading multilingual reviews (OOD)...")
        ml = load_dataset("amazon_reviews_multi", "es", split="test")
        idx = np.random.RandomState(2).choice(len(ml), size=min(size_per_source // 2, len(ml)), replace=False)
        for i in idx:
            ood_texts.append(ml[int(i)]["review_body"])
            ood_sources.append("spanish")
    except Exception as e:
        log.warning(f"  Could not load multilingual, using synthetic Spanish samples: {e}")
        synthetic_es = [
            "Esta película es absolutamente maravillosa, una de las mejores que he visto.",
            "Un desastre de producción, actores mediocres y guión flojo.",
            "Muy recomendable, te mantiene en tensión de principio a fin.",
            "No merece la pena gastar dinero en esta película, predecible y aburrida.",
            "Una obra maestra del cine moderno, guión impecable.",
        ] * (size_per_source // 10 + 1)
        for t in synthetic_es[: size_per_source // 2]:
            ood_texts.append(t)
            ood_sources.append("spanish_synth")

    log.info(f"  Total OOD samples: {len(ood_texts)}")
    return ood_texts, ood_sources


# -----------------------------
# Pipeline
# -----------------------------

def run(
    imdb_size: int,
    ood_size_per_source: int,
    val_size: float,
    test_size: float,
    label_noise: float,
    missing_rate: float,
    typo_doc_rate: float,
    minority_ratio: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OOD_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load IMDb ---
    log.info("Loading IMDb...")
    from datasets import load_dataset
    imdb = load_dataset("imdb")
    texts = list(imdb["train"]["text"]) + list(imdb["test"]["text"])
    labels = np.array(list(imdb["train"]["label"]) + list(imdb["test"]["label"]), dtype=np.int64)
    if imdb_size < len(texts):
        idx = rng.choice(len(texts), size=imdb_size, replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]
    log.info(f"  IMDb samples: {len(texts)}")

    # --- Clean ---
    log.info("Cleaning...")
    texts = [clean_text(t) for t in texts]

    # --- Inject realistic deficiencies (in this order matters) ---
    log.info("Injecting deficiencies...")
    texts = inject_missing(texts, missing_rate, rng)
    texts = inject_typos(texts, typo_doc_rate, token_rate=0.1, rng=rng)

    # --- Stratified split BEFORE label noise (to preserve true labels on val/test) ---
    log.info("Splitting...")
    idx_all = np.arange(len(texts))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, stratify=labels, random_state=seed
    )
    rel_val = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=rel_val, stratify=labels[idx_trainval], random_state=seed
    )
    log.info(f"  Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    # --- Apply class imbalance ONLY on train ---
    train_texts = [texts[i] for i in idx_train]
    train_labels = labels[idx_train]
    train_texts, train_labels, keep_local = apply_class_imbalance(
        train_texts, train_labels, minority_ratio, rng
    )
    idx_train_final = idx_train[keep_local]

    # --- Inject label noise ONLY on train (and record mask) ---
    true_labels = labels.copy()
    noisy_labels = labels.copy()
    train_noisy, flip_mask_local = inject_label_noise(
        train_labels, train_texts, label_noise, rng
    )
    noisy_labels[idx_train_final] = train_noisy

    # --- Reassemble keep order ---
    # Final in-distribution index order: train_final ++ val ++ test
    id_idx = np.concatenate([idx_train_final, idx_val, idx_test])
    id_texts = [texts[i] for i in id_idx]
    id_labels_noisy = noisy_labels[id_idx]
    id_labels_true = true_labels[id_idx]

    N_tr, N_va, N_te = len(idx_train_final), len(idx_val), len(idx_test)
    splits = {
        "train": list(range(0, N_tr)),
        "val": list(range(N_tr, N_tr + N_va)),
        "test": list(range(N_tr + N_va, N_tr + N_va + N_te)),
    }

    # --- TF-IDF (fit on train only, no leakage) ---
    log.info("Building TF-IDF...")
    vec = TfidfVectorizer(**TFIDF_CONFIG)
    vec.fit([id_texts[i] for i in splits["train"]])
    tfidf = vec.transform(id_texts)
    log.info(f"  TF-IDF shape: {tfidf.shape}")

    # --- BERT embeddings for in-distribution ---
    log.info("Computing BERT embeddings (in-distribution)...")
    emb_id = build_bert_embeddings(id_texts)

    # --- OOD ---
    log.info("Loading OOD...")
    ood_texts, ood_sources = load_ood_samples(ood_size_per_source)
    ood_texts = [clean_text(t) for t in ood_texts]
    log.info("Computing BERT embeddings (OOD)...")
    emb_ood = build_bert_embeddings(ood_texts)

    # --- Save ---
    log.info("Saving...")
    np.save(PROCESSED_DIR / "bert_embeddings.npy", emb_id.astype(np.float32))
    sp.save_npz(PROCESSED_DIR / "tfidf_matrix.npz", tfidf)
    with open(PROCESSED_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    np.save(PROCESSED_DIR / "labels.npy", id_labels_noisy.astype(np.int64))
    np.save(PROCESSED_DIR / "true_labels.npy", id_labels_true.astype(np.int64))
    with open(PROCESSED_DIR / "splits.json", "w") as f:
        json.dump(splits, f)

    np.save(OOD_DIR / "ood_embeddings.npy", emb_ood.astype(np.float32))
    with open(OOD_DIR / "ood_metadata.json", "w") as f:
        json.dump({"sources": ood_sources, "n": len(ood_sources)}, f)
    # Save OOD TF-IDF too (needed for sLDA on OOD)
    ood_tfidf = vec.transform(ood_texts)
    sp.save_npz(OOD_DIR / "ood_tfidf.npz", ood_tfidf)

    # Config snapshot
    with open(PROCESSED_DIR / "config.json", "w") as f:
        json.dump({
            "imdb_size": imdb_size,
            "label_noise": label_noise,
            "missing_rate": missing_rate,
            "typo_doc_rate": typo_doc_rate,
            "minority_ratio": minority_ratio,
            "seed": seed,
            "n_train": N_tr,
            "n_val": N_va,
            "n_test": N_te,
            "n_ood": len(ood_sources),
        }, f, indent=2)

    log.info("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imdb-size", type=int, default=20000)
    p.add_argument("--ood-size-per-source", type=int, default=2000)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--label-noise", type=float, default=0.10)
    p.add_argument("--missing-rate", type=float, default=0.05)
    p.add_argument("--typo-doc-rate", type=float, default=0.08)
    p.add_argument("--minority-ratio", type=float, default=0.25,
                   help="Positive = minority_ratio * negative in train")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
