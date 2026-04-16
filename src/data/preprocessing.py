"""
src/data/preprocessing.py

Preprocessing pipeline for Bayesian Sentiment Analysis.
Produces two representations per review:
  1. TF-IDF sparse matrix  →  fed to LDA for soft Dirichlet clustering
  2. DistilBERT [CLS] embeddings  →  fed to BNN classifier

Output (saved to data/processed/):
  - tfidf_matrix.npz         : sparse TF-IDF (N x V)
  - tfidf_vectorizer.pkl     : fitted TfidfVectorizer (needed by LDA at inference)
  - bert_embeddings.npy      : float32 array (N x 768)
  - labels.npy               : int array (N,)  0=negative, 1=positive
  - splits.json              : train/val/test indices
"""

import os
import json
import pickle
import logging
import numpy as np
import scipy.sparse as sp

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR = "data/processed"
RAW_CACHE_DIR = "data/raw"

TFIDF_CONFIG = {
    "max_features": 10_000,
    "stop_words": 'english',   # vocabulary size for LDA
    "min_df": 5,              # ignore very rare words
    "max_df": 0.7,           # ignore extremely common words
    "ngram_range": (1, 2),    # unigrams + bigrams
    "sublinear_tf": True,     # log-scale TF
}

BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_BATCH_SIZE = 64
BERT_MAX_LENGTH = 256         # truncate long reviews

RANDOM_SEED = 42
VAL_SIZE    = 0.10            # 10% validation
TEST_SIZE   = 0.10            # 10% test

# Artificial noise injection (mimics real-world data deficiencies)
# Required by the project rubric: "datos faltantes, ruido, desbalanceo"
NOISE_CONFIG = {
    "missing_rate": 0.02,     # 2% of reviews replaced with empty string
    "ood_rate":     0.05,     # 5% OOD samples injected (saved separately)
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Load raw IMDb data
# ---------------------------------------------------------------------------

def load_imdb() -> tuple[list[str], list[int]]:
    """
    Downloads IMDb via HuggingFace datasets.
    Returns (texts, labels) where label 0=negative, 1=positive.
    """
    log.info("Loading IMDb dataset...")
    dataset = load_dataset("imdb", cache_dir=RAW_CACHE_DIR)

    # Seleccionamos el subconjunto
    train_data = dataset["train"].shuffle(seed=42).select(range(5000))
    test_data = dataset["test"].shuffle(seed=42).select(range(5000))

    # EXTRAEMOS LAS COLUMNAS COMO LISTAS DE PYTHON
    texts = list(train_data["text"]) + list(test_data["text"])
    labels = list(train_data["label"]) + list(test_data["label"])

    log.info(f"  Total samples (subset): {len(texts)}")
    return texts, labels

# ---------------------------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------------------------

import re

def clean_text(text: str) -> str:
    """
    Minimal cleaning that preserves semantic content.
    We intentionally keep negations ('not', 'never') and punctuation
    that carries sentiment signal.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags (IMDb has these)
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z0-9\s\'\-]", " ", text)  # keep letters, digits, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. Noise injection  (aleatoric uncertainty source)
# ---------------------------------------------------------------------------

OOD_SAMPLES = [
    "This is a product review about a blender that works great.",
    "The quarterly earnings exceeded analyst expectations by 12%.",
    "El clima en Madrid es perfecto para salir a correr.",
    "As a nurse, I recommend washing hands frequently.",
    "git commit -m 'fix: resolve merge conflict in main branch'",
]

def inject_noise(
    texts: list[str],
    labels: list[int],
    rng: np.random.Generator,
) -> tuple[list[str], list[int], list[int]]:
    """
    Injects two types of deficiencies:
      - Missing data : some reviews replaced with empty string
      - OOD samples  : out-of-domain texts appended with label=-1

    Returns (noisy_texts, noisy_labels, ood_indices).
    """
    texts  = list(texts)
    labels = list(labels)
    n      = len(texts)

    # Missing data
    missing_idx = rng.choice(n, size=int(n * NOISE_CONFIG["missing_rate"]), replace=False)
    for i in missing_idx:
        texts[i] = ""
    log.info(f"  Injected {len(missing_idx)} missing samples")

    # OOD samples
    n_ood     = int(n * NOISE_CONFIG["ood_rate"])
    ood_texts = (OOD_SAMPLES * (n_ood // len(OOD_SAMPLES) + 1))[:n_ood]
    ood_start = len(texts)
    texts    += ood_texts
    labels   += [-1] * n_ood
    ood_indices = list(range(ood_start, ood_start + n_ood))
    log.info(f"  Injected {n_ood} OOD samples")

    return texts, labels, ood_indices


# ---------------------------------------------------------------------------
# 4. TF-IDF  (input for LDA)
# ---------------------------------------------------------------------------

def build_tfidf(
    texts: list[str],
    fit_on_indices: list[int],
) -> tuple[sp.csr_matrix, TfidfVectorizer]:
    """
    Fits TF-IDF on training texts only (no data leakage),
    then transforms all texts.

    Args:
        texts           : full list of cleaned texts
        fit_on_indices  : indices of the training split

    Returns:
        tfidf_matrix : sparse (N x vocab_size)
        vectorizer   : fitted TfidfVectorizer
    """
    log.info("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)

    train_texts = [texts[i] for i in fit_on_indices]
    vectorizer.fit(train_texts)

    tfidf_matrix = vectorizer.transform(texts)
    log.info(f"  TF-IDF shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


# ---------------------------------------------------------------------------
# 5. DistilBERT embeddings  (input for BNN)
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def build_bert_embeddings(texts: list[str]) -> np.ndarray:
    """
    Extracts [CLS] token embeddings from DistilBERT.
    Empty strings (missing data) get a zero vector — the BNN will
    naturally assign high epistemic uncertainty to these.

    Returns:
        embeddings : float32 array of shape (N, 768)
    """
    log.info("Extracting DistilBERT embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model     = DistilBertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()

    # Replace empty strings with a [UNK] placeholder so tokenizer doesn't crash
    safe_texts = [t if t.strip() else "[UNK]" for t in texts]

    dataset    = TextDataset(safe_texts, tokenizer, BERT_MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BERT_BATCH_SIZE, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Encoding batches"):
            batch  = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            # [CLS] token is the first token of last hidden state
            cls_embeddings = output.last_hidden_state[:, 0, :]  # (B, 768)
            all_embeddings.append(cls_embeddings.cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    log.info(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# 6. Train / Val / Test split
# ---------------------------------------------------------------------------

def make_splits(
    n_total: int,
    labels: list[int],
    ood_indices: list[int],
    rng_seed: int,
) -> dict[str, list[int]]:
    """
    Stratified split that excludes OOD samples from train/val/test.
    OOD indices are stored separately for evaluation only.

    Returns dict with keys: 'train', 'val', 'test', 'ood'
    """
    in_dist_idx = [i for i in range(n_total) if i not in set(ood_indices)]
    in_dist_labels = [labels[i] for i in in_dist_idx]

    # First split off test
    train_val_idx, test_idx = train_test_split(
        in_dist_idx,
        test_size=TEST_SIZE,
        stratify=in_dist_labels,
        random_state=rng_seed,
    )
    train_val_labels = [labels[i] for i in train_val_idx]

    # Then split val from train
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        stratify=train_val_labels,
        random_state=rng_seed,
    )

    log.info(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}  OOD: {len(ood_indices)}")
    return {
        "train": train_idx,
        "val":   val_idx,
        "test":  test_idx,
        "ood":   ood_indices,
    }


# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    tfidf_matrix:  sp.csr_matrix,
    vectorizer:    TfidfVectorizer,
    bert_embeddings: np.ndarray,
    labels:        list[int],
    splits:        dict[str, list[int]],
    ood_texts:     list[str],
) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("data/ood", exist_ok=True)

    # TF-IDF
    sp.save_npz(f"{PROCESSED_DIR}/tfidf_matrix.npz", tfidf_matrix)
    with open(f"{PROCESSED_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # BERT embeddings
    np.save(f"{PROCESSED_DIR}/bert_embeddings.npy", bert_embeddings)

    # Labels
    np.save(f"{PROCESSED_DIR}/labels.npy", np.array(labels, dtype=np.int32))

    # Splits
    with open(f"{PROCESSED_DIR}/splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # OOD texts (for evaluation notebook)
    ood_idx = splits["ood"]
    with open("data/ood/ood_texts.json", "w") as f:
        json.dump([ood_texts[i] for i in range(len(ood_idx))], f, indent=2)

    log.info(f"All outputs saved to '{PROCESSED_DIR}/'")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def run_preprocessing() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # Load
    texts, labels = load_imdb()

    # Clean
    log.info("Cleaning texts...")
    texts = [clean_text(t) for t in texts]

    # Inject noise
    log.info("Injecting artificial noise...")
    texts, labels, ood_indices = inject_noise(texts, labels, rng)

    # Split (needed before fitting TF-IDF to avoid leakage)
    log.info("Computing splits...")
    splits = make_splits(len(texts), labels, ood_indices, RANDOM_SEED)

    # TF-IDF (fit on train only)
    tfidf_matrix, vectorizer = build_tfidf(texts, splits["train"])

    # DistilBERT embeddings
    bert_embeddings = build_bert_embeddings(texts)

    # Save
    ood_texts = [texts[i] for i in ood_indices]
    save_outputs(tfidf_matrix, vectorizer, bert_embeddings, labels, splits, ood_texts)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
