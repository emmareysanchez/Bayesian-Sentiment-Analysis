from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyro
import torch
from pyro.infer.autoguide import AutoNormal
from sklearn.model_selection import train_test_split

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data.loader import Standardizer, load_bow, load_tfidf
from src.data.preprocessing import (
    apply_class_imbalance,
    clean_text,
    inject_missing,
    inject_typos,
)
from src.evaluation.uncertainty import decompose_mc
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig
from src.models.deterministic import DeterministicMLP
from src.models.mc_dropout import MCDropoutMLP
from src.models.slda import AmortizedSLDA

# ------------------------------------------------------------
# Paths / constants
# ------------------------------------------------------------

MODELS_ROOT = Path("experiments/results/models")
SLDA_DIR = Path("experiments/results/slda")
OUT_PATH = Path("experiments/results/example_reviews.json")
METRICS_RAW = Path("experiments/results/evaluation/metrics_raw.csv")
CACHE_DIR = Path("experiments/results/cache")
RAW_CACHE_PATH = CACHE_DIR / "extract_examples_raw_cache.json"
FEATS_CACHE_PATH = CACHE_DIR / "extract_examples_online_features.npz"

ALL_MODELS = ["deterministic", "mc_dropout", "bnn_base", "bnn_moe", "bnn_moe_hetero"]
N_EACH = 5

# Speed / robustness knobs.
# For a demo, using the best seed per model is usually enough and much faster.
USE_BEST_SEED_ONLY = True
MC_SAMPLES = 80
BERT_BATCH_SIZE = 128
INFER_BATCH_SIZE = 512
FORCE_REBUILD_CACHE = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def _fingerprint_inputs() -> str:
    payload: dict[str, Any] = {}
    for p in [Path("data/processed/config.json"), Path("data/processed/splits.json")]:
        if p.exists():
            payload[str(p)] = p.read_text(encoding="utf-8")
    payload["models"] = sorted([p.name for p in MODELS_ROOT.iterdir() if p.is_dir()]) if MODELS_ROOT.exists() else []
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _clean_once(text: str) -> str:
    cleaned = clean_text(text)
    return cleaned if cleaned.strip() else "[UNK]"


def _available_seeds(model_dir: Path) -> list[str]:
    if not model_dir.exists():
        return []
    return sorted([s.name for s in model_dir.iterdir() if s.is_dir() and s.name.startswith("seed_")])


def _best_seed(model_name: str, seeds: list[str]) -> str | None:
    if not seeds:
        return None
    if not METRICS_RAW.exists():
        return seeds[0]

    best_seed = None
    best_acc = -1.0
    try:
        with open(METRICS_RAW, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name or row.get("split") != "test":
                    continue
                seed_name = f"seed_{row['seed']}"
                if seed_name not in seeds:
                    continue
                acc = float(row["accuracy"])
                if acc > best_acc:
                    best_acc = acc
                    best_seed = seed_name
    except Exception:
        return seeds[0]

    return best_seed if best_seed is not None else seeds[0]


def _selected_runs() -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for model_name in ALL_MODELS:
        model_dir = MODELS_ROOT / model_name
        seeds = _available_seeds(model_dir)
        if not seeds:
            continue
        if USE_BEST_SEED_ONLY:
            best = _best_seed(model_name, seeds)
            if best is not None:
                runs.append((model_name, best))
        else:
            runs.extend((model_name, seed) for seed in seeds)
    return runs


# ------------------------------------------------------------
# Rebuild the exact raw test texts
# ------------------------------------------------------------

def rebuild_test_raw_texts() -> tuple[list[str], np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint_inputs()

    if RAW_CACHE_PATH.exists() and not FORCE_REBUILD_CACHE:
        try:
            cached = json.loads(RAW_CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                print(f"Loaded raw test cache from {RAW_CACHE_PATH}")
                texts = cached["texts_raw"]
                labels = np.asarray(cached["labels"], dtype=np.int64)
                return texts, labels
        except Exception:
            pass

    cfg = json.load(open("data/processed/config.json", encoding="utf-8"))
    seed = cfg["seed"]
    imdb_size = cfg["imdb_size"]
    missing = cfg["missing_rate"]
    typo = cfg["typo_doc_rate"]
    minority = cfg["minority_ratio"]
    val_size = 0.10
    test_size = 0.10

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Loading IMDb from Hugging Face cache/dataset...")
    from datasets import load_dataset
    imdb = load_dataset("imdb")

    texts_raw = list(imdb["train"]["text"]) + list(imdb["test"]["text"])
    labels_raw = np.array(
        list(imdb["train"]["label"]) + list(imdb["test"]["label"]),
        dtype=np.int64,
    )

    idx = rng.choice(len(texts_raw), size=imdb_size, replace=False)
    texts_original = [texts_raw[i] for i in idx]
    labels = labels_raw[idx]

    # Reproduce the preprocessing only to recover the exact split logic.
    texts_processed = [_clean_once(t) for t in tqdm(texts_original, desc="Cleaning raw texts")]
    texts_processed = inject_missing(texts_processed, missing, rng)
    texts_processed = inject_typos(texts_processed, typo, token_rate=0.1, rng=rng)

    idx_all = np.arange(len(texts_processed))
    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    rel_val = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        stratify=labels[idx_trainval],
        random_state=seed,
    )

    train_texts = [texts_processed[i] for i in idx_train]
    train_labels = labels[idx_train]
    _, _, keep_local = apply_class_imbalance(train_texts, train_labels, minority, rng)
    idx_train_final = idx_train[keep_local]

    id_idx = np.concatenate([idx_train_final, idx_val, idx_test])
    id_texts_original = [texts_original[i] for i in id_idx]
    id_labels = labels[id_idx]

    splits = json.load(open("data/processed/splits.json", encoding="utf-8"))
    test_local = splits["test"]

    test_texts_raw = [id_texts_original[i] for i in test_local]
    test_labels = id_labels[test_local]

    RAW_CACHE_PATH.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "texts_raw": test_texts_raw,
                "labels": test_labels.tolist(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved raw test cache to {RAW_CACHE_PATH}")
    print(f"Recovered {len(test_texts_raw)} raw test texts.")
    return test_texts_raw, test_labels


# ------------------------------------------------------------
# Online inference path = same logic as the app
# ------------------------------------------------------------

def load_bert(device: torch.device):
    from transformers import DistilBertModel, DistilBertTokenizer

    tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    model.to(device)
    return tok, model


def slda_vectorizer_type() -> str:
    meta_path = SLDA_DIR / "slda_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("vectorizer", "bow")
    return "bow"


def load_slda():
    if not (SLDA_DIR / "lda.pkl").exists():
        return None, None
    slda = AmortizedSLDA.load(SLDA_DIR, device="cpu")
    vec_type = slda_vectorizer_type()
    if vec_type == "tfidf":
        _, vec = load_tfidf()
    else:
        _, vec = load_bow()
    return slda, vec


def encode_texts_online(
    texts: list[str],
    tok,
    bert,
    device: torch.device,
    batch_size: int = BERT_BATCH_SIZE,
) -> np.ndarray:
    cleaned = [_clean_once(t) for t in texts]
    embs: list[np.ndarray] = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(cleaned), batch_size), desc=f"Embedding texts on {device.type}"):
            batch = cleaned[i:i + batch_size]
            enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = bert(**enc).last_hidden_state[:, 0, :]
            embs.append(h.detach().cpu().float().numpy())

    return np.concatenate(embs, axis=0)


def compute_theta_online(texts: list[str], slda, vec) -> np.ndarray | None:
    if slda is None or vec is None:
        return None
    cleaned = [_clean_once(t) for t in texts]
    X = vec.transform(cleaned)
    return slda.theta(X)


def maybe_load_cached_features() -> tuple[np.ndarray, np.ndarray | None] | None:
    if not FEATS_CACHE_PATH.exists() or FORCE_REBUILD_CACHE:
        return None
    try:
        cached = np.load(FEATS_CACHE_PATH, allow_pickle=False)
        emb = cached["emb"]
        theta = cached["theta"] if "theta" in cached.files else None
        print(f"Loaded online feature cache from {FEATS_CACHE_PATH}")
        return emb, theta
    except Exception:
        return None


def save_cached_features(emb: np.ndarray, theta: np.ndarray | None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"emb": emb}
    if theta is not None:
        payload["theta"] = theta
    np.savez_compressed(FEATS_CACHE_PATH, **payload)
    print(f"Saved online feature cache to {FEATS_CACHE_PATH}")


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_model(name: str, seed_name: str, device: torch.device) -> dict | None:
    seed_dir = MODELS_ROOT / name / seed_name
    if not seed_dir.exists():
        return None

    if name in ("deterministic", "mc_dropout"):
        ck = torch.load(seed_dir / "state.pt", map_location="cpu", weights_only=False)
        input_dim = ck["input_dim"]
        if name == "deterministic":
            model = DeterministicMLP(input_dim=input_dim)
        else:
            model = MCDropoutMLP(input_dim=input_dim, dropout_p=ck.get("dropout_p", 0.2))
        model.load_state_dict(ck["state"])
        model.eval()
        model.to(device)
        scaler = Standardizer(ck["scaler_mean"], ck["scaler_std"])
        return {
            "name": name,
            "seed": seed_name,
            "kind": name,
            "model": model,
            "guide": None,
            "scaler": scaler,
            "use_theta": False,
            "device": device,
        }

    meta = torch.load(seed_dir / "meta.pt", map_location="cpu", weights_only=False)
    cfg = BayesianMoEConfig(**meta["cfg"])

    pyro.clear_param_store()
    state = torch.load(seed_dir / "pyro_store.pt", map_location=device, weights_only=False)
    pyro.get_param_store().set_state(state)

    model = BayesianMoE(cfg)
    model.to(device)
    guide = AutoNormal(model)

    dummy_x = torch.zeros(2, cfg.input_dim, device=device)
    dummy_y = torch.zeros(2, dtype=torch.long, device=device)
    if meta["use_theta"]:
        dummy_th = torch.ones(2, cfg.n_experts, device=device) / cfg.n_experts
        guide(dummy_x, dummy_th, dummy_y)
    else:
        guide(dummy_x, None, dummy_y)

    scaler = Standardizer(meta["scaler_mean"], meta["scaler_std"])
    param_snapshot = {k: v.detach().clone() for k, v in pyro.get_param_store().items()}
    return {
        "name": name,
        "seed": seed_name,
        "kind": "bnn",
        "model": model,
        "guide": guide,
        "scaler": scaler,
        "use_theta": bool(meta["use_theta"]),
        "cfg": cfg,
        "device": device,
        "_param_snapshot": param_snapshot,
    }


# ------------------------------------------------------------
# Batched prediction
# ------------------------------------------------------------

def predict_batch(
    bundle: dict,
    emb: np.ndarray,
    theta: np.ndarray | None,
    mc_samples: int = MC_SAMPLES,
    batch_size: int = INFER_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    device = bundle["device"]

    # Standardize on CPU because the scaler was saved CPU-side.
    emb_t_cpu = torch.tensor(emb, dtype=torch.float32)
    emb_std_cpu = bundle["scaler"].transform(emb_t_cpu)

    all_probs: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []

    if bundle["kind"] == "bnn" and "_param_snapshot" in bundle:
        pyro.clear_param_store()
        for k, v in bundle["_param_snapshot"].items():
            pyro.get_param_store()[k] = v.detach().clone()

    iterator = range(0, emb_std_cpu.shape[0], batch_size)
    iterator = tqdm(iterator, desc=f"Predicting {bundle['name']}/{bundle['seed']}", leave=False)

    for i in iterator:
        xb = emb_std_cpu[i:i + batch_size].to(device)

        if bundle["use_theta"]:
            if theta is None:
                raise ValueError(f"theta is required for model {bundle['name']}")
            tb = torch.tensor(theta[i:i + batch_size], dtype=torch.float32, device=device)
        else:
            tb = None

        if bundle["kind"] == "deterministic":
            with torch.inference_mode():
                probs = torch.softmax(bundle["model"](xb), dim=-1)
            mc = probs.unsqueeze(0)
        elif bundle["kind"] == "mc_dropout":
            with torch.inference_mode():
                logits = bundle["model"].mc_forward(xb, mc_samples=mc_samples)
                mc = torch.softmax(logits, dim=-1)
        else:
            predictive = pyro.infer.Predictive(
                bundle["model"],
                guide=bundle["guide"],
                num_samples=mc_samples,
                return_sites=("_RETURN",),
            )
            if tb is not None:
                logits = predictive(xb, tb)["_RETURN"]
            else:
                logits = predictive(xb)["_RETURN"]
            mc = torch.softmax(logits, dim=-1)

        probs_mean = mc.mean(0).detach().cpu().numpy()
        decomp = decompose_mc(mc)
        entropy = decomp["predictive_entropy"].detach().cpu().numpy()

        all_probs.append(probs_mean)
        all_entropy.append(entropy)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_entropy, axis=0)


# ------------------------------------------------------------
# Main selection logic
# ------------------------------------------------------------

def is_clear_enough(text: str) -> bool:
    n_chars = len(text.strip())
    n_words = len(text.split())
    if n_chars < 120:
        return False
    if n_chars > 1800:
        return False
    if n_words < 25:
        return False
    return True


def pick_examples(
    texts_raw: list[str],
    y: np.ndarray,
    emb_online: np.ndarray,
    theta_online: np.ndarray | None,
    all_correct: np.ndarray,
    max_entropy: np.ndarray,
    mean_entropy: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    readable = np.array([is_clear_enough(t) for t in texts_raw], dtype=bool)

    def pick(label: int, n: int) -> list[dict[str, Any]]:
        mask = all_correct & readable & (y == label)
        cands = np.where(mask)[0]
        ranked = sorted(cands, key=lambda i: (max_entropy[i], mean_entropy[i]))
        chosen = ranked[:n]
        out: list[dict[str, Any]] = []
        for i in chosen:
            record: dict[str, Any] = {
                "text": texts_raw[i].strip(),
                "label": int(label),
                "max_entropy": float(max_entropy[i]),
                "mean_entropy": float(mean_entropy[i]),
                "embedding": emb_online[i].astype(np.float32).tolist(),
            }
            if theta_online is not None:
                record["theta"] = theta_online[i].astype(np.float32).tolist()
            out.append(record)
        return out

    return pick(1, N_EACH), pick(0, N_EACH)


def main() -> None:
    print(f"Using device: {DEVICE}")
    print(f"MC_SAMPLES={MC_SAMPLES} | USE_BEST_SEED_ONLY={USE_BEST_SEED_ONLY}")

    runs = _selected_runs()
    if not runs:
        raise RuntimeError("No model/seed runs found.")
    print(f"Selected runs: {runs}")

    texts_raw, y = rebuild_test_raw_texts()

    cached = maybe_load_cached_features()
    if cached is None:
        tok, bert = load_bert(DEVICE)
        print("Encoding test texts online...")
        emb_online = encode_texts_online(texts_raw, tok, bert, device=DEVICE)

        needs_theta = any(name in {"bnn_moe", "bnn_moe_hetero"} for name, _ in runs)
        if needs_theta:
            print("Computing theta online...")
            slda, vec = load_slda()
            theta_online = compute_theta_online(texts_raw, slda, vec)
        else:
            theta_online = None

        save_cached_features(emb_online, theta_online)
        del bert
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    else:
        emb_online, theta_online = cached

    all_correct = np.ones(len(texts_raw), dtype=bool)
    max_entropy = np.zeros(len(texts_raw), dtype=float)
    mean_entropy_accum = np.zeros(len(texts_raw), dtype=float)
    n_runs = 0

    for model_name, seed_name in tqdm(runs, desc="Models"):
        bundle = load_model(model_name, seed_name, DEVICE)
        if bundle is None:
            continue

        probs, entropy = predict_batch(
            bundle=bundle,
            emb=emb_online,
            theta=theta_online if bundle["use_theta"] else None,
            mc_samples=MC_SAMPLES,
            batch_size=INFER_BATCH_SIZE,
        )
        preds = probs.argmax(axis=1)
        correct = preds == y

        all_correct &= correct
        max_entropy = np.maximum(max_entropy, entropy)
        mean_entropy_accum += entropy
        n_runs += 1

        print(f"  {model_name}/{seed_name}  acc={correct.mean():.3f}")

        del bundle
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if n_runs == 0:
        raise RuntimeError("No model/seed runs finished correctly.")

    mean_entropy = mean_entropy_accum / n_runs
    pos, neg = pick_examples(
        texts_raw=texts_raw,
        y=y,
        emb_online=emb_online,
        theta_online=theta_online,
        all_correct=all_correct,
        max_entropy=max_entropy,
        mean_entropy=mean_entropy,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "positive": pos,
                "negative": neg,
                "note": (
                    f"Selected using the same online inference path as the app. "
                    f"Correct in all {n_runs} selected runs. "
                    f"Ranked by lowest worst-case predictive entropy."
                ),
                "device": str(DEVICE),
                "mc_samples": MC_SAMPLES,
                "use_best_seed_only": USE_BEST_SEED_ONLY,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved to {OUT_PATH}")
    print(f"Always-correct examples: {all_correct.sum()} / {len(texts_raw)}")
    print(f"Selected positives: {len(pos)} | negatives: {len(neg)}")


if __name__ == "__main__":
    main()
