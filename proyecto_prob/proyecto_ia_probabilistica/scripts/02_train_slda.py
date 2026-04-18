"""Train amortized supervised LDA (LDA + Bayesian supervised head).

Saves:
    experiments/results/slda/lda.pkl
    experiments/results/slda/supervised_head.pt
    experiments/results/slda/cfg.pkl
    experiments/results/slda/theta_{train,val,test,ood}.npy
    experiments/results/slda/topic_words.json
    experiments/results/slda/topic_sentiment.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data.loader import load_ood, load_raw, load_tfidf, split_tfidf  # noqa: E402
from src.models.slda import AmortizedSLDA, SLDAConfig  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-topics", type=int, default=10)
    ap.add_argument("--prior-std-w", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="experiments/results/slda")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load data ---
    raw = load_raw()
    tfidf, vec = load_tfidf()
    splits = raw["splits"]
    tfidf_splits = split_tfidf(tfidf, splits)
    y_train = raw["labels"][splits["train"]]
    y_val = raw["labels"][splits["val"]]
    y_test = raw["labels"][splits["test"]]

    print(f"Train TF-IDF: {tfidf_splits['train'].shape}, pos={float((y_train==1).mean()):.3f}")

    # --- Train ---
    cfg = SLDAConfig(
        n_topics=args.n_topics,
        prior_std_w=args.prior_std_w,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    slda = AmortizedSLDA(cfg)
    slda.fit(
        tfidf_train=tfidf_splits["train"],
        y_train=y_train,
        tfidf_val=tfidf_splits["val"],
        y_val=y_val,
        device=device,
    )

    # --- Compute theta for all splits and OOD ---
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    theta = {
        "train": slda.theta(tfidf_splits["train"]),
        "val":   slda.theta(tfidf_splits["val"]),
        "test":  slda.theta(tfidf_splits["test"]),
    }
    for k, v in theta.items():
        np.save(out_dir / f"theta_{k}.npy", v)
    try:
        ood = load_ood()
        theta_ood = slda.theta(ood["tfidf"])
        np.save(out_dir / "theta_ood.npy", theta_ood)
        print(f"  theta_ood: {theta_ood.shape}")
    except Exception as e:
        print(f"  (no OOD theta): {e}")

    # --- Save model + diagnostics ---
    slda.save(out_dir)
    feat_names = vec.get_feature_names_out()
    top_words = {str(k): ws for k, ws in slda.topics.top_words(feat_names, top_k=12).items()}
    topic_sent = slda.topic_sentiment(mc=200)
    with open(out_dir / "topic_words.json", "w") as f:
        json.dump(top_words, f, indent=2)
    with open(out_dir / "topic_sentiment.json", "w") as f:
        json.dump({str(k): float(v) for k, v in enumerate(topic_sent)}, f, indent=2)

    print("\nTop words per topic (sentiment score = pos - neg logit):")
    for k in range(args.n_topics):
        s = float(topic_sent[k])
        tag = "POS" if s > 0.1 else ("NEG" if s < -0.1 else "neu")
        print(f"  Topic {k:2d} [{tag} {s:+.2f}] {' '.join(top_words[str(k)][:8])}")

    # --- Quick sanity accuracy using only theta ---
    import torch as _t
    for split, y in [("val", y_val), ("test", y_test)]:
        probs = slda.predict_probs(theta[split], mc=200)
        acc = float((probs.argmax(1) == _t.tensor(y)).float().mean().item())
        print(f"  {split} accuracy from theta only: {acc:.4f}")

    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
