from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# -------------------------------------------------------------
# Default project configuration
# Edit these only if your project uses different paths.
# -------------------------------------------------------------
DEFAULT_OUTPUT = "experiments/results/prepared_features.pt"
DEFAULT_BATCH_SIZE = 64
DEFAULT_BERT_DIM = 768
DEFAULT_N_TOPICS = 10
DEFAULT_LDA_MODEL_PATH = "experiments/results/lda_model.pkl"


def loader_to_tensors(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    # Preserve split order from the dataset itself so train labels stay aligned
    # across repeated calls to make_dataloaders (train loader uses shuffle=True).
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "X") and hasattr(ds, "y"):
        return ds.X.detach().cpu().float(), ds.y.detach().cpu().long()

    xs = []
    ys = []
    for xb, yb in loader:
        xs.append(xb.detach().cpu())
        ys.append(yb.detach().cpu())
    x = torch.cat(xs, dim=0).float()
    y = torch.cat(ys, dim=0).long()
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single .pt bundle with BERT-only and BERT+LDA features using the exact data "
            "pipeline already used in the project's notebooks."
        )
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bert-dim", type=int, default=DEFAULT_BERT_DIM)
    parser.add_argument("--n-topics", type=int, default=DEFAULT_N_TOPICS)
    parser.add_argument("--lda-model-path", type=str, default=DEFAULT_LDA_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.getcwd().endswith("notebooks"):
        os.chdir("..")

    root = os.getcwd()
    if root not in sys.path:
        sys.path.append(root)

    required_files = [
        "data/processed/bert_embeddings.npy",
        "data/processed/tfidf_matrix.npz",
        "data/processed/labels.npy",
    ]
    missing = [p for p in required_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed files required by your notebooks:\n- " + "\n- ".join(missing)
        )

    from src.data.loader import make_dataloaders, load_tfidf_splits
    from src.models.lda import TopicModeler

    print(f"Working directory: {root}")
    print("Using the same project data pipeline as the notebooks.")

    print("\n[1/4] Building BERT-only splits from make_dataloaders(batch_size=...)")
    train_loader_bert, val_loader_bert, test_loader_bert = make_dataloaders(batch_size=args.batch_size)

    x_train_bert, y_train = loader_to_tensors(train_loader_bert)
    x_val_bert, y_val = loader_to_tensors(val_loader_bert)
    x_test_bert, y_test = loader_to_tensors(test_loader_bert)

    print(f"x_train_bert_only: {tuple(x_train_bert.shape)}")
    print(f"x_val_bert_only:   {tuple(x_val_bert.shape)}")
    print(f"x_test_bert_only:  {tuple(x_test_bert.shape)}")

    if x_train_bert.shape[1] != args.bert_dim:
        raise ValueError(
            f"Expected BERT dim {args.bert_dim}, but the loader returned {x_train_bert.shape[1]}. "
            "If your embeddings do not have size 768, edit DEFAULT_BERT_DIM at the top of the script."
        )

    print("\n[2/4] Loading TF-IDF splits for LDA")
    X_tr, X_val, X_te, _ = load_tfidf_splits()
    print(f"TF-IDF train shape: {X_tr.shape}")
    print(f"TF-IDF val shape:   {X_val.shape}")
    print(f"TF-IDF test shape:  {X_te.shape}")

    print("\n[3/4] Loading or training the LDA model")
    lda_model_path = Path(args.lda_model_path)
    lda_model_path.parent.mkdir(parents=True, exist_ok=True)

    if lda_model_path.exists():
        lda_modeler = TopicModeler.load(str(lda_model_path))
        print(f"Loaded LDA model from: {lda_model_path}")
    else:
        print(f"No LDA model found at: {lda_model_path}")
        print(f"Training a new LDA model with n_topics={args.n_topics} on the TF-IDF train split...")
        lda_modeler = TopicModeler(n_components=args.n_topics)
        lda_modeler.fit(X_tr)
        lda_modeler.save(str(lda_model_path))
        print(f"Saved new LDA model to: {lda_model_path}")

    theta_tr = lda_modeler.get_topics(X_tr)
    theta_val = lda_modeler.get_topics(X_val)
    theta_te = lda_modeler.get_topics(X_te)

    lda_dim = theta_tr.shape[1]
    print(f"LDA topic dimension detected: {lda_dim}")

    all_theta = np.vstack([theta_tr, theta_val, theta_te])

    print("\n[4/4] Building BERT+LDA splits from make_dataloaders(..., topic_vecs=all_theta)")
    train_loader_combo, val_loader_combo, test_loader_combo = make_dataloaders(
        batch_size=args.batch_size,
        topic_vecs=all_theta,
    )

    x_train_combo, y_train_combo = loader_to_tensors(train_loader_combo)
    x_val_combo, y_val_combo = loader_to_tensors(val_loader_combo)
    x_test_combo, y_test_combo = loader_to_tensors(test_loader_combo)

    print(f"x_train_bert_lda: {tuple(x_train_combo.shape)}")
    print(f"x_val_bert_lda:   {tuple(x_val_combo.shape)}")
    print(f"x_test_bert_lda:  {tuple(x_test_combo.shape)}")

    expected_combo_dim = args.bert_dim + lda_dim
    if x_train_combo.shape[1] != expected_combo_dim:
        raise ValueError(
            f"Expected BERT+LDA dim {expected_combo_dim}, but got {x_train_combo.shape[1]}. "
            "That means topic vectors are not being concatenated correctly inside make_dataloaders."
        )

    for split_name, y_a, y_b in [
        ("train", y_train, y_train_combo),
        ("val", y_val, y_val_combo),
        ("test", y_test, y_test_combo),
    ]:
        if not torch.equal(y_a, y_b):
            raise ValueError(
                f"Label mismatch between BERT-only and BERT+LDA for split '{split_name}'. "
                "This means the splits are not aligned and the comparison would be invalid."
            )

    bundle = {
        "x_train_bert_only": x_train_bert,
        "x_val_bert_only": x_val_bert,
        "x_test_bert_only": x_test_bert,
        "x_train_bert_lda": x_train_combo,
        "x_val_bert_lda": x_val_combo,
        "x_test_bert_lda": x_test_combo,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "bert_dim": args.bert_dim,
        "lda_dim": lda_dim,
        "n_topics": lda_dim,
        "x_train": x_train_combo,
        "x_val": x_val_combo,
        "x_test": x_test_combo,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)

    print("\nBundle saved successfully.")
    print(f"Output path: {output_path}")
    print("\nSaved keys:")
    for key, value in bundle.items():
        if isinstance(value, torch.Tensor):
            print(f"- {key}: {tuple(value.shape)}")
        else:
            print(f"- {key}: {value}")

    print("\nDone. You can now run:")
    print("  python tune_bnn.py")
    print("  python tune_bnn.py --feature-mode bert_lda")
    print("  python compare_models_final.py")


if __name__ == "__main__":
    main()
