from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from probabilistic_experiment_suite import (
    BNNConfig,
    DeterministicConfig,
    aggregate_rows,
    load_bundle,
    run_bnn_config,
    run_det_config,
    write_csv,
)

# -------------------------------------------------------------
# Default project configuration
# Edit these only if your project uses different paths.
# -------------------------------------------------------------
DEFAULT_DATA_PATH = "experiments/results/prepared_features.pt"
DEFAULT_BERT_DIM = 768
DEFAULT_BERT_BEST_CONFIG = "experiments/results/study_bert_only/best_bnn_config_bert_only.json"
DEFAULT_BERT_LDA_BEST_CONFIG = "experiments/results/study_bert_lda/best_bnn_config_bert_lda.json"
DEFAULT_OUTPUT_DIR = "experiments/results/final_comparison"
DEFAULT_DET_HIDDEN_DIM = 128
DEFAULT_DET_LR = 1e-3
DEFAULT_DET_WEIGHT_DECAY = 1e-4
DEFAULT_DET_EPOCHS = 25
DEFAULT_DET_PATIENCE = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEEDS = [42, 43, 44]


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)



def bnn_cfg_from_json(payload: Dict[str, Any], feature_mode_override: str | None = None) -> BNNConfig:
    return BNNConfig(
        feature_mode=feature_mode_override or payload["feature_mode"],
        hidden_dim=int(payload["hidden_dim"]),
        prior_std=float(payload["prior_std"]),
        lr=float(payload["lr"]),
        dropout_p=float(payload["dropout_p"]),
        batch_size=int(payload["batch_size"]),
        epochs=int(payload["epochs"]),
        patience=int(payload["patience"]),
        mc_samples_eval=int(payload.get("mc_samples_eval", 100)),
        fan_in_scaled_prior=bool(payload.get("fan_in_scaled_prior", True)),
        guide=str(payload.get("guide", "diag")),
    )



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final comparison: deterministic vs BNN(BERT) vs BNN(BERT+LDA).")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to bundle .pt")
    p.add_argument("--bert-dim", type=int, default=DEFAULT_BERT_DIM)
    p.add_argument("--bert-best-config", type=str, default=DEFAULT_BERT_BEST_CONFIG)
    p.add_argument("--bert-lda-best-config", type=str, default=DEFAULT_BERT_LDA_BEST_CONFIG)
    p.add_argument("--det-hidden-dim", type=int, default=DEFAULT_DET_HIDDEN_DIM)
    p.add_argument("--det-lr", type=float, default=DEFAULT_DET_LR)
    p.add_argument("--det-weight-decay", type=float, default=DEFAULT_DET_WEIGHT_DECAY)
    p.add_argument("--det-epochs", type=int, default=DEFAULT_DET_EPOCHS)
    p.add_argument("--det-patience", type=int, default=DEFAULT_DET_PATIENCE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--include-det-bert-lda", action="store_true")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()



def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = load_bundle(args.data)

    bert_best = load_json(args.bert_best_config)
    bert_bnn_cfg = bnn_cfg_from_json(bert_best, feature_mode_override="bert_only")

    lda_cfg_path = Path(args.bert_lda_best_config)
    if lda_cfg_path.exists():
        bert_lda_best = load_json(lda_cfg_path)
        bert_lda_bnn_cfg = bnn_cfg_from_json(bert_lda_best, feature_mode_override="bert_lda")
    else:
        print(f"BERT+LDA best config not found at {lda_cfg_path}. Reusing the BERT-only best config.")
        bert_lda_bnn_cfg = bnn_cfg_from_json(bert_best, feature_mode_override="bert_lda")

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        print(f"Running seed={seed}")

        det_bert_cfg = DeterministicConfig(
            feature_mode="bert_only",
            hidden_dim=args.det_hidden_dim,
            lr=args.det_lr,
            weight_decay=args.det_weight_decay,
            batch_size=args.batch_size,
            epochs=args.det_epochs,
            patience=args.det_patience,
        )
        rows.append(run_det_config(bundle, det_bert_cfg, bert_dim=args.bert_dim, seed=seed, device=device))

        bert_run_cfg = BNNConfig(**{**bert_bnn_cfg.__dict__, "batch_size": args.batch_size})
        rows.append(run_bnn_config(bundle, bert_run_cfg, bert_dim=args.bert_dim, seed=seed, device=device))

        try:
            bert_lda_run_cfg = BNNConfig(**{**bert_lda_bnn_cfg.__dict__, "batch_size": args.batch_size})
            rows.append(run_bnn_config(bundle, bert_lda_run_cfg, bert_dim=args.bert_dim, seed=seed, device=device))
        except Exception as exc:
            print(f"Skipping bert_lda BNN for seed={seed}: {exc}")

        if args.include_det_bert_lda:
            try:
                det_lda_cfg = DeterministicConfig(
                    feature_mode="bert_lda",
                    hidden_dim=args.det_hidden_dim,
                    lr=args.det_lr,
                    weight_decay=args.det_weight_decay,
                    batch_size=args.batch_size,
                    epochs=args.det_epochs,
                    patience=args.det_patience,
                )
                rows.append(run_det_config(bundle, det_lda_cfg, bert_dim=args.bert_dim, seed=seed, device=device))
            except Exception as exc:
                print(f"Skipping deterministic bert_lda for seed={seed}: {exc}")

    raw_path = out_dir / "final_comparison_raw.csv"
    write_csv(raw_path, rows)

    group_keys = [
        "model",
        "feature_mode",
        "hidden_dim",
        "prior_std",
        "lr",
        "dropout_p",
        "weight_decay",
        "batch_size",
        "epochs",
        "patience",
        "mc_samples_eval",
        "fan_in_scaled_prior",
        "guide",
    ]
    aggregated = aggregate_rows(rows, group_keys=group_keys)
    aggregated_sorted = sorted(
        aggregated,
        key=lambda r: (
            r.get("model"),
            r.get("feature_mode"),
            r.get("mean_val_nll", float("inf")),
        ),
    )

    summary_path = out_dir / "final_comparison_summary.csv"
    write_csv(summary_path, aggregated_sorted)

    print(f"\nSaved raw comparison to: {raw_path.resolve()}")
    print(f"Saved summary to: {summary_path.resolve()}")
    print("\nCompact summary:")
    for row in aggregated_sorted:
        print({
            "model": row.get("model"),
            "feature_mode": row.get("feature_mode"),
            "mean_val_accuracy": round(row.get("mean_val_accuracy", float("nan")), 4),
            "mean_val_nll": round(row.get("mean_val_nll", float("nan")), 4),
            "mean_val_ece": round(row.get("mean_val_ece", float("nan")), 4),
            "mean_test_accuracy": round(row.get("mean_test_accuracy", float("nan")), 4) if "mean_test_accuracy" in row else None,
            "mean_test_nll": round(row.get("mean_test_nll", float("nan")), 4) if "mean_test_nll" in row else None,
        })


if __name__ == "__main__":
    main()
