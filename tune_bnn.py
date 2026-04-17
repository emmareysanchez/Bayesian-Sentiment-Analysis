from __future__ import annotations

import argparse
from pathlib import Path

import torch

from probabilistic_experiment_suite import (
    BNNConfig,
    aggregate_rows,
    choose_best_bnn_config,
    load_bundle,
    run_bnn_config,
    save_json,
    write_csv,
)

# -------------------------------------------------------------
# Default project configuration
# Edit these only if your project uses different paths.
# -------------------------------------------------------------
DEFAULT_DATA_PATH = "experiments/results/prepared_features.pt"
DEFAULT_BERT_DIM = 768
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 5
DEFAULT_MC_SAMPLES_EVAL = 100
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_HIDDEN_DIMS = [64, 128, 256]
DEFAULT_PRIOR_STDS = [0.025, 0.05, 0.1, 0.2]
DEFAULT_LRS = [1e-3, 5e-4, 2e-4]
DEFAULT_DROPOUTS = [0.0, 0.1, 0.2]
DEFAULT_OUTPUT_ROOT = "experiments/results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-search for the Bayesian MLP.")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to bundle .pt")
    p.add_argument("--feature-mode", type=str, default="bert_only", choices=["bert_only", "bert_lda", "lda_only"])
    p.add_argument("--bert-dim", type=int, default=DEFAULT_BERT_DIM)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--mc-samples-eval", type=int, default=DEFAULT_MC_SAMPLES_EVAL)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=DEFAULT_HIDDEN_DIMS)
    p.add_argument("--prior-stds", type=float, nargs="+", default=DEFAULT_PRIOR_STDS)
    p.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    p.add_argument("--dropouts", type=float, nargs="+", default=DEFAULT_DROPOUTS)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"{DEFAULT_OUTPUT_ROOT}/study_{args.feature_mode}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = load_bundle(args.data)

    rows = []
    total = len(args.seeds) * len(args.hidden_dims) * len(args.prior_stds) * len(args.lrs) * len(args.dropouts)
    idx = 0
    for seed in args.seeds:
        for hidden_dim in args.hidden_dims:
            for prior_std in args.prior_stds:
                for lr in args.lrs:
                    for dropout_p in args.dropouts:
                        idx += 1
                        cfg = BNNConfig(
                            feature_mode=args.feature_mode,
                            hidden_dim=hidden_dim,
                            prior_std=prior_std,
                            lr=lr,
                            dropout_p=dropout_p,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            patience=args.patience,
                            mc_samples_eval=args.mc_samples_eval,
                        )
                        print(
                            f"[{idx}/{total}] seed={seed} feature={args.feature_mode} hidden={hidden_dim} "
                            f"prior={prior_std} lr={lr} dropout={dropout_p}"
                        )
                        row = run_bnn_config(bundle, cfg, bert_dim=args.bert_dim, seed=seed, device=device)
                        rows.append(row)

    raw_path = out_dir / f"bnn_tuning_raw_{args.feature_mode}.csv"
    write_csv(raw_path, rows)

    group_keys = [
        "model",
        "feature_mode",
        "hidden_dim",
        "prior_std",
        "lr",
        "dropout_p",
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
            r.get("mean_val_nll", float("inf")),
            r.get("mean_val_ece", float("inf")),
            -r.get("mean_val_accuracy", float("-inf")),
        ),
    )

    agg_path = out_dir / f"bnn_tuning_summary_{args.feature_mode}.csv"
    write_csv(agg_path, aggregated_sorted)

    best = choose_best_bnn_config(aggregated_sorted)
    best_export = {
        "feature_mode": best["feature_mode"],
        "hidden_dim": best["hidden_dim"],
        "prior_std": best["prior_std"],
        "lr": best["lr"],
        "dropout_p": best["dropout_p"],
        "batch_size": best["batch_size"],
        "epochs": best["epochs"],
        "patience": best["patience"],
        "mc_samples_eval": best["mc_samples_eval"],
        "fan_in_scaled_prior": best.get("fan_in_scaled_prior", True),
        "guide": best.get("guide", "diag"),
        "selection_rule": "min mean_val_nll, then min mean_val_ece, then max mean_val_accuracy",
        "score_snapshot": {
            "mean_val_accuracy": best.get("mean_val_accuracy"),
            "mean_val_nll": best.get("mean_val_nll"),
            "mean_val_ece": best.get("mean_val_ece"),
            "mean_val_brier": best.get("mean_val_brier"),
        },
    }
    best_path = out_dir / f"best_bnn_config_{args.feature_mode}.json"
    save_json(best_path, best_export)

    print(f"\nSaved raw results to: {raw_path.resolve()}")
    print(f"Saved aggregated summary to: {agg_path.resolve()}")
    print(f"Saved best config to: {best_path.resolve()}")
    print("\nTop 5 configs:")
    for row in aggregated_sorted[:5]:
        print({
            "feature_mode": row.get("feature_mode"),
            "hidden_dim": row.get("hidden_dim"),
            "prior_std": row.get("prior_std"),
            "lr": row.get("lr"),
            "dropout_p": row.get("dropout_p"),
            "mean_val_accuracy": round(row.get("mean_val_accuracy", float("nan")), 4),
            "mean_val_nll": round(row.get("mean_val_nll", float("nan")), 4),
            "mean_val_ece": round(row.get("mean_val_ece", float("nan")), 4),
        })


if __name__ == "__main__":
    main()
