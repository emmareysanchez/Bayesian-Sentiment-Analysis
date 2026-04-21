"""Full offline evaluation (E1..E7 in the plan).

Reads per-(model, seed) predictions saved by 03_train_all_models.py and computes:
    - Predictive/Calibration metrics (E1, E3, E6): acc, nll, brier, ece, mce
    - Reliability diagrams
    - Selective prediction (E4): AURC, risk-coverage curves, compare scores
    - OOD detection (E5): AUROC, AUPR, FPR@95TPR
    - Results tables (CSV)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.evaluation.metrics import all_metrics, reliability_bins  # noqa: E402
from src.evaluation.ood import MahalanobisOOD, ood_report  # noqa: E402
from src.evaluation.selective import compare_scores, risk_coverage_curve  # noqa: E402
from src.utils.io import save_json, write_csv  # noqa: E402


def _list_seeds(model_dir: Path) -> List[Path]:
    return sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])


def _load_split(seed_dir: Path, split: str):
    probs_path = seed_dir / f"probs_{split}.npy"
    if not probs_path.exists():
        return None
    mean_probs = np.load(probs_path)
    y = np.load(seed_dir / f"y_{split}.npy") if (seed_dir / f"y_{split}.npy").exists() else None
    unc = np.load(seed_dir / f"uncertainty_{split}.npz") if (seed_dir / f"uncertainty_{split}.npz").exists() else None
    preds = np.load(seed_dir / f"preds_{split}.npy") if (seed_dir / f"preds_{split}.npy").exists() else mean_probs.argmax(1)
    return {
        "mean_probs": mean_probs,
        "y": y,
        "uncertainty": dict(unc) if unc is not None else {},
        "preds": preds,
    }


def _max_softmax_unc(mean_probs: np.ndarray) -> np.ndarray:
    """Higher = more uncertain. Use 1 - max(softmax)."""
    return 1.0 - mean_probs.max(axis=1)


def _fit_mahalanobis() -> MahalanobisOOD:
    """Fit Mahalanobis detector on train-split BERT embeddings."""
    import json
    bert = np.load("data/processed/bert_embeddings.npy")
    labels = np.load("data/processed/labels.npy")
    with open("data/processed/splits.json") as f:
        splits = json.load(f)
    train_idx = np.array(splits["train"])
    detector = MahalanobisOOD()
    detector.fit(bert[train_idx], labels[train_idx])
    return detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=str, default="experiments/results/models")
    ap.add_argument("--out-dir", type=str, default="experiments/results/evaluation")
    ap.add_argument("--splits", type=str, nargs="+", default=["val", "test"])
    ap.add_argument("--no-mahalanobis", action="store_true",
                    help="Skip Mahalanobis OOD score (faster but less complete)")
    args = ap.parse_args()

    models_root = Path(args.models_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Fit Mahalanobis once; scores are model-independent (feature space)
    mahal_detector = None
    if not args.no_mahalanobis:
        try:
            import json as _json
            bert_all = np.load("data/processed/bert_embeddings.npy")
            labels_all = np.load("data/processed/labels.npy")
            with open("data/processed/splits.json") as _f:
                _splits = _json.load(_f)
            ood_bert = np.load("data/ood/ood_embeddings.npy")
            mahal_detector = _fit_mahalanobis()
            mahal_score_test = mahal_detector.score(bert_all[np.array(_splits["test"])])
            mahal_score_ood  = mahal_detector.score(ood_bert)
            print(f"Mahalanobis fitted. test mean={mahal_score_test.mean():.1f}  ood mean={mahal_score_ood.mean():.1f}")
        except Exception as e:
            print(f"Warning: could not fit Mahalanobis detector: {e}")
            mahal_detector = None

    model_dirs = sorted([d for d in models_root.iterdir() if d.is_dir()])
    if not model_dirs:
        print(f"No models found in {models_root}")
        return

    # ---------- E1/E3/E6: accuracy, calibration ----------
    summary_rows = []
    reliability_data = {}

    for md in model_dirs:
        model_name = md.name
        for sd in _list_seeds(md):
            for split in args.splits:
                data = _load_split(sd, split)
                if data is None or data["y"] is None:
                    continue
                probs = torch.as_tensor(data["mean_probs"])
                y = torch.as_tensor(data["y"]).long()
                m = all_metrics(probs, y)
                row = {"model": model_name, "seed": int(sd.name.split("_")[1]), "split": split, **m}
                summary_rows.append(row)

                # Save reliability per (model, split) using pooled first seed only (for plot)
                if split == "test" and model_name not in reliability_data:
                    rb = reliability_bins(probs, y, n_bins=15)
                    reliability_data[model_name] = rb

    write_csv(out_dir / "metrics_raw.csv", summary_rows)

    # Aggregate mean ± std across seeds
    by_key: Dict = {}
    for r in summary_rows:
        key = (r["model"], r["split"])
        by_key.setdefault(key, []).append(r)
    agg_rows = []
    for (m_name, split), rows in by_key.items():
        agg = {"model": m_name, "split": split, "n_seeds": len(rows)}
        for metric in ["accuracy", "nll", "brier", "ece", "mce"]:
            vals = [r[metric] for r in rows]
            agg[f"mean_{metric}"] = float(np.mean(vals))
            agg[f"std_{metric}"] = float(np.std(vals))
        agg_rows.append(agg)
    agg_rows.sort(key=lambda r: (r["split"], r["mean_nll"]))
    write_csv(out_dir / "metrics_summary.csv", agg_rows)

    # ---------- Reliability diagram ----------
    if reliability_data:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
        for name, rb in reliability_data.items():
            ax.plot(rb["bin_confidence"], rb["bin_accuracy"], "o-", label=name, markersize=5)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram (test)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        fig.tight_layout()
        fig.savefig(out_dir / "reliability_diagram.pdf")
        fig.savefig(out_dir / "reliability_diagram.png", dpi=150)
        plt.close(fig)

    # ---------- E4: Selective prediction ----------
    selective_rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for md in model_dirs:
        model_name = md.name
        seeds = _list_seeds(md)
        if not seeds:
            continue
        sd = seeds[0]  # use first seed for plotting; aggregate AURC across all
        data = _load_split(sd, "test")
        if data is None or data["y"] is None:
            continue

        correct = (data["preds"] == data["y"]).astype(int)
        scores = {}
        if data["uncertainty"]:
            if "predictive_entropy" in data["uncertainty"]:
                scores["pred_entropy"] = data["uncertainty"]["predictive_entropy"]
            if "mutual_info" in data["uncertainty"]:
                scores["mutual_info"] = data["uncertainty"]["mutual_info"]
        scores["max_softmax"] = _max_softmax_unc(data["mean_probs"])

        # Pick best available score for plot (prefer mutual_info for Bayesian models)
        plot_score_name = "mutual_info" if "mutual_info" in scores and np.std(scores["mutual_info"]) > 1e-6 else "pred_entropy" if "pred_entropy" in scores else "max_softmax"
        rc = risk_coverage_curve(scores[plot_score_name], correct)
        ax.plot(rc["coverage"], rc["risk"], label=f"{model_name} ({plot_score_name})", linewidth=1.8)

        # Aggregate AURC across seeds for each score
        for score_name in scores:
            vals = []
            for s_d in seeds:
                d = _load_split(s_d, "test")
                if d is None or d["y"] is None:
                    continue
                c = (d["preds"] == d["y"]).astype(int)
                if score_name == "max_softmax":
                    sc = _max_softmax_unc(d["mean_probs"])
                elif score_name in d["uncertainty"]:
                    sc = d["uncertainty"][score_name]
                else:
                    continue
                res = compare_scores({score_name: sc}, c)[score_name]
                vals.append(res)
            if vals:
                row = {"model": model_name, "score": score_name, "n_seeds": len(vals)}
                for k in vals[0]:
                    arr = [v[k] for v in vals]
                    row[f"mean_{k}"] = float(np.mean(arr))
                    row[f"std_{k}"] = float(np.std(arr))
                selective_rows.append(row)

    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk (error rate)")
    ax.set_title("Risk–Coverage (test set)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.pdf"); fig.savefig(out_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)
    write_csv(out_dir / "selective_prediction.csv",
              sorted(selective_rows, key=lambda r: (r["model"], r["score"])))

    # ---------- E5: OOD detection ----------
    ood_rows = []
    for md in model_dirs:
        model_name = md.name
        seeds = _list_seeds(md)
        ood_scores_per_seed = []
        for sd in seeds:
            id_data = _load_split(sd, "test")
            ood_data = _load_split(sd, "ood")
            if id_data is None or ood_data is None:
                continue
            # Build scores: max_softmax always; entropies/MI if available
            def _scores(d):
                s = {"max_softmax": _max_softmax_unc(d["mean_probs"])}
                if d["uncertainty"]:
                    for k in ["predictive_entropy", "mutual_info"]:
                        if k in d["uncertainty"]:
                            s[k] = d["uncertainty"][k]
                return s
            id_s = _scores(id_data); ood_s = _scores(ood_data)
            # Add Mahalanobis (same scores for all model seeds — feature-space only)
            if mahal_detector is not None:
                id_s["mahalanobis"] = mahal_score_test
                ood_s["mahalanobis"] = mahal_score_ood
            rep = ood_report(id_s, ood_s)
            ood_scores_per_seed.append(rep)

        # Aggregate
        if not ood_scores_per_seed:
            continue
        score_names = set(ood_scores_per_seed[0].keys())
        for sn in score_names:
            aurocs = [r[sn]["auroc"] for r in ood_scores_per_seed if sn in r]
            auprs = [r[sn]["aupr_ood"] for r in ood_scores_per_seed if sn in r]
            fprs = [r[sn]["fpr_at_tpr_95"] for r in ood_scores_per_seed if sn in r]
            ood_rows.append({
                "model": model_name, "score": sn, "n_seeds": len(aurocs),
                "mean_auroc": float(np.mean(aurocs)), "std_auroc": float(np.std(aurocs)),
                "mean_aupr": float(np.mean(auprs)), "std_aupr": float(np.std(auprs)),
                "mean_fpr95": float(np.mean(fprs)), "std_fpr95": float(np.std(fprs)),
            })
    write_csv(out_dir / "ood_detection.csv",
              sorted(ood_rows, key=lambda r: (r["model"], r["score"])))

    # ---------- OOD plot: histogram of uncertainty (best model) ----------
    # Try bnn_moe_hetero first, else last model
    preferred = ["bnn_moe_hetero", "bnn_moe", "bnn_base", "mc_dropout", "deterministic"]
    plot_model = None
    for pm in preferred:
        if (models_root / pm).exists():
            plot_model = pm
            break
    if plot_model:
        seeds = _list_seeds(models_root / plot_model)
        if seeds:
            sd = seeds[0]
            id_d = _load_split(sd, "test"); ood_d = _load_split(sd, "ood")
            if id_d is not None and ood_d is not None:
                # use mutual_info if available else pred_entropy else max_softmax
                score_name = ("mutual_info" if "mutual_info" in id_d["uncertainty"]
                              else "predictive_entropy" if "predictive_entropy" in id_d["uncertainty"]
                              else "max_softmax")
                if score_name == "max_softmax":
                    id_s = _max_softmax_unc(id_d["mean_probs"])
                    ood_s = _max_softmax_unc(ood_d["mean_probs"])
                else:
                    id_s = id_d["uncertainty"][score_name]
                    ood_s = ood_d["uncertainty"][score_name]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(id_s, bins=40, alpha=0.6, label="In-distribution (test)", density=True)
                ax.hist(ood_s, bins=40, alpha=0.6, label="Out-of-distribution", density=True)
                ax.set_xlabel(f"{score_name} (higher = more uncertain)")
                ax.set_ylabel("Density")
                ax.set_title(f"Uncertainty separation — {plot_model}")
                ax.legend(); ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(out_dir / "ood_uncertainty_hist.pdf"); fig.savefig(out_dir / "ood_uncertainty_hist.png", dpi=150)
                plt.close(fig)

    # ---------- Final json summary ----------
    summary = {
        "metrics_summary_csv": str((out_dir / "metrics_summary.csv").resolve()),
        "selective_prediction_csv": str((out_dir / "selective_prediction.csv").resolve()),
        "ood_detection_csv": str((out_dir / "ood_detection.csv").resolve()),
        "plots": sorted(str(p.name) for p in out_dir.glob("*.pdf")),
        "models_evaluated": [d.name for d in model_dirs],
    }
    save_json(out_dir / "evaluation_summary.json", summary)
    print(f"\nEvaluation complete. Outputs in: {out_dir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
