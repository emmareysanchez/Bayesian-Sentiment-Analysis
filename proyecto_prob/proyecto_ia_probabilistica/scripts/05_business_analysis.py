"""Cost-benefit analysis of the rejection threshold.

For each model, computes the expected total cost as a function of the
uncertainty threshold τ and locates τ* that minimizes cost, using the true
labels on the test set.

Usage:
    python scripts/05_business_analysis.py --cost-fp 1.0 --cost-fn 5.0 --cost-review 0.2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.evaluation.business import expected_cost_curve, optimal_threshold  # noqa: E402
from src.utils.io import save_json, write_csv  # noqa: E402


def _max_softmax_unc(mean_probs: np.ndarray) -> np.ndarray:
    return 1.0 - mean_probs.max(axis=1)


def _pick_score(seed_dir: Path):
    """Return (score_name, score_array) using best available Bayesian score."""
    unc_path = seed_dir / "uncertainty_test.npz"
    if unc_path.exists():
        unc = np.load(unc_path)
        if "mutual_info" in unc and np.std(unc["mutual_info"]) > 1e-6:
            return "mutual_info", unc["mutual_info"]
        if "predictive_entropy" in unc:
            return "predictive_entropy", unc["predictive_entropy"]
    probs = np.load(seed_dir / "probs_test.npy")
    return "max_softmax", _max_softmax_unc(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=str, default="experiments/results/models")
    ap.add_argument("--out-dir", type=str, default="experiments/results/business")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=5.0)
    ap.add_argument("--cost-review", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    models_root = Path(args.models_root)
    model_dirs = sorted([d for d in models_root.iterdir() if d.is_dir()])
    if not model_dirs:
        print(f"No models found in {models_root}")
        return

    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))

    for md in model_dirs:
        model_name = md.name
        seeds = sorted([d for d in md.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        if not seeds:
            continue

        # Aggregate across seeds
        taus, costs, rejs = [], [], []
        for sd in seeds:
            probs = np.load(sd / "probs_test.npy")
            preds = np.load(sd / "preds_test.npy")
            y = np.load(sd / "y_test.npy")
            score_name, u = _pick_score(sd)
            curve = expected_cost_curve(
                u, preds, y, args.cost_fp, args.cost_fn, args.cost_review,
                n_thresholds=200,
            )
            opt = optimal_threshold(
                u, preds, y, args.cost_fp, args.cost_fn, args.cost_review,
            )
            taus.append(opt["tau_star"])
            costs.append(opt["total_cost_at_star"])
            rejs.append(opt["rejection_rate_at_star"])
            rows.append({
                "model": model_name,
                "seed": int(sd.name.split("_")[1]),
                "score": score_name,
                "cost_fp": args.cost_fp, "cost_fn": args.cost_fn, "cost_review": args.cost_review,
                "tau_star": opt["tau_star"],
                "total_cost_at_star": opt["total_cost_at_star"],
                "avg_cost_at_star": opt["avg_cost_at_star"],
                "rejection_rate_at_star": opt["rejection_rate_at_star"],
                "baseline_always_classify": opt["baseline_always_classify"],
                "baseline_always_review": opt["baseline_always_review"],
                "savings_vs_always_classify": opt["savings_vs_always_classify"],
                "savings_pct": opt["savings_pct_vs_always_classify"],
            })

        # Plot curve using first seed
        sd = seeds[0]
        probs = np.load(sd / "probs_test.npy")
        preds = np.load(sd / "preds_test.npy")
        y = np.load(sd / "y_test.npy")
        score_name, u = _pick_score(sd)
        curve = expected_cost_curve(
            u, preds, y, args.cost_fp, args.cost_fn, args.cost_review, n_thresholds=200,
        )
        ax.plot(curve["rejection_rate"], curve["avg_cost"],
                label=f"{model_name} ({score_name})", linewidth=1.8)

    ax.axhline(args.cost_review, color="k", linestyle=":", alpha=0.5, label=f"always review = {args.cost_review}")
    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Average cost per sample")
    ax.set_title(f"Expected cost vs rejection rate  "
                 f"(FP={args.cost_fp}, FN={args.cost_fn}, review={args.cost_review})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "expected_cost.pdf"); fig.savefig(out_dir / "expected_cost.png", dpi=150)
    plt.close(fig)

    write_csv(out_dir / "business_analysis.csv", rows)

    # Aggregate summary (mean across seeds)
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    agg = []
    for name, rs in by_model.items():
        agg.append({
            "model": name,
            "score": rs[0]["score"],
            "n_seeds": len(rs),
            "mean_tau_star": float(np.mean([r["tau_star"] for r in rs])),
            "mean_avg_cost_at_star": float(np.mean([r["avg_cost_at_star"] for r in rs])),
            "mean_rejection_rate": float(np.mean([r["rejection_rate_at_star"] for r in rs])),
            "mean_savings_pct": float(np.mean([r["savings_pct"] for r in rs])),
        })
    agg.sort(key=lambda r: r["mean_avg_cost_at_star"])
    write_csv(out_dir / "business_summary.csv", agg)
    save_json(out_dir / "cost_config.json", {
        "cost_fp": args.cost_fp, "cost_fn": args.cost_fn, "cost_review": args.cost_review,
    })

    print("\nBusiness analysis complete.")
    print(f"  CSV: {out_dir / 'business_summary.csv'}")
    print(f"  PDF: {out_dir / 'expected_cost.pdf'}")
    print("\nRanking by average cost at τ*:")
    for r in agg:
        print(f"  {r['model']:25s} τ*={r['mean_tau_star']:.3f}  avg_cost={r['mean_avg_cost_at_star']:.4f}  "
              f"reject={r['mean_rejection_rate']*100:.1f}%  savings={r['mean_savings_pct']:.1f}%")


if __name__ == "__main__":
    main()
