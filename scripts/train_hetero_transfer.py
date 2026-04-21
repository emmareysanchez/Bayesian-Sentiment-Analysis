#!/usr/bin/env python3
"""
Quick script to train bnn_moe_hetero with transfer learning from bnn_moe.

Usage:
    python scripts/train_hetero_transfer.py --transfer-seed 42 --epochs 30 --lr 1e-4

This loads the bnn_moe weights from seed 42 and continues training with the
heteroscedastic head.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyro
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from pyro.infer.autoguide import AutoNormal
from src.data.loader import Standardizer, load_ood, load_raw, make_loader
from src.evaluation.uncertainty import decompose_mc
from src.inference.svi_trainer import SVITrainer, SVITrainerConfig
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig
from src.utils.seed import set_seed


def _to_t(x):
    return torch.as_tensor(x, dtype=torch.float32)


def _load_theta(split: str):
    p = Path("experiments/results/slda") / f"theta_{split}.npy"
    if p.exists():
        return np.load(p)
    return None


def _split_data(raw, use_theta: bool):
    """Return {'train','val','test': (x, y, theta or None)}."""
    splits = raw["splits"]
    bert = raw["bert"]
    y = raw["labels"]
    out = {}
    for name in ["train", "val", "test"]:
        idx = splits[name]
        x_s = _to_t(bert[idx])
        y_s = _to_t(y[idx]).long()
        th = _load_theta(name) if use_theta else None
        th_t = _to_t(th[: len(idx)]) if th is not None else None
        out[name] = (x_s, y_s, th_t)
    return out


def _load_ood_data(use_theta: bool):
    try:
        from src.data.loader import load_ood

        ood = load_ood()
    except Exception:
        return None
    x = _to_t(ood["bert"])
    th = _load_theta("ood") if use_theta else None
    th_t = _to_t(th) if th is not None else None
    return (x, th_t)


def transfer_bnn_weights(src_param_path: Path, dst_model: BayesianMoE, dst_guide):
    """Transfer weights from bnn_moe to bnn_moe_hetero."""
    src_state = torch.load(src_param_path, weights_only=False)

    for name, value in src_state["params"].items():
        if "hetero" not in name.lower():
            if name not in pyro.get_param_store():
                pyro.get_param_store()[name] = value.detach().clone()
            else:
                pyro.get_param_store()[name].data.copy_(value.detach())

    print(
        f"✓ Transferred {sum(1 for n in src_state['params'].keys() if 'hetero' not in n.lower())} parameters"
    )


def _bnn_mc_predict(model, guide, loader, device, mc_samples: int, use_theta: bool):
    """Return MC probs [S, N, 2]."""
    chunks = []
    for batch in loader:
        if use_theta:
            xb, yb, tb = batch
            xb = xb.to(device)
            tb = tb.to(device)
        else:
            xb, yb = batch
            xb = xb.to(device)

        predictive = pyro.infer.Predictive(
            model, guide=guide, num_samples=mc_samples, return_sites=("_RETURN",)
        )
        if use_theta:
            logits = predictive(xb, tb)["_RETURN"]
        else:
            logits = predictive(xb)["_RETURN"]
        chunks.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def predict_bnn(model, guide, scaler, x, device, theta, mc_samples, batch_size=256):
    """Predict with MC sampling."""
    x_std = scaler.transform(x)
    if theta is not None:
        dl = make_loader(
            x_std,
            torch.zeros(len(x_std), dtype=torch.long),
            batch_size,
            shuffle=False,
            theta=theta,
        )
    else:
        dl = make_loader(
            x_std, torch.zeros(len(x_std), dtype=torch.long), batch_size, shuffle=False
        )
    return _bnn_mc_predict(
        model, guide, dl, device, mc_samples=mc_samples, use_theta=(theta is not None)
    )


def main():
    ap = argparse.ArgumentParser(
        description="Train bnn_moe_hetero with transfer learning from bnn_moe"
    )
    ap.add_argument(
        "--transfer-seed", type=int, default=42, help="Seed of bnn_moe to transfer from"
    )
    ap.add_argument(
        "--train-seed", type=int, default=45, help="Random seed for this training run"
    )
    ap.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (typically lower for transfer learning)",
    )
    ap.add_argument("--n-topics", type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--prior-std", type=float, default=1.0)
    ap.add_argument("--mc-samples", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-root", type=str, default="experiments/results/models")
    args = ap.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    set_seed(args.train_seed)

    # Load data
    print("Loading data...")
    raw = load_raw()
    splits_dict = _split_data(raw, use_theta=True)
    ood_data = _load_ood_data(use_theta=True)

    # Setup directories
    src_dir = Path(args.out_root) / "bnn_moe" / f"seed_{args.transfer_seed}"
    transfer_path = src_dir / "pyro_store.pt"

    out_dir = (
        Path(args.out_root) / "bnn_moe_hetero_transfer" / f"seed_{args.train_seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not transfer_path.exists():
        print(f"ERROR: Source model not found at {transfer_path}")
        print(f"Make sure bnn_moe was trained with seed {args.transfer_seed}")
        return 1

    print(f"\nTransfer setup:")
    print(f"  Source: {transfer_path}")
    print(f"  Output: {out_dir}")

    # Prepare data loaders
    scaler = Standardizer.fit(splits_dict["train"][0])
    x_tr = scaler.transform(splits_dict["train"][0])
    y_tr = splits_dict["train"][1]
    th_tr = splits_dict["train"][2]
    x_va = scaler.transform(splits_dict["val"][0])
    y_va = splits_dict["val"][1]
    th_va = splits_dict["val"][2]

    dl_tr = make_loader(x_tr, y_tr, args.batch_size, shuffle=True, theta=th_tr)
    dl_va = make_loader(x_va, y_va, args.batch_size, shuffle=False, theta=th_va)

    # Create model
    print("\nInitializing bnn_moe_hetero...")
    pyro.clear_param_store()
    cfg_bnn = BayesianMoEConfig(
        input_dim=splits_dict["train"][0].shape[1],
        hidden_dim=args.hidden_dim,
        n_experts=args.n_topics,
        prior_std_w=args.prior_std,
        prior_std_b=args.prior_std,
        use_heteroscedastic=True,
        fan_in_scaled_prior=True,
    )
    model = BayesianMoE(cfg_bnn).to(device)
    guide = AutoNormal(model)

    # Transfer weights
    print("Transferring weights from bnn_moe...")
    transfer_bnn_weights(transfer_path, model, guide)

    # Setup training
    svi_cfg = SVITrainerConfig(
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        mc_samples_eval=args.mc_samples,
    )

    def val_metric_fn(m, g, loader, dev):
        mc_probs = _bnn_mc_predict(
            m, g, loader, dev, mc_samples=svi_cfg.mc_samples_eval, use_theta=True
        )
        probs = mc_probs.mean(0)
        ys = torch.cat([b[1] for b in loader], dim=0)
        idx = torch.arange(ys.shape[0])
        nll = float(-torch.log(probs[idx, ys].clamp_min(1e-8)).mean().item())
        acc = float((probs.argmax(1) == ys).float().mean().item())
        return {"nll": nll, "accuracy": acc}

    # Train
    print("\nTraining bnn_moe_hetero with hetero head...")
    trainer = SVITrainer(model=model, guide=guide, cfg=svi_cfg, device=device)
    trainer.fit(dl_tr, val_loader=dl_va, val_metric_fn=val_metric_fn)

    # Save
    print(f"\nSaving results to {out_dir}...")
    pyro.get_param_store().save(str(out_dir / "pyro_store.pt"))
    torch.save(
        {
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "cfg": vars(cfg_bnn),
            "use_theta": True,
            "transfer_from_seed": args.transfer_seed,
        },
        out_dir / "meta.pt",
    )

    # Predict
    print("Computing predictions...")
    mc_probs = {
        "val": predict_bnn(
            model,
            guide,
            scaler,
            splits_dict["val"][0],
            device,
            theta=splits_dict["val"][2],
            mc_samples=args.mc_samples,
        ),
        "test": predict_bnn(
            model,
            guide,
            scaler,
            splits_dict["test"][0],
            device,
            theta=splits_dict["test"][2],
            mc_samples=args.mc_samples,
        ),
    }
    if ood_data is not None:
        mc_probs["ood"] = predict_bnn(
            model,
            guide,
            scaler,
            ood_data[0],
            device,
            theta=ood_data[1],
            mc_samples=args.mc_samples,
        )

    # Save predictions
    for split, mc in mc_probs.items():
        mean_probs = mc.mean(0).numpy()
        preds = mean_probs.argmax(1)
        np.save(out_dir / f"probs_{split}.npy", mean_probs)
        np.save(out_dir / f"mc_probs_{split}.npy", mc.numpy())
        np.save(out_dir / f"preds_{split}.npy", preds)

    y_true_map = {
        "val": raw["labels"][raw["splits"]["val"]],
        "test": raw["labels"][raw["splits"]["test"]],
    }
    for split in ["val", "test"]:
        np.save(out_dir / f"y_{split}.npy", y_true_map[split])

    with open(out_dir / "cfg.json", "w") as f:
        json.dump(
            {
                "model": "bnn_moe_hetero_transfer",
                "train_seed": args.train_seed,
                "transfer_seed": args.transfer_seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "lr": args.lr,
                "n_topics": args.n_topics,
                "mc_samples": args.mc_samples,
            },
            f,
            indent=2,
        )

    print(f"✓ Complete! Results saved to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
