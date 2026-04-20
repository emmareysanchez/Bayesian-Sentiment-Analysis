"""KL-weight sweep for BNN.

Trains bnn_moe with reduced KL regularization by scaling the likelihood
relative to the KL term in the ELBO:

    ELBO = (1/beta) * E_q[log p(y|x,w)] - KL[q(w)||p(w)]

Equivalently: KL weight = beta relative to likelihood.
  beta=1.0  → standard ELBO  (saved as bnn_moe_kl_full)
  beta=0.1  → KL / 10        (saved as bnn_moe_kl_div10)
  beta=0.01 → KL / 100       (saved as bnn_moe_kl_div100)

Usage:
    python scripts/07_kl_sweep.py --seeds 42 43
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyro
import torch
import torch.nn as nn
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data.loader import Standardizer, make_loader, load_raw  # noqa: E402
from src.evaluation.uncertainty import decompose_mc               # noqa: E402
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig    # noqa: E402
from src.utils.seed import set_seed                               # noqa: E402

KL_BETAS = {
    "bnn_moe_kl_full":   1.0,
    "bnn_moe_kl_div10":  0.1,
    "bnn_moe_kl_div100": 0.01,
}


# ---------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------

def _to_t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _load_theta(split: str):
    p = Path("experiments/results/slda") / f"theta_{split}.npy"
    return np.load(p) if p.exists() else None


def _split_data(raw):
    splits = raw["splits"]
    bert   = raw["bert"]
    y      = raw["labels"]
    out    = {}
    for name in ["train", "val", "test"]:
        idx  = splits[name]
        x_s  = _to_t(bert[idx])
        y_s  = _to_t(y[idx]).long()
        th   = _load_theta(name)
        th_t = _to_t(th[: len(idx)]) if th is not None else None
        out[name] = (x_s, y_s, th_t)
    return out


# ---------------------------------------------------------------
# Training loop with kl_beta scaling
# ---------------------------------------------------------------

def _bnn_mc_predict(model, guide, loader, device, mc_samples, use_theta):
    chunks = []
    for batch in loader:
        if use_theta:
            xb, yb, tb = batch; xb = xb.to(device); tb = tb.to(device)
            pred   = pyro.infer.Predictive(model, guide=guide, num_samples=mc_samples,
                                           return_sites=("_RETURN",))
            logits = pred(xb, tb)["_RETURN"]
        else:
            xb, yb = batch[:2]; xb = xb.to(device)
            pred   = pyro.infer.Predictive(model, guide=guide, num_samples=mc_samples,
                                           return_sites=("_RETURN",))
            logits = pred(xb)["_RETURN"]
        chunks.append(torch.softmax(logits, -1).cpu())
    return torch.cat(chunks, dim=1)


def train_bnn_kl(splits, cfg_bnn: BayesianMoEConfig, device, use_theta,
                 batch_size, lr, epochs, patience, mc_samples_eval,
                 kl_beta: float):
    """Train BNN with likelihood scaled by 1/kl_beta (reduces KL weight by kl_beta)."""
    scaler = Standardizer.fit(splits["train"][0])
    x_tr   = scaler.transform(splits["train"][0]); y_tr = splits["train"][1]; th_tr = splits["train"][2]
    x_va   = scaler.transform(splits["val"][0]);   y_va = splits["val"][1];   th_va = splits["val"][2]

    if use_theta:
        dl_tr = make_loader(x_tr, y_tr, batch_size, shuffle=True,  theta=th_tr)
        dl_va = make_loader(x_va, y_va, batch_size, shuffle=False, theta=th_va)
    else:
        dl_tr = make_loader(x_tr, y_tr, batch_size, shuffle=True)
        dl_va = make_loader(x_va, y_va, batch_size, shuffle=False)

    pyro.clear_param_store()
    model = BayesianMoE(cfg_bnn).to(device)
    guide = AutoNormal(model)

    # Wrap model: poutine.scale scales likelihood by 1/kl_beta
    # TraceMeanField_ELBO computes KL analytically → NOT scaled by poutine.scale
    # Net effect: KL is kl_beta times less important than the likelihood
    scaled_model = pyro.poutine.scale(model, scale=1.0 / kl_beta)

    # Initialise guide
    first   = next(iter(dl_tr))
    x0, y0  = first[0].to(device), first[1].to(device)
    th0     = first[2].to(device) if use_theta and len(first) > 2 else None
    n_data  = len(dl_tr.dataset)
    if th0 is not None:
        guide(x0, th0, y0, n_data=n_data)
    else:
        guide(x0, None, y0, n_data=n_data)

    svi = SVI(scaled_model, guide, Adam({"lr": lr}), loss=TraceMeanField_ELBO())

    best_val  = float("inf")
    best_store = None
    bad        = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0; N = 0
        for batch in dl_tr:
            if use_theta:
                xb, yb, tb = batch; xb = xb.to(device); yb = yb.to(device); tb = tb.to(device)
                loss = svi.step(xb, theta=tb, y=yb, n_data=n_data)
            else:
                xb, yb = batch[:2]; xb = xb.to(device); yb = yb.to(device)
                loss = svi.step(xb, y=yb, n_data=n_data)
            total += loss; N += xb.shape[0]

        # Val metric
        model.eval()
        mc   = _bnn_mc_predict(model, guide, dl_va, device, mc_samples_eval, use_theta)
        prob = mc.mean(0)
        ys   = torch.cat([b[1] for b in dl_va], dim=0)
        nll  = float(-torch.log(prob[torch.arange(len(ys)), ys].clamp_min(1e-8)).mean())
        acc  = float((prob.argmax(1) == ys).float().mean())
        marker = " *" if nll < best_val - 1e-4 else ""
        print(f"  [β={kl_beta}] epoch {epoch+1:02d}  elbo={total/max(N,1):.2f}"
              f"  val_nll={nll:.4f}  val_acc={acc:.4f}{marker}")

        if nll < best_val - 1e-4:
            best_val   = nll; bad = 0
            best_store = {k: v.detach().cpu().clone() for k, v in pyro.get_param_store().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  [β={kl_beta}] early stopping"); break

    if best_store:
        pyro.clear_param_store()
        for k, v in best_store.items():
            pyro.get_param_store()[k] = v.clone().to(device)

    return scaler, model, guide


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def eval_test(model, guide, scaler, splits, device, use_theta, mc_samples, batch_size=256):
    x_te = splits["test"][0]; y_te = splits["test"][1].numpy(); th_te = splits["test"][2]
    x_std = scaler.transform(x_te)
    if use_theta and th_te is not None:
        dl = make_loader(x_std, torch.zeros(len(x_std), dtype=torch.long),
                         batch_size, shuffle=False, theta=th_te)
    else:
        dl = make_loader(x_std, torch.zeros(len(x_std), dtype=torch.long),
                         batch_size, shuffle=False)
    mc    = _bnn_mc_predict(model, guide, dl, device, mc_samples, use_theta and th_te is not None)
    probs = mc.mean(0).numpy()
    preds = probs.argmax(1)
    acc   = float((preds == y_te).mean())

    pos_mask = y_te == 1
    recall_pos = float((preds[pos_mask] == 1).mean()) if pos_mask.sum() > 0 else float("nan")
    neg_mask   = y_te == 0
    recall_neg = float((preds[neg_mask] == 0).mean()) if neg_mask.sum() > 0 else float("nan")

    return {"accuracy": acc, "recall_pos": recall_pos, "recall_neg": recall_neg,
            "n_pos": int(pos_mask.sum()), "n_neg": int(neg_mask.sum())}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",      type=int, nargs="+", default=[42, 43])
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--patience",   type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--prior-std",  type=float, default=1.0)
    ap.add_argument("--n-topics",   type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mc-samples", type=int, default=100)
    ap.add_argument("--out-root",   type=str, default="experiments/results/models")
    ap.add_argument("--device",     type=str, default=None)
    ap.add_argument("--betas",      type=str, nargs="+",
                    default=["bnn_moe_kl_full", "bnn_moe_kl_div10", "bnn_moe_kl_div100"],
                    help="Subset of KL_BETAS keys to train")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    raw      = load_raw()
    use_theta = True

    summary = []

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n{'='*60}  SEED {seed}  {'='*60}")
        splits = _split_data(raw)

        cfg_bnn = BayesianMoEConfig(
            input_dim=splits["train"][0].shape[1],
            hidden_dim=args.hidden_dim,
            n_experts=args.n_topics,
            prior_std_w=args.prior_std,
            prior_std_b=args.prior_std,
            fan_in_scaled_prior=True,
        )

        for name, kl_beta in KL_BETAS.items():
            if name not in args.betas:
                continue
            print(f"\n--- {name}  (β={kl_beta}) ---")

            scaler, model, guide = train_bnn_kl(
                splits, cfg_bnn, device, use_theta,
                args.batch_size, args.lr, args.epochs, args.patience,
                args.mc_samples, kl_beta,
            )

            out_dir = Path(args.out_root) / name / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            pyro.get_param_store().save(str(out_dir / "pyro_store.pt"))
            torch.save({
                "scaler_mean": scaler.mean, "scaler_std": scaler.std,
                "cfg": vars(cfg_bnn), "use_theta": use_theta,
                "kl_beta": kl_beta,
            }, out_dir / "meta.pt")
            with open(out_dir / "cfg.json", "w") as f:
                json.dump({"seed": seed, "kl_beta": kl_beta, "model": name}, f, indent=2)

            metrics = eval_test(model, guide, scaler, splits, device, use_theta, args.mc_samples)
            summary.append({"model": name, "seed": seed, "kl_beta": kl_beta, **metrics})
            print(f"  acc={metrics['accuracy']:.4f}  recall_pos={metrics['recall_pos']:.4f}"
                  f"  recall_neg={metrics['recall_neg']:.4f}")
            print(f"  Saved → {out_dir.resolve()}")

    print(f"\n{'='*60}  SUMMARY  {'='*60}")
    print(f"{'Model':<25} {'seed':>4} {'beta':>6} {'acc':>7} {'rec+':>7} {'rec-':>7}")
    for r in summary:
        print(f"{r['model']:<25} {r['seed']:>4} {r['kl_beta']:>6} "
              f"{r['accuracy']:>7.4f} {r['recall_pos']:>7.4f} {r['recall_neg']:>7.4f}")


if __name__ == "__main__":
    main()
