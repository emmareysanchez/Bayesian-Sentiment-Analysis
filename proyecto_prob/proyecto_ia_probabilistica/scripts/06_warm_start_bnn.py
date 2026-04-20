"""Warm-start BNN from pretrained deterministic weights.

Pipeline:
  1. Train DeterministicMLP (same config as 03_train_all_models.py)
  2. Copy shared layer (768->128) directly to BNN shared layer
  3. Seed AutoNormal posterior means with deterministic 128->2 weights
  4. Run SVI from that warm starting point
  5. Compare test accuracy vs cold-start BNN

Saves to: experiments/results/models/bnn_moe_warm/seed_<seed>/
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
from pyro.infer.autoguide import AutoNormal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data.loader import Standardizer, make_loader, load_raw  # noqa: E402
from src.evaluation.uncertainty import decompose_mc               # noqa: E402
from src.inference.svi_trainer import SVITrainer, SVITrainerConfig  # noqa: E402
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig    # noqa: E402
from src.models.deterministic import DeterministicMLP             # noqa: E402
from src.utils.seed import set_seed                               # noqa: E402


# ---------------------------------------------------------------
# Helpers (copied from 03_train_all_models.py)
# ---------------------------------------------------------------

def _to_t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _load_theta(split: str):
    p = Path("experiments/results/slda") / f"theta_{split}.npy"
    return np.load(p) if p.exists() else None


def _split_data(raw, use_theta: bool):
    splits = raw["splits"]
    bert   = raw["bert"]
    y      = raw["labels"]
    out    = {}
    for name in ["train", "val", "test"]:
        idx  = splits[name]
        x_s  = _to_t(bert[idx])
        y_s  = _to_t(y[idx]).long()
        th   = _load_theta(name) if use_theta else None
        th_t = _to_t(th[: len(idx)]) if th is not None else None
        out[name] = (x_s, y_s, th_t)
    return out


# ---------------------------------------------------------------
# Step 1: train deterministic MLP
# ---------------------------------------------------------------

def train_deterministic(splits, cfg: dict, device: torch.device):
    scaler = Standardizer.fit(splits["train"][0])
    x_tr   = scaler.transform(splits["train"][0]); y_tr = splits["train"][1]
    x_va   = scaler.transform(splits["val"][0]);   y_va = splits["val"][1]

    dl_tr = make_loader(x_tr, y_tr, cfg["batch_size"], shuffle=True)
    dl_va = make_loader(x_va, y_va, cfg["batch_size"], shuffle=False)

    model = DeterministicMLP(
        input_dim=x_tr.shape[1], hidden_dim=cfg["hidden_dim"]
    ).to(device)
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    crit = nn.CrossEntropyLoss()

    best_state = None; best_nll = float("inf"); bad = 0
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0; n = 0; correct = 0
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                p = torch.softmax(model(xb), -1)
                vl      += float(-torch.log(p[torch.arange(len(yb)), yb].clamp_min(1e-8)).sum())
                correct += int((p.argmax(1) == yb).sum())
                n       += len(yb)
        v       = vl / max(n, 1)
        val_acc = correct / max(n, 1)
        marker  = " *" if v < best_nll - 1e-4 else ""
        print(f"  [det] epoch {ep+1:02d}  val_nll={v:.4f}  val_acc={val_acc:.4f}{marker}")
        if v < best_nll - 1e-4:
            best_nll = v; bad = 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("  [det] early stopping"); break
    if best_state:
        model.load_state_dict(best_state)
    return scaler, model


# ---------------------------------------------------------------
# Step 2: warm-start BNN guide from deterministic weights
# ---------------------------------------------------------------

def warm_start_guide(model_bnn: BayesianMoE, guide: AutoNormal,
                     det_model: DeterministicMLP, device: torch.device):
    """Copy deterministic weights into BNN.

    - shared layer (deterministic): direct weight copy
    - expert posterior means: set AutoNormal locs to det output-layer weights
    """
    # -- shared layer (det net[0] → bnn shared[0]) --
    with torch.no_grad():
        model_bnn.shared[0].weight.copy_(det_model.net[0].weight.to(device))
        model_bnn.shared[0].bias.copy_(det_model.net[0].bias.to(device))

    # -- expert posterior means --
    # net = [Linear(768,H), ReLU, Identity/Dropout, Linear(H,2)]
    out_layer = next(m for m in reversed(list(det_model.net)) if isinstance(m, nn.Linear))
    det_w = out_layer.weight.detach().to(device)   # [2, H]
    det_b = out_layer.bias.detach().to(device)     # [2]

    store = pyro.get_param_store()
    n_copied = 0
    for k in range(model_bnn.cfg.n_experts):
        for wname, src in [
            (f"AutoNormal.locs.experts.{k}.weight", det_w),
            (f"AutoNormal.locs.experts.{k}.bias",   det_b),
        ]:
            if wname in store:
                store[wname].data.copy_(src)
                n_copied += 1
    print(f"  [warm] copied {n_copied} param tensors into guide locs")


# ---------------------------------------------------------------
# Step 3: BNN training with val metric
# ---------------------------------------------------------------

def _bnn_mc_predict(model, guide, loader, device, mc_samples, use_theta):
    chunks = []
    for batch in loader:
        if use_theta:
            xb, yb, tb = batch; xb = xb.to(device); tb = tb.to(device)
            pred = pyro.infer.Predictive(model, guide=guide, num_samples=mc_samples,
                                         return_sites=("_RETURN",))
            logits = pred(xb, tb)["_RETURN"]
        else:
            xb, yb = batch; xb = xb.to(device)
            pred = pyro.infer.Predictive(model, guide=guide, num_samples=mc_samples,
                                         return_sites=("_RETURN",))
            logits = pred(xb)["_RETURN"]
        chunks.append(torch.softmax(logits, -1).cpu())
    return torch.cat(chunks, dim=1)


def train_bnn_warm(splits, cfg_bnn: BayesianMoEConfig, svi_cfg: SVITrainerConfig,
                   device, use_theta, batch_size, det_model, scaler):
    x_tr = scaler.transform(splits["train"][0]); y_tr = splits["train"][1]; th_tr = splits["train"][2]
    x_va = scaler.transform(splits["val"][0]);   y_va = splits["val"][1];   th_va = splits["val"][2]

    if use_theta:
        dl_tr = make_loader(x_tr, y_tr, batch_size, shuffle=True,  theta=th_tr)
        dl_va = make_loader(x_va, y_va, batch_size, shuffle=False, theta=th_va)
    else:
        dl_tr = make_loader(x_tr, y_tr, batch_size, shuffle=True)
        dl_va = make_loader(x_va, y_va, batch_size, shuffle=False)

    pyro.clear_param_store()
    model_bnn = BayesianMoE(cfg_bnn).to(device)
    guide     = AutoNormal(model_bnn)

    # Initialise guide (required before setting locs)
    first = next(iter(dl_tr))
    x0, y0 = first[0].to(device), first[1].to(device)
    th0 = first[2].to(device) if use_theta else None
    if th0 is not None:
        guide(x0, th0, y0)
    else:
        guide(x0, None, y0)

    # Warm-start
    warm_start_guide(model_bnn, guide, det_model, device)

    def val_metric_fn(m, g, loader, dev):
        mc   = _bnn_mc_predict(m, g, loader, dev, svi_cfg.mc_samples_eval, use_theta)
        prob = mc.mean(0)
        ys   = torch.cat([b[1] for b in loader], dim=0)
        nll  = float(-torch.log(prob[torch.arange(len(ys)), ys].clamp_min(1e-8)).mean())
        acc  = float((prob.argmax(1) == ys).float().mean())
        return {"nll": nll, "accuracy": acc}

    trainer = SVITrainer(model=model_bnn, guide=guide, cfg=svi_cfg, device=device)
    trainer.fit(dl_tr, val_loader=dl_va, val_metric_fn=val_metric_fn)
    return model_bnn, guide


# ---------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------

@torch.no_grad()
def predict_bnn(model, guide, scaler, x, device, theta, mc_samples, batch_size=256):
    x_std = scaler.transform(x)
    if theta is not None:
        dl = make_loader(x_std, torch.zeros(len(x_std), dtype=torch.long),
                         batch_size, shuffle=False, theta=theta)
    else:
        dl = make_loader(x_std, torch.zeros(len(x_std), dtype=torch.long),
                         batch_size, shuffle=False)
    return _bnn_mc_predict(model, guide, dl, device, mc_samples, theta is not None)


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
    use_theta = True   # bnn_moe variant
    cfg_det  = dict(hidden_dim=args.hidden_dim, lr=args.lr, wd=1e-4,
                    batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n{'='*50}  SEED {seed}  {'='*50}")

        splits   = _split_data(raw, use_theta=use_theta)
        cold_dir = Path(args.out_root) / "bnn_moe" / f"seed_{seed}"
        warm_dir = Path(args.out_root) / "bnn_moe_warm" / f"seed_{seed}"
        warm_dir.mkdir(parents=True, exist_ok=True)

        # 1. Deterministic
        print("\n[1/3] Training deterministic MLP...")
        scaler, det_model = train_deterministic(splits, cfg_det, device)

        # 2. BNN warm-start
        print("\n[2/3] Training BNN (warm start from deterministic)...")
        cfg_bnn = BayesianMoEConfig(
            input_dim=splits["train"][0].shape[1],
            hidden_dim=args.hidden_dim,
            n_experts=args.n_topics,
            prior_std_w=args.prior_std,
            prior_std_b=args.prior_std,
            fan_in_scaled_prior=True,
        )
        svi_cfg = SVITrainerConfig(
            lr=args.lr, epochs=args.epochs, patience=args.patience,
            mc_samples_eval=args.mc_samples,
        )
        model_warm, guide_warm = train_bnn_warm(
            splits, cfg_bnn, svi_cfg, device, use_theta,
            args.batch_size, det_model, scaler,
        )

        # 3. Save warm model
        pyro.get_param_store().save(str(warm_dir / "pyro_store.pt"))
        torch.save({
            "scaler_mean": scaler.mean, "scaler_std": scaler.std,
            "cfg": vars(cfg_bnn), "use_theta": use_theta,
        }, warm_dir / "meta.pt")

        # 4. Compare vs cold-start on test
        print("\n[3/3] Comparing warm vs cold on test set...")
        x_te = splits["test"][0]; y_te = splits["test"][1].numpy(); th_te = splits["test"][2]

        mc_warm = predict_bnn(model_warm, guide_warm, scaler, x_te, device, th_te, args.mc_samples)
        acc_warm = float((mc_warm.mean(0).argmax(1).numpy() == y_te).mean())

        # Load cold-start for comparison
        if (cold_dir / "pyro_store.pt").exists():
            meta_cold = torch.load(cold_dir / "meta.pt", map_location="cpu", weights_only=False)
            cfg_cold  = BayesianMoEConfig(**meta_cold["cfg"])
            pyro.clear_param_store()
            state_cold = torch.load(str(cold_dir / "pyro_store.pt"), map_location=device, weights_only=False)
            pyro.get_param_store().set_state(state_cold)
            model_cold = BayesianMoE(cfg_cold).to(device)
            guide_cold = AutoNormal(model_cold)
            dummy_x    = torch.zeros(2, cfg_cold.input_dim, device=device)
            dummy_y    = torch.zeros(2, dtype=torch.long, device=device)
            guide_cold(dummy_x, torch.ones(2, cfg_cold.n_experts, device=device) / cfg_cold.n_experts, dummy_y)
            scaler_cold = Standardizer(meta_cold["scaler_mean"], meta_cold["scaler_std"])
            mc_cold     = predict_bnn(model_cold, guide_cold, scaler_cold, x_te, device, th_te, args.mc_samples)
            acc_cold    = float((mc_cold.mean(0).argmax(1).numpy() == y_te).mean())
            print(f"\n  seed={seed}  cold-start acc={acc_cold:.4f}  warm-start acc={acc_warm:.4f}  Δ={acc_warm-acc_cold:+.4f}")
        else:
            print(f"\n  seed={seed}  warm-start acc={acc_warm:.4f}  (no cold-start to compare)")

        with open(warm_dir / "cfg.json", "w") as f:
            json.dump({"seed": seed, "warm_start": True, **vars(args)}, f, indent=2)
        print(f"  Saved to {warm_dir.resolve()}")


if __name__ == "__main__":
    main()
