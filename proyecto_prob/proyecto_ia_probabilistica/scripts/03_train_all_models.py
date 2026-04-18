"""Train all model variants across seeds.

Models trained:
    - deterministic            (BERT only)
    - mc_dropout               (BERT only)
    - bnn_base                 (BERT only, single-head BNN; equivalent to K=1 MoE)
    - bnn_moe                  (BERT + theta gating, no hetero)
    - bnn_moe_hetero           (BERT + theta gating + heteroscedastic head)

For each (model, seed) we save:
    experiments/results/models/<model>/seed_<seed>/
        state.pt               torch state dict (det/mc_dropout)
        pyro_store.pt          pyro param store (bnn*)
        cfg.json               training config snapshot
        theta_test.npy etc.    (if applicable)
        probs_{val,test,ood}.npy       MC predictive probabilities per sample
        mc_probs_{val,test,ood}.npy    individual MC samples  [S, N, 2]
        preds_{val,test,ood}.npy
        uncertainty_{val,test,ood}.npz  predictive_entropy, aleatoric, mutual_info
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pyro
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data.loader import Standardizer, load_ood, load_raw, make_loader  # noqa: E402
from src.evaluation.uncertainty import decompose_mc  # noqa: E402
from src.inference.svi_trainer import SVITrainer, SVITrainerConfig  # noqa: E402
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig  # noqa: E402
from src.models.deterministic import DeterministicMLP  # noqa: E402
from src.models.mc_dropout import MCDropoutMLP  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

from pyro.infer.autoguide import AutoNormal  # noqa: E402


# -----------------------------
# Helpers
# -----------------------------

def _to_t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _load_theta(split: str) -> Optional[np.ndarray]:
    p = Path("experiments/results/slda") / f"theta_{split}.npy"
    if p.exists():
        return np.load(p)
    return None


def _split_data(
    raw: Dict, use_theta: bool
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
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
        th_t = _to_t(th[:len(idx)]) if th is not None else None
        out[name] = (x_s, y_s, th_t)
    return out


def _load_ood_data(use_theta: bool) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    try:
        ood = load_ood()
    except Exception:
        return None
    x = _to_t(ood["bert"])
    th = _load_theta("ood") if use_theta else None
    th_t = _to_t(th) if th is not None else None
    return (x, th_t)


# -----------------------------
# Deterministic / MC-Dropout training
# -----------------------------

def train_deterministic(
    splits, cfg: dict, device: torch.device, mc_dropout: bool
):
    scaler = Standardizer.fit(splits["train"][0])
    x_tr = scaler.transform(splits["train"][0]); y_tr = splits["train"][1]
    x_va = scaler.transform(splits["val"][0]);   y_va = splits["val"][1]

    dl_tr = make_loader(x_tr, y_tr, cfg["batch_size"], shuffle=True)
    dl_va = make_loader(x_va, y_va, cfg["batch_size"], shuffle=False)

    dp = cfg.get("dropout_p", 0.0)
    if mc_dropout:
        model = MCDropoutMLP(input_dim=x_tr.shape[1], hidden_dim=cfg["hidden_dim"], dropout_p=dp).to(device)
    else:
        model = DeterministicMLP(input_dim=x_tr.shape[1], hidden_dim=cfg["hidden_dim"], dropout_p=dp).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = nn.CrossEntropyLoss()

    best_state = None; best_nll = float("inf"); bad = 0
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()

        # val NLL
        model.eval()
        with torch.no_grad():
            vl = 0.0; n = 0
            for xb, yb in dl_va:
                xb = xb.to(device); yb = yb.to(device)
                p = torch.softmax(model(xb), dim=-1)
                vl += float(-torch.log(p[torch.arange(len(yb)), yb].clamp_min(1e-8)).sum())
                n += len(yb)
            v = vl / max(n, 1)
        if v < best_nll - 1e-4:
            best_nll = v; bad = 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return scaler, model


@torch.no_grad()
def predict_deterministic(
    model, scaler, x: torch.Tensor, device: torch.device, batch_size: int = 256,
    mc_dropout: bool = False, mc_samples: int = 50,
) -> torch.Tensor:
    """Return MC probs [S, N, 2]. For deterministic, S=1."""
    x_std = scaler.transform(x).to(device)
    out_chunks = []
    for i in range(0, len(x_std), batch_size):
        xb = x_std[i : i + batch_size]
        if mc_dropout:
            logits = model.mc_forward(xb, mc_samples=mc_samples)     # [S, B, 2]
            out_chunks.append(torch.softmax(logits, dim=-1).cpu())
        else:
            p = torch.softmax(model(xb), dim=-1).unsqueeze(0)        # [1, B, 2]
            out_chunks.append(p.cpu())
    return torch.cat(out_chunks, dim=1)


# -----------------------------
# BNN training (generic)
# -----------------------------

def train_bnn(
    splits, cfg: BayesianMoEConfig, svi_cfg: SVITrainerConfig,
    device: torch.device, use_theta: bool, batch_size: int,
):
    scaler = Standardizer.fit(splits["train"][0])
    x_tr = scaler.transform(splits["train"][0]); y_tr = splits["train"][1]; th_tr = splits["train"][2]
    x_va = scaler.transform(splits["val"][0]);   y_va = splits["val"][1];   th_va = splits["val"][2]

    if use_theta:
        dl_tr = make_loader(x_tr, y_tr, batch_size, shuffle=True, theta=th_tr)
        dl_va = make_loader(x_va, y_va, batch_size, shuffle=False, theta=th_va)
    else:
        dl_tr = make_loader(x_tr, y_tr, batch_size, shuffle=True)
        dl_va = make_loader(x_va, y_va, batch_size, shuffle=False)

    pyro.clear_param_store()
    model = BayesianMoE(cfg).to(device)
    guide = AutoNormal(model)

    def val_metric_fn(m, g, loader, dev):
        mc_probs = _bnn_mc_predict(m, g, loader, dev, mc_samples=svi_cfg.mc_samples_eval, use_theta=use_theta)
        probs = mc_probs.mean(0)
        # extract labels from loader
        ys = torch.cat([b[1] for b in loader], dim=0)
        idx = torch.arange(ys.shape[0])
        nll = float(-torch.log(probs[idx, ys].clamp_min(1e-8)).mean().item())
        acc = float((probs.argmax(1) == ys).float().mean().item())
        return {"nll": nll, "accuracy": acc}

    trainer = SVITrainer(model=model, guide=guide, cfg=svi_cfg, device=device)
    trainer.fit(dl_tr, val_loader=dl_va, val_metric_fn=val_metric_fn)
    return scaler, model, guide


def _bnn_mc_predict(
    model, guide, loader, device, mc_samples: int, use_theta: bool,
) -> torch.Tensor:
    """Return MC probs [S, N, 2] aligned with loader order."""
    chunks = []
    for batch in loader:
        if use_theta:
            xb, yb, tb = batch
            xb = xb.to(device); tb = tb.to(device)
            predictive = pyro.infer.Predictive(
                model, guide=guide, num_samples=mc_samples, return_sites=("_RETURN",)
            )
            logits = predictive(xb, tb)["_RETURN"]
        else:
            xb, yb = batch
            xb = xb.to(device)
            predictive = pyro.infer.Predictive(
                model, guide=guide, num_samples=mc_samples, return_sites=("_RETURN",)
            )
            logits = predictive(xb)["_RETURN"]
        chunks.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def predict_bnn(
    model, guide, scaler, x: torch.Tensor, device: torch.device, theta: Optional[torch.Tensor],
    mc_samples: int, batch_size: int = 256,
) -> torch.Tensor:
    x_std = scaler.transform(x)
    if theta is not None:
        dl = make_loader(x_std, torch.zeros(len(x_std), dtype=torch.long), batch_size, shuffle=False, theta=theta)
    else:
        dl = make_loader(x_std, torch.zeros(len(x_std), dtype=torch.long), batch_size, shuffle=False)
    return _bnn_mc_predict(model, guide, dl, device, mc_samples=mc_samples, use_theta=(theta is not None))


# -----------------------------
# Save outputs
# -----------------------------

def save_predictions(
    out_dir: Path,
    mc_probs: Dict[str, torch.Tensor],       # {split: [S, N, 2]}
    y_true: Dict[str, np.ndarray],            # {split: [N]}  (not for ood)
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, mc in mc_probs.items():
        mean_probs = mc.mean(0).numpy()
        preds = mean_probs.argmax(1)
        decomp = decompose_mc(mc)
        np.save(out_dir / f"probs_{split}.npy", mean_probs)
        np.save(out_dir / f"mc_probs_{split}.npy", mc.numpy())
        np.save(out_dir / f"preds_{split}.npy", preds)
        np.savez(
            out_dir / f"uncertainty_{split}.npz",
            predictive_entropy=decomp["predictive_entropy"].numpy(),
            aleatoric_entropy=decomp["aleatoric_entropy"].numpy(),
            mutual_info=decomp["mutual_info"].numpy(),
        )
        if split in y_true and y_true[split] is not None:
            np.save(out_dir / f"y_{split}.npy", y_true[split])


# -----------------------------
# Main orchestrator
# -----------------------------

MODELS_TO_TRAIN = [
    # name, fn_key
    ("deterministic",    "deterministic"),
    ("mc_dropout",       "mc_dropout"),
    ("bnn_base",         "bnn_no_theta"),       # K=1 effectively
    ("bnn_moe",          "bnn_with_theta"),
    ("bnn_moe_hetero",   "bnn_with_theta_hetero"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prior-std", type=float, default=0.1)
    ap.add_argument("--n-topics", type=int, default=10)
    ap.add_argument("--mc-samples", type=int, default=100)
    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="Subset of model names to train")
    ap.add_argument("--out-root", type=str, default="experiments/results/models")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw = load_raw()

    # Determine selection
    selected = args.only if args.only else [m[0] for m in MODELS_TO_TRAIN]

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n=========== SEED {seed} ===========")

        for model_name, fn_key in MODELS_TO_TRAIN:
            if model_name not in selected:
                continue
            print(f"\n--- Training {model_name} (seed {seed}) ---")

            use_theta = fn_key in ("bnn_with_theta", "bnn_with_theta_hetero")
            splits_dict = _split_data(raw, use_theta=use_theta)
            ood_data = _load_ood_data(use_theta=use_theta)

            seed_dir = Path(args.out_root) / model_name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            y_true_map = {
                "val":  raw["labels"][raw["splits"]["val"]],
                "test": raw["labels"][raw["splits"]["test"]],
            }

            if fn_key == "deterministic":
                cfg_d = dict(hidden_dim=args.hidden_dim, dropout_p=0.0, lr=args.lr,
                             weight_decay=1e-4, batch_size=args.batch_size,
                             epochs=args.epochs, patience=args.patience)
                scaler, model = train_deterministic(splits_dict, cfg_d, device, mc_dropout=False)
                torch.save({"state": model.state_dict(), "scaler_mean": scaler.mean, "scaler_std": scaler.std,
                            "input_dim": splits_dict["train"][0].shape[1]}, seed_dir / "state.pt")
                mc_probs = {
                    "val":  predict_deterministic(model, scaler, splits_dict["val"][0], device, mc_dropout=False),
                    "test": predict_deterministic(model, scaler, splits_dict["test"][0], device, mc_dropout=False),
                }
                if ood_data is not None:
                    mc_probs["ood"] = predict_deterministic(model, scaler, ood_data[0], device, mc_dropout=False)
                save_predictions(seed_dir, mc_probs, y_true_map)

            elif fn_key == "mc_dropout":
                cfg_d = dict(hidden_dim=args.hidden_dim, dropout_p=0.2, lr=args.lr,
                             weight_decay=1e-4, batch_size=args.batch_size,
                             epochs=args.epochs, patience=args.patience)
                scaler, model = train_deterministic(splits_dict, cfg_d, device, mc_dropout=True)
                torch.save({"state": model.state_dict(), "scaler_mean": scaler.mean, "scaler_std": scaler.std,
                            "input_dim": splits_dict["train"][0].shape[1], "dropout_p": 0.2}, seed_dir / "state.pt")
                mc_probs = {
                    "val":  predict_deterministic(model, scaler, splits_dict["val"][0], device, mc_dropout=True, mc_samples=args.mc_samples),
                    "test": predict_deterministic(model, scaler, splits_dict["test"][0], device, mc_dropout=True, mc_samples=args.mc_samples),
                }
                if ood_data is not None:
                    mc_probs["ood"] = predict_deterministic(model, scaler, ood_data[0], device, mc_dropout=True, mc_samples=args.mc_samples)
                save_predictions(seed_dir, mc_probs, y_true_map)

            else:  # Bayesian variants
                hetero = (fn_key == "bnn_with_theta_hetero")
                cfg_bnn = BayesianMoEConfig(
                    input_dim=splits_dict["train"][0].shape[1],
                    hidden_dim=args.hidden_dim,
                    n_experts=args.n_topics if use_theta else 1,
                    prior_std_w=args.prior_std,
                    prior_std_b=args.prior_std,
                    use_heteroscedastic=hetero,
                    fan_in_scaled_prior=True,
                )
                svi_cfg = SVITrainerConfig(
                    lr=args.lr, epochs=args.epochs, patience=args.patience,
                    mc_samples_eval=args.mc_samples,
                )
                scaler, model, guide = train_bnn(
                    splits_dict, cfg_bnn, svi_cfg, device,
                    use_theta=use_theta, batch_size=args.batch_size,
                )
                pyro.get_param_store().save(str(seed_dir / "pyro_store.pt"))
                torch.save({
                    "scaler_mean": scaler.mean, "scaler_std": scaler.std,
                    "cfg": vars(cfg_bnn), "use_theta": use_theta,
                }, seed_dir / "meta.pt")

                mc_probs = {
                    "val":  predict_bnn(model, guide, scaler, splits_dict["val"][0], device,
                                        theta=splits_dict["val"][2], mc_samples=args.mc_samples),
                    "test": predict_bnn(model, guide, scaler, splits_dict["test"][0], device,
                                        theta=splits_dict["test"][2], mc_samples=args.mc_samples),
                }
                if ood_data is not None:
                    mc_probs["ood"] = predict_bnn(model, guide, scaler, ood_data[0], device,
                                                  theta=ood_data[1], mc_samples=args.mc_samples)
                save_predictions(seed_dir, mc_probs, y_true_map)

            with open(seed_dir / "cfg.json", "w") as f:
                json.dump({
                    "model": model_name,
                    "seed": seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "hidden_dim": args.hidden_dim,
                    "lr": args.lr,
                    "prior_std": args.prior_std,
                    "n_topics": args.n_topics,
                    "mc_samples": args.mc_samples,
                }, f, indent=2)
            print(f"  Saved to: {seed_dir.resolve()}")


if __name__ == "__main__":
    main()
