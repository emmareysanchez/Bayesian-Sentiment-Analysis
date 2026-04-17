from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import pyro
    import pyro.distributions as dist
    import pyro.infer
    from pyro.infer.autoguide import AutoDiagonalNormal
    from pyro.nn import PyroModule, PyroSample
    from pyro.optim import Adam as PyroAdam
    _PYRO_AVAILABLE = True
    _PYRO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency guard
    pyro = None
    dist = None
    AutoDiagonalNormal = None
    PyroModule = nn.Module
    class _MissingPyroSample:
        def __init__(self, *args, **kwargs):
            raise ImportError("pyro is required for Bayesian experiments. Install it with: pip install pyro-ppl")
    PyroSample = _MissingPyroSample
    PyroAdam = None
    _PYRO_AVAILABLE = False
    _PYRO_IMPORT_ERROR = exc



def ensure_pyro_available() -> None:
    if not _PYRO_AVAILABLE:
        raise ImportError(
            "pyro is required for Bayesian experiments. Install it with: pip install pyro-ppl\n"
            f"Original import error: {_PYRO_IMPORT_ERROR}"
        )

# -----------------------------
# Generic utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _PYRO_AVAILABLE:
        pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


@dataclass
class BNNConfig:
    feature_mode: str
    hidden_dim: int = 128
    prior_std: float = 0.1
    lr: float = 1e-3
    dropout_p: float = 0.0
    batch_size: int = 64
    epochs: int = 25
    patience: int = 5
    mc_samples_eval: int = 100
    fan_in_scaled_prior: bool = True
    guide: str = "diag"


@dataclass
class DeterministicConfig:
    feature_mode: str
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 25
    patience: int = 5


@dataclass
class TrainingArtifacts:
    standardizer: Standardizer
    model: Any
    guide: Any = None


# -----------------------------
# Data bundle handling
# -----------------------------


def load_bundle(path: str | Path) -> Dict[str, Any]:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError("Bundle must be a dict saved with torch.save(...).")
    return bundle



def _to_tensor(x: Any, floatify: bool = True) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.tensor(x)
    if floatify and not t.dtype.is_floating_point:
        return t.float()
    return t



def _extract_split(bundle: Dict[str, Any], split: str, feature_mode: str, bert_dim: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Priority 1: explicit keys per feature mode
    explicit_x_key = f"x_{split}_{feature_mode}"
    y_key = f"y_{split}"
    if explicit_x_key in bundle:
        return _to_tensor(bundle[explicit_x_key], floatify=True), _to_tensor(bundle[y_key], floatify=False).long()

    # Priority 2: generic x_split with slicing based on feature mode
    generic_x_key = f"x_{split}"
    if generic_x_key not in bundle or y_key not in bundle:
        raise KeyError(f"Missing keys for split '{split}'. Expected either {explicit_x_key} + {y_key} or {generic_x_key} + {y_key}.")

    x = _to_tensor(bundle[generic_x_key], floatify=True)
    y = _to_tensor(bundle[y_key], floatify=False).long()

    if feature_mode == "bert_lda":
        return x, y

    if bert_dim is None:
        bert_dim = int(bundle.get("bert_dim", 0))
    if bert_dim <= 0 or bert_dim > x.shape[1]:
        raise ValueError(
            f"Invalid bert_dim={bert_dim} for input_dim={x.shape[1]}. Provide bert_dim in the bundle or as argument."
        )

    if feature_mode == "bert_only":
        return x[:, :bert_dim], y
    if feature_mode == "lda_only":
        if bert_dim == x.shape[1]:
            raise ValueError("Requested lda_only but no LDA dimensions were found.")
        return x[:, bert_dim:], y

    raise ValueError(f"Unknown feature_mode='{feature_mode}'")



def load_feature_splits(
    bundle: Dict[str, Any],
    feature_mode: str,
    bert_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for split in ["train", "val"]:
        x, y = _extract_split(bundle, split, feature_mode, bert_dim)
        out[f"x_{split}"] = x
        out[f"y_{split}"] = y

    if (f"x_test_{feature_mode}" in bundle) or ("x_test" in bundle and "y_test" in bundle):
        x_test, y_test = _extract_split(bundle, "test", feature_mode, bert_dim)
        out["x_test"] = x_test
        out["y_test"] = y_test
    return out



def fit_standardizer(x_train: torch.Tensor) -> Standardizer:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return Standardizer(mean=mean, std=std)



def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(x, y.long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# -----------------------------
# Metrics
# -----------------------------


def multiclass_nll_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> float:
    idx = torch.arange(targets.shape[0])
    return float(-torch.log(probs[idx, targets].clamp_min(1e-8)).mean().item())



def brier_binary_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> float:
    p1 = probs[:, 1]
    y = targets.float()
    return float(((p1 - y) ** 2).mean().item())



def ece_binary_from_probs(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    conf, pred = probs.max(dim=1)
    acc = pred.eq(targets)
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            bin_acc = acc[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece += mask.float().mean() * (bin_acc - bin_conf).abs()
    return float(ece.item())



def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)


# -----------------------------
# Models
# -----------------------------


class DeterministicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianMLP(PyroModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        prior_std: float = 0.1,
        dropout_p: float = 0.0,
        fan_in_scaled_prior: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prior_std = prior_std
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.relu = nn.ReLU()

        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        fc1_std = prior_std / math.sqrt(max(input_dim, 1)) if fan_in_scaled_prior else prior_std
        self.fc1.weight = PyroSample(dist.Normal(0.0, fc1_std).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0.0, prior_std).expand([hidden_dim]).to_event(1))

        self.out = PyroModule[nn.Linear](hidden_dim, 2)
        out_std = prior_std / math.sqrt(max(hidden_dim, 1)) if fan_in_scaled_prior else prior_std
        self.out.weight = PyroSample(dist.Normal(0.0, out_std).expand([2, hidden_dim]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0.0, prior_std).expand([2]).to_event(1))

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.out(x)
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


# -----------------------------
# Training + evaluation
# -----------------------------


def train_deterministic(
    splits: Dict[str, torch.Tensor],
    cfg: DeterministicConfig,
    device: torch.device,
) -> TrainingArtifacts:
    scaler = fit_standardizer(splits["x_train"])
    x_train = scaler.transform(splits["x_train"])
    x_val = scaler.transform(splits["x_val"])
    y_train = splits["y_train"]
    y_val = splits["y_val"]

    train_loader = make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg.batch_size, shuffle=False)

    model = DeterministicMLP(x_train.shape[1], hidden_dim=cfg.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_nll = float("inf")
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        val_metrics = evaluate_deterministic(model, val_loader, device)
        if val_metrics["nll"] < best_val_nll - 1e-6:
            best_val_nll = val_metrics["nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingArtifacts(standardizer=scaler, model=model)



def evaluate_deterministic(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1).cpu()
            all_probs.append(probs)
            all_targets.append(yb.cpu())

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    preds = probs.argmax(dim=1)
    return {
        "accuracy": float((preds == targets).float().mean().item()),
        "nll": multiclass_nll_from_probs(probs, targets),
        "brier": brier_binary_from_probs(probs, targets),
        "ece": ece_binary_from_probs(probs, targets),
    }



def train_bnn(
    splits: Dict[str, torch.Tensor],
    cfg: BNNConfig,
    device: torch.device,
) -> TrainingArtifacts:
    ensure_pyro_available()
    pyro.clear_param_store()

    scaler = fit_standardizer(splits["x_train"])
    x_train = scaler.transform(splits["x_train"])
    x_val = scaler.transform(splits["x_val"])
    y_train = splits["y_train"]
    y_val = splits["y_val"]

    train_loader = make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg.batch_size, shuffle=False)

    model = BayesianMLP(
        input_dim=x_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        prior_std=cfg.prior_std,
        dropout_p=cfg.dropout_p,
        fan_in_scaled_prior=cfg.fan_in_scaled_prior,
    ).to(device)

    if cfg.guide != "diag":
        raise ValueError(f"Unsupported guide='{cfg.guide}'. Only 'diag' is implemented.")

    guide = AutoDiagonalNormal(model)
    svi = pyro.infer.SVI(
        model=model,
        guide=guide,
        optim=PyroAdam({"lr": cfg.lr}),
        loss=pyro.infer.TraceMeanField_ELBO(),
    )

    best_store_state = None
    best_val_nll = float("inf")
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            svi.step(xb, yb)

        val_metrics = evaluate_bnn(model, guide, val_loader, device, mc_samples=cfg.mc_samples_eval)
        if val_metrics["nll"] < best_val_nll - 1e-6:
            best_val_nll = val_metrics["nll"]
            best_store_state = {k: v.detach().cpu().clone() for k, v in pyro.get_param_store().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_store_state is not None:
        pyro.clear_param_store()
        for key, value in best_store_state.items():
            pyro.get_param_store()[key] = value.clone()

    return TrainingArtifacts(standardizer=scaler, model=model, guide=guide)



def predict_bnn_mc(model: BayesianMLP, guide: Any, x: torch.Tensor, mc_samples: int, device: torch.device) -> torch.Tensor:
    ensure_pyro_available()
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=mc_samples, return_sites=("_RETURN",))
    sampled_logits = predictive(x.to(device))["_RETURN"]  # [S, B, C]
    return torch.softmax(sampled_logits, dim=-1).cpu()



def evaluate_bnn(
    model: BayesianMLP,
    guide: Any,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int = 100,
) -> Dict[str, float]:
    ensure_pyro_available()
    model.eval()
    all_mean_probs, all_targets = [], []
    pred_entropies, exp_entropies, mutual_infos = [], [], []

    for xb, yb in loader:
        mc_probs = predict_bnn_mc(model, guide, xb, mc_samples=mc_samples, device=device)
        mean_probs = mc_probs.mean(dim=0)
        pred_entropy = entropy_from_probs(mean_probs)
        exp_entropy = entropy_from_probs(mc_probs).mean(dim=0)
        mutual_info = pred_entropy - exp_entropy

        all_mean_probs.append(mean_probs)
        all_targets.append(yb.cpu())
        pred_entropies.append(pred_entropy)
        exp_entropies.append(exp_entropy)
        mutual_infos.append(mutual_info)

    probs = torch.cat(all_mean_probs)
    targets = torch.cat(all_targets)
    preds = probs.argmax(dim=1)
    return {
        "accuracy": float((preds == targets).float().mean().item()),
        "nll": multiclass_nll_from_probs(probs, targets),
        "brier": brier_binary_from_probs(probs, targets),
        "ece": ece_binary_from_probs(probs, targets),
        "predictive_entropy": float(torch.cat(pred_entropies).mean().item()),
        "aleatoric_entropy": float(torch.cat(exp_entropies).mean().item()),
        "epistemic_mi": float(torch.cat(mutual_infos).mean().item()),
    }



def evaluate_artifacts(
    artifacts: TrainingArtifacts,
    splits: Dict[str, torch.Tensor],
    model_type: str,
    batch_size: int,
    device: torch.device,
    mc_samples: int = 100,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    scaler = artifacts.standardizer

    for split in ["val", "test"]:
        x_key, y_key = f"x_{split}", f"y_{split}"
        if x_key not in splits or y_key not in splits:
            continue
        x = scaler.transform(splits[x_key])
        y = splits[y_key]
        loader = make_loader(x, y, batch_size=batch_size, shuffle=False)
        if model_type == "deterministic":
            split_metrics = evaluate_deterministic(artifacts.model, loader, device)
        elif model_type == "bnn":
            split_metrics = evaluate_bnn(artifacts.model, artifacts.guide, loader, device, mc_samples=mc_samples)
        else:
            raise ValueError(f"Unknown model_type='{model_type}'")
        metrics.update({f"{split}_{k}": v for k, v in split_metrics.items()})
    return metrics


# -----------------------------
# Experiment runners
# -----------------------------


def run_bnn_config(
    bundle: Dict[str, Any],
    cfg: BNNConfig,
    bert_dim: Optional[int],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)
    splits = load_feature_splits(bundle, cfg.feature_mode, bert_dim=bert_dim)
    artifacts = train_bnn(splits, cfg, device)
    metrics = evaluate_artifacts(
        artifacts,
        splits,
        model_type="bnn",
        batch_size=cfg.batch_size,
        device=device,
        mc_samples=cfg.mc_samples_eval,
    )
    row = asdict(cfg)
    row.update({"seed": seed, "model": "bnn", **metrics})
    return row



def run_det_config(
    bundle: Dict[str, Any],
    cfg: DeterministicConfig,
    bert_dim: Optional[int],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)
    splits = load_feature_splits(bundle, cfg.feature_mode, bert_dim=bert_dim)
    artifacts = train_deterministic(splits, cfg, device)
    metrics = evaluate_artifacts(
        artifacts,
        splits,
        model_type="deterministic",
        batch_size=cfg.batch_size,
        device=device,
    )
    row = asdict(cfg)
    row.update({"seed": seed, "model": "deterministic"})
    row["prior_std"] = float("nan")
    row["dropout_p"] = float("nan")
    row["mc_samples_eval"] = float("nan")
    row["fan_in_scaled_prior"] = float("nan")
    row["guide"] = "-"
    row.update(metrics)
    return row



def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def aggregate_rows(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    numeric_exclude = set(group_keys) | {"seed"}
    for key, bucket in buckets.items():
        agg: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        for metric_key in bucket[0].keys():
            if metric_key in numeric_exclude:
                continue
            values = [r[metric_key] for r in bucket if isinstance(r.get(metric_key), (int, float)) and not (isinstance(r.get(metric_key), float) and math.isnan(r.get(metric_key)))]
            if values:
                agg[f"mean_{metric_key}"] = float(np.mean(values))
                agg[f"std_{metric_key}"] = float(np.std(values))
            elif metric_key not in agg:
                agg[metric_key] = bucket[0].get(metric_key)
        agg["n_runs"] = len(bucket)
        out.append(agg)
    return out



def choose_best_bnn_config(aggregated_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not aggregated_rows:
        raise ValueError("No aggregated rows provided.")
    ranked = sorted(
        aggregated_rows,
        key=lambda r: (
            r.get("mean_val_nll", float("inf")),
            r.get("mean_val_ece", float("inf")),
            -r.get("mean_val_accuracy", float("-inf")),
        ),
    )
    best = ranked[0].copy()
    return best



def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
