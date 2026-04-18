"""Streamlit demo for the Bayesian sentiment analyzer.

Usage (from project root):
    streamlit run app/streamlit_app.py

Features:
    - Paste a review text
    - Get p(y|x) with Monte Carlo BNN prediction
    - See uncertainty decomposition (aleatoric, epistemic)
    - See decision: classify automatically, or route to human review

The app loads the first seed of the best available model (prefers bnn_moe_hetero
-> bnn_moe -> bnn_base -> mc_dropout -> deterministic).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pyro
from pyro.infer.autoguide import AutoNormal

from src.data.loader import Standardizer, load_tfidf
from src.evaluation.uncertainty import decompose_mc
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig
from src.models.deterministic import DeterministicMLP
from src.models.mc_dropout import MCDropoutMLP
from src.models.slda import AmortizedSLDA

MODELS_PREFERENCE = [
    "bnn_moe_hetero",
    "bnn_moe",
    "bnn_base",
    "mc_dropout",
    "deterministic",
]


# -----------------------------
# Load artifacts
# -----------------------------

@st.cache_resource
def load_bert():
    from transformers import DistilBertModel, DistilBertTokenizer
    tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    return tok, model


@st.cache_resource
def load_slda(slda_dir: str = "experiments/results/slda"):
    p = Path(slda_dir)
    if not (p / "lda.pkl").exists():
        return None, None
    slda = AmortizedSLDA.load(p, device="cpu")
    _, vec = load_tfidf()
    return slda, vec


def _find_available_model(models_root: Path) -> str | None:
    for name in MODELS_PREFERENCE:
        d = models_root / name
        if d.exists():
            seeds = sorted([s for s in d.iterdir() if s.is_dir()])
            if seeds:
                return name
    return None


@st.cache_resource
def load_sentiment_model(models_root_s: str = "experiments/results/models"):
    models_root = Path(models_root_s)
    name = _find_available_model(models_root)
    if name is None:
        return None
    seed_dir = sorted([s for s in (models_root / name).iterdir() if s.is_dir()])[0]

    if name in ("deterministic", "mc_dropout"):
        ck = torch.load(seed_dir / "state.pt", map_location="cpu", weights_only=False)
        input_dim = ck["input_dim"]
        if name == "deterministic":
            model = DeterministicMLP(input_dim=input_dim)
        else:
            model = MCDropoutMLP(input_dim=input_dim, dropout_p=ck.get("dropout_p", 0.2))
        model.load_state_dict(ck["state"])
        model.eval()
        scaler = Standardizer(ck["scaler_mean"], ck["scaler_std"])
        return {
            "name": name, "kind": name, "model": model, "guide": None,
            "scaler": scaler, "use_theta": False,
        }

    # Bayesian
    meta = torch.load(seed_dir / "meta.pt", map_location="cpu", weights_only=False)
    cfg_d = meta["cfg"]
    cfg = BayesianMoEConfig(**cfg_d)
    pyro.clear_param_store()
    pyro.get_param_store().load(str(seed_dir / "pyro_store.pt"), map_location="cpu")
    model = BayesianMoE(cfg)
    guide = AutoNormal(model)
    # Warm guide with dummy input to initialize param shapes
    dummy_x = torch.zeros(2, cfg.input_dim)
    dummy_y = torch.zeros(2, dtype=torch.long)
    if meta["use_theta"]:
        dummy_th = torch.ones(2, cfg.n_experts) / cfg.n_experts
        guide(dummy_x, dummy_th, dummy_y)
    else:
        guide(dummy_x, None, dummy_y)
    scaler = Standardizer(meta["scaler_mean"], meta["scaler_std"])
    return {
        "name": name, "kind": "bnn", "model": model, "guide": guide,
        "scaler": scaler, "use_theta": bool(meta["use_theta"]),
        "cfg": cfg,
    }


# -----------------------------
# Inference
# -----------------------------

def encode_text(text: str, tok, bert) -> np.ndarray:
    enc = tok([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        h = bert(**enc).last_hidden_state[:, 0, :]
    return h.cpu().float().numpy()


def theta_for_text(text: str, slda, vec) -> np.ndarray | None:
    if slda is None or vec is None:
        return None
    X = vec.transform([text])
    return slda.theta(X)


def predict(text: str, bundle: dict, tok, bert, slda=None, vec=None, mc_samples: int = 100):
    emb = encode_text(text, tok, bert)             # [1, 768]
    emb_t = torch.tensor(emb, dtype=torch.float32)
    emb_std = bundle["scaler"].transform(emb_t)

    theta_np = None
    if bundle["use_theta"]:
        theta_np = theta_for_text(text, slda, vec)
        if theta_np is None:
            theta_np = np.ones((1, bundle["cfg"].n_experts), dtype=np.float32) / bundle["cfg"].n_experts
        theta_t = torch.tensor(theta_np, dtype=torch.float32)
    else:
        theta_t = None

    if bundle["kind"] == "deterministic":
        with torch.no_grad():
            probs = torch.softmax(bundle["model"](emb_std), dim=-1)
        mc = probs.unsqueeze(0)  # [1, 1, 2]
    elif bundle["kind"] == "mc_dropout":
        with torch.no_grad():
            logits = bundle["model"].mc_forward(emb_std, mc_samples=mc_samples)  # [S,1,2]
            mc = torch.softmax(logits, dim=-1)
    else:  # bnn
        predictive = pyro.infer.Predictive(
            bundle["model"], guide=bundle["guide"],
            num_samples=mc_samples, return_sites=("_RETURN",),
        )
        if theta_t is not None:
            logits = predictive(emb_std, theta_t)["_RETURN"]
        else:
            logits = predictive(emb_std)["_RETURN"]
        mc = torch.softmax(logits, dim=-1)

    decomp = decompose_mc(mc)
    return {
        "mean_probs": mc.mean(0).squeeze(0).numpy(),
        "predictive_entropy": float(decomp["predictive_entropy"].item()),
        "aleatoric_entropy": float(decomp["aleatoric_entropy"].item()),
        "mutual_info": float(decomp["mutual_info"].item()),
        "theta": theta_np[0] if theta_np is not None else None,
        "mc_samples_count": mc.shape[0],
    }


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Bayesian Sentiment", page_icon="🧠", layout="wide")
st.title("🧠 Bayesian Sentiment Analyzer")
st.caption("Monteagudo & Rey — MUIA 2025/2026 · IA Probabilística")

with st.sidebar:
    st.header("Model")
    bundle = load_sentiment_model()
    if bundle is None:
        st.error("No trained model found. Run `scripts/03_train_all_models.py` first.")
        st.stop()
    st.info(f"Loaded: **{bundle['name']}**")
    st.write(f"Uses topic gating: {bundle['use_theta']}")

    st.header("Decision rule")
    default_tau = 0.4
    try:
        import pandas as pd
        biz = pd.read_csv("experiments/results/business/business_summary.csv")
        row = biz[biz["model"] == bundle["name"]]
        if not row.empty:
            default_tau = float(row.iloc[0]["mean_tau_star"])
            st.caption(f"τ* from business analysis = {default_tau:.3f}")
    except Exception:
        pass
    tau = st.slider("Rejection threshold τ (on predictive entropy)",
                    0.0, 0.693, default_tau, step=0.005)
    st.caption("Higher τ → the model abstains less often")

tok, bert = load_bert()
slda, vec = load_slda() if bundle["use_theta"] else (None, None)

col1, col2 = st.columns([2, 1])
with col1:
    text = st.text_area(
        "Review text",
        value="This movie was absolutely amazing, I loved every second of it!",
        height=180,
    )

if st.button("Analyze", type="primary"):
    with st.spinner("Running Monte Carlo predictions..."):
        out = predict(text, bundle, tok, bert, slda=slda, vec=vec, mc_samples=100)

    p_neg, p_pos = float(out["mean_probs"][0]), float(out["mean_probs"][1])
    H = out["predictive_entropy"]
    H_ale = out["aleatoric_entropy"]
    MI = out["mutual_info"]

    decision = "🙋 REVIEW (abstain)" if H > tau else ("😊 POSITIVE" if p_pos >= p_neg else "😞 NEGATIVE")

    c1, c2, c3 = st.columns(3)
    c1.metric("p(negative)", f"{p_neg:.3f}")
    c2.metric("p(positive)", f"{p_pos:.3f}")
    c3.metric("Decision", decision)

    st.subheader("Uncertainty decomposition")
    d1, d2, d3 = st.columns(3)
    d1.metric("Predictive entropy  H[E(p)]", f"{H:.3f}", help="Total uncertainty")
    d2.metric("Aleatoric  E[H(p)]", f"{H_ale:.3f}", help="Data noise (irreducible)")
    d3.metric("Epistemic  MI", f"{MI:.3f}", help="Model uncertainty (reducible with more data)")

    if H > tau:
        st.warning(
            f"Predictive entropy {H:.3f} > τ = {tau:.3f}. "
            "Recommendation: route this review to human review."
        )
    else:
        st.success(f"Predictive entropy {H:.3f} ≤ τ = {tau:.3f}. Auto-classify.")

    if out["theta"] is not None:
        st.subheader("Topic mixture θ (from sLDA)")
        try:
            with open("experiments/results/slda/topic_words.json") as f:
                tw = json.load(f)
            with open("experiments/results/slda/topic_sentiment.json") as f:
                ts = json.load(f)
        except Exception:
            tw, ts = {}, {}
        theta = out["theta"]
        top3 = np.argsort(theta)[::-1][:3]
        for k in top3:
            words = " ".join(tw.get(str(int(k)), [])[:8])
            sent = float(ts.get(str(int(k)), 0.0))
            tag = "POS" if sent > 0.1 else ("NEG" if sent < -0.1 else "neu")
            st.write(f"- **Topic {int(k)}** ({theta[k]*100:.1f}%, {tag} {sent:+.2f}): _{words}_")

    st.expander("Raw output").json({
        "mean_probs": out["mean_probs"].tolist(),
        "predictive_entropy": H,
        "aleatoric_entropy": H_ale,
        "mutual_info": MI,
        "mc_samples": out["mc_samples_count"],
    })
