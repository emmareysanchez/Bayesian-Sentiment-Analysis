"""Streamlit demo — Bayesian Sentiment Analyzer (bnn_moe).

Usage (from project root):
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pyro
from pyro.infer.autoguide import AutoNormal

from src.data.loader import Standardizer, load_bow, load_tfidf
from src.data.preprocessing import clean_text
from src.evaluation.uncertainty import decompose_mc
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig
from src.models.deterministic import DeterministicMLP
from src.models.mc_dropout import MCDropoutMLP
from src.models.slda import AmortizedSLDA

AVAILABLE_MODELS = ["deterministic", "mc_dropout", "bnn_base", "bnn_moe", "bnn_moe_hetero"]
MODELS_ROOT = Path("experiments/results/models")
SLDA_DIR    = Path("experiments/results/slda")
METRICS_RAW = Path("experiments/results/evaluation/metrics_raw.csv")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# -----------------------------
# Predefined examples
# -----------------------------

EXAMPLES = {
    "— write your own —": "",
    "Positive review": (
        "An absolute masterpiece. The performances are extraordinary — every single one. "
        "The direction is flawless, the pacing is perfect, and the emotional impact is "
        "devastating in the best possible way. I was completely absorbed from the very first "
        "minute to the last. One of the greatest productions I have ever had the privilege "
        "of watching. Unmissable."
    ),
    "Negative review": (
        "A complete and utter disaster from start to finish. The performances are wooden and "
        "unconvincing, the pacing is unbearably slow, and the plot is a mess. I kept waiting "
        "for something — anything — to improve, but it only got worse. One of the most tedious "
        "and poorly made productions I have ever been unfortunate enough to sit through. "
        "Avoid at all costs."
    ),
    "Ambiguous review": (
        "There are genuinely impressive moments here, and the central performance is committed "
        "and often moving. But the pacing drags in the second half, and the ending feels rushed "
        "and deeply unsatisfying. A frustrating mixed bag that hints at something much greater "
        "but never quite delivers on its promise. Worth a watch for the highlights, but do not "
        "expect it to come together as a whole."
    ),
}

# -----------------------------
# Helpers
# -----------------------------

def _available_seeds(model_name: str) -> list[str]:
    d = MODELS_ROOT / model_name
    if not d.exists():
        return []
    return sorted([s.name for s in d.iterdir() if s.is_dir() and s.name.startswith("seed_")])


def _best_seed(model_name: str) -> str | None:
    if not METRICS_RAW.exists():
        return None
    try:
        import csv
        best, best_acc = None, -1.0
        with open(METRICS_RAW, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == model_name and row["split"] == "test":
                    acc = float(row["accuracy"])
                    if acc > best_acc:
                        best_acc = acc
                        best = f"seed_{row['seed']}"
        return best
    except Exception:
        return None


def _slda_vectorizer_type() -> str:
    p = SLDA_DIR / "slda_meta.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("vectorizer", "bow")
    return "bow"


# -----------------------------
# Load artifacts
# -----------------------------

@st.cache_resource
def load_bert(device_str: str):
    from transformers import DistilBertModel, DistilBertTokenizer
    tok   = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval().to(torch.device(device_str))
    return tok, model


@st.cache_resource
def load_slda_model():
    if not (SLDA_DIR / "lda.pkl").exists():
        return None, None
    try:
        slda = AmortizedSLDA.load(SLDA_DIR, device="cpu")
        _, vec = load_bow() if _slda_vectorizer_type() == "bow" else load_tfidf()
        return slda, vec
    except Exception as e:
        st.warning(f"Could not load sLDA model (numpy version mismatch?): {e}. Topic gating disabled.")
        return None, None


@st.cache_resource
def load_model(model_name: str, seed_name: str, device_str: str):
    device   = torch.device(device_str)
    seed_dir = MODELS_ROOT / model_name / seed_name
    if not seed_dir.exists():
        return None

    if model_name in ("deterministic", "mc_dropout"):
        ck        = torch.load(seed_dir / "state.pt", map_location="cpu", weights_only=False)
        input_dim = ck["input_dim"]
        model     = DeterministicMLP(input_dim=input_dim) if model_name == "deterministic" \
                    else MCDropoutMLP(input_dim=input_dim, dropout_p=ck.get("dropout_p", 0.2))
        model.load_state_dict(ck["state"])
        model.eval().to(device)
        scaler = Standardizer(ck["scaler_mean"], ck["scaler_std"])
        return {
            "name": model_name, "kind": model_name,
            "model": model, "guide": None,
            "scaler": scaler, "use_theta": False,
            "device": device, "_param_snapshot": None,
        }

    # bnn_moe
    meta  = torch.load(seed_dir / "meta.pt", map_location="cpu", weights_only=False)
    cfg   = BayesianMoEConfig(**meta["cfg"])
    pyro.clear_param_store()
    state = torch.load(str(seed_dir / "pyro_store.pt"), map_location=device, weights_only=False)
    pyro.get_param_store().set_state(state)
    model = BayesianMoE(cfg).to(device)
    guide = AutoNormal(model)
    dummy_x = torch.zeros(2, cfg.input_dim, device=device)
    dummy_y = torch.zeros(2, dtype=torch.long, device=device)
    if meta["use_theta"]:
        guide(dummy_x, torch.ones(2, cfg.n_experts, device=device) / cfg.n_experts, dummy_y)
    else:
        guide(dummy_x, None, dummy_y)
    scaler         = Standardizer(meta["scaler_mean"], meta["scaler_std"])
    param_snapshot = {k: v.detach().clone() for k, v in pyro.get_param_store().items()}
    return {
        "name": model_name, "kind": "bnn",
        "cfg": cfg, "model": model, "guide": guide,
        "scaler": scaler, "use_theta": bool(meta["use_theta"]),
        "device": device, "_param_snapshot": param_snapshot,
    }


# -----------------------------
# Inference
# -----------------------------

def encode_text(text: str, tok, bert, device: torch.device) -> np.ndarray:
    text = clean_text(text) or "[UNK]"
    enc  = tok([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
    enc  = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        h = bert(**enc).last_hidden_state[:, 0, :]
    return h.detach().cpu().float().numpy()   # (1, 768)


def theta_for_text(text: str, slda, vec) -> np.ndarray | None:
    if slda is None or vec is None:
        return None
    cleaned = clean_text(text) or "[UNK]"
    return slda.theta(vec.transform([cleaned]))   # (1, K)


def predict(text: str, bundle: dict, tok, bert, slda=None, vec=None, mc_samples: int = 100):
    device = bundle["device"]
    emb    = encode_text(text, tok, bert, device)          # (1, 768)
    emb_t  = torch.tensor(emb, dtype=torch.float32)
    emb_std = bundle["scaler"].transform(emb_t).to(device)

    theta_np = None
    if bundle["use_theta"]:
        theta_np = theta_for_text(text, slda, vec)
        if theta_np is None:
            k = bundle["cfg"].n_experts
            theta_np = np.ones((1, k), dtype=np.float32) / k
        theta_t = torch.tensor(theta_np, dtype=torch.float32, device=device)
    else:
        theta_t = None

    if bundle["kind"] == "deterministic":
        with torch.inference_mode():
            probs = torch.softmax(bundle["model"](emb_std), dim=-1)
        mc = probs.unsqueeze(0)
    elif bundle["kind"] == "mc_dropout":
        with torch.inference_mode():
            logits = bundle["model"].mc_forward(emb_std, mc_samples=mc_samples)
            mc = torch.softmax(logits, dim=-1)
    else:  # bnn_moe — restore param store before inference
        pyro.clear_param_store()
        for k, v in bundle["_param_snapshot"].items():
            pyro.get_param_store()[k] = v.detach().clone()
        predictive = pyro.infer.Predictive(
            bundle["model"], guide=bundle["guide"],
            num_samples=mc_samples, return_sites=("_RETURN",),
        )
        logits = predictive(emb_std, theta_t)["_RETURN"] if theta_t is not None \
                 else predictive(emb_std)["_RETURN"]
        mc = torch.softmax(logits, dim=-1)

    decomp = decompose_mc(mc)
    mean_probs = mc.mean(0).squeeze(0).detach().cpu().numpy()
    pred_idx   = int(mean_probs.argmax())
    label_map  = {0: "NEGATIVE", 1: "POSITIVE"}
    shown_label = label_map[pred_idx]

    return {
        "mean_probs":          mean_probs,
        "predictive_entropy":  float(decomp["predictive_entropy"].item()),
        "aleatoric_entropy":   float(decomp["aleatoric_entropy"].item()),
        "mutual_info":         float(decomp["mutual_info"].item()),
        "theta":               theta_np[0] if theta_np is not None else None,
        "mc_samples_count":    int(mc.shape[0]),
    }


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Bayesian Sentiment", page_icon="🧠", layout="wide")
st.title("🧠 Bayesian Sentiment Analyzer")
st.caption("Jimena Monteagudo & Emma Rey — MUIA 2025/2026 · IA Probabilística")

with st.sidebar:
    st.header("Model")

    selected_model = st.selectbox(
        "Architecture",
        [m for m in AVAILABLE_MODELS if (MODELS_ROOT / m).exists()],
        index=AVAILABLE_MODELS.index("bnn_moe") if "bnn_moe" in AVAILABLE_MODELS else 0,
    )

    seeds = _available_seeds(selected_model)
    if not seeds:
        st.error(f"No trained seeds found for {selected_model}.")
        st.stop()

    best          = _best_seed(selected_model)
    seed_options  = ["Auto (best accuracy)"] + seeds
    seed_choice   = st.selectbox("Seed", seed_options, index=0)
    resolved_seed = (best if best and best in seeds else seeds[0]) \
                    if seed_choice == "Auto (best accuracy)" else seed_choice

    if seed_choice == "Auto (best accuracy)" and best:
        st.caption(f"Best seed by test accuracy: **{resolved_seed}**")

    bundle = load_model(selected_model, resolved_seed, str(DEVICE))
    if bundle is None:
        st.error(f"Could not load {selected_model} / {resolved_seed}")
        st.stop()

    bayesian = bundle["kind"] == "bnn"
    st.info(f"Loaded: **{selected_model}** · {resolved_seed} · {DEVICE}")
    st.write(f"Bayesian: {bayesian} · Topic gating: {bundle['use_theta']}")

    if best and METRICS_RAW.exists():
        try:
            import csv
            with open(METRICS_RAW, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["model"] == selected_model and row["split"] == "test" \
                            and f"seed_{row['seed']}" == resolved_seed:
                        st.caption(f"Test accuracy (this seed): {float(row['accuracy']):.3f}")
                        break
        except Exception:
            pass

    st.divider()
    st.header("Sampling")
    mc_samples = st.slider("MC samples", 20, 200, 100, step=10,
                           disabled=not bayesian)
    st.caption("Only applies to Bayesian models." if not bayesian else
               "More samples → more stable uncertainty estimate.")

    st.divider()
    st.header("Decision rule")
    default_tau = 0.4
    try:
        import pandas as pd
        biz = pd.read_csv("experiments/results/business/business_summary.csv")
        row = biz[biz["model"] == selected_model]
        if not row.empty:
            default_tau = float(row.iloc[0]["mean_tau_star"])
            st.caption(f"τ* from business analysis = {default_tau:.3f}")
    except Exception:
        pass
    tau = st.slider("Rejection threshold τ (predictive entropy)",
                    0.0, 0.693, default_tau, step=0.005)
    st.caption("Higher τ → model abstains less often.")

    if bundle["use_theta"]:
        st.caption(f"sLDA vectorizer: **{_slda_vectorizer_type()}**")

tok, bert = load_bert(str(DEVICE))
slda, vec = load_slda_model() if bundle.get("use_theta") else (None, None)

# -----------------------------
# Main panel
# -----------------------------

example_choice = st.selectbox("Predefined example", list(EXAMPLES.keys()), index=0)
text = st.text_area(
    "Review text",
    value=EXAMPLES[example_choice],
    height=180,
    placeholder="Paste a movie review here...",
)

if st.button("Analyze", type="primary"):
    with st.spinner("Running Monte Carlo predictions..."):
        out = predict(text, bundle, tok, bert, slda=slda, vec=vec, mc_samples=mc_samples)

    p_neg = float(out["mean_probs"][0])
    p_pos = float(out["mean_probs"][1])
    H     = out["predictive_entropy"]
    H_ale = out["aleatoric_entropy"]
    MI    = out["mutual_info"]

    decision = "🙋 ABSTAIN (human review)" if H > tau \
               else ("😊 POSITIVE" if p_pos >= p_neg else "😞 NEGATIVE")

    st.subheader("Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("p(negative)", f"{p_neg:.3f}")
    c2.metric("p(positive)", f"{p_pos:.3f}")
    c3.metric("Decision", decision)

    if H > tau:
        st.warning(f"Predictive entropy {H:.3f} > τ = {tau:.3f}. Routing to human review.")
    else:
        st.success(f"Predictive entropy {H:.3f} ≤ τ = {tau:.3f}. Auto-classify.")

    st.subheader("Uncertainty")
    if bayesian:
        d1, d2, d3 = st.columns(3)
        d1.metric("Predictive entropy H[ȳ]", f"{H:.4f}",    help="Total uncertainty — used for τ")
        d2.metric("Aleatoric  E[H(p)]",      f"{H_ale:.4f}", help="Irreducible data noise")
        d3.metric("Epistemic  MI",           f"{MI:.4f}",    help="Model uncertainty (reducible with more data)")
    else:
        st.metric("Predictive entropy", f"{H:.4f}")
        st.caption("Deterministic model — no epistemic/aleatoric decomposition available.")

    if out["theta"] is not None:
        st.subheader("Topic mixture θ (sLDA)")
        try:
            with open(SLDA_DIR / "topic_words.json",    encoding="utf-8") as f: tw = json.load(f)
            with open(SLDA_DIR / "topic_sentiment.json", encoding="utf-8") as f: ts = json.load(f)
        except Exception:
            tw, ts = {}, {}
        theta = out["theta"]
        for k in np.argsort(theta)[::-1][:3]:
            words = " ".join(tw.get(str(int(k)), [])[:8])
            sent  = float(ts.get(str(int(k)), 0.0))
            tag   = "POS" if sent > 0.1 else ("NEG" if sent < -0.1 else "neu")
            st.write(f"- **Topic {int(k)}** ({theta[k]*100:.1f}%, {tag} {sent:+.2f}): _{words}_")

    with st.expander("Raw output"):
        st.json({
            "mean_probs":         out["mean_probs"].tolist(),
            "predictive_entropy": H,
            "aleatoric_entropy":  H_ale,
            "mutual_info":        MI,
            "mc_samples":         out["mc_samples_count"],
        })
