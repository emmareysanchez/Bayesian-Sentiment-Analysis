"""Streamlit — BNN experiment comparison.

Compares any two BNN variants side-by-side:
  bnn_moe          (cold-start, standard KL)
  bnn_moe_warm     (warm-start from deterministic)
  bnn_moe_kl_full  (KL sweep β=1.0, same as cold but retrained)
  bnn_moe_kl_div10 (KL / 10, β=0.1)
  bnn_moe_kl_div100(KL / 100, β=0.01)

Usage (from project root):
    streamlit run app/streamlit_app_warm.py --server.port 8502
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

from src.data.loader import Standardizer, load_bow
from src.data.preprocessing import clean_text
from src.evaluation.uncertainty import decompose_mc
from src.models.bnn_moe import BayesianMoE, BayesianMoEConfig
from src.models.slda import AmortizedSLDA

MODELS_ROOT = Path("experiments/results/models")
SLDA_DIR    = Path("experiments/results/slda")

ALL_MODELS = {
    "bnn_moe (cold, β=1)":   "bnn_moe",
    "bnn_moe (warm, β=1)":   "bnn_moe_warm",
    "bnn_moe (KL full, β=1)":"bnn_moe_kl_full",
    "bnn_moe (KL/10, β=0.1)":"bnn_moe_kl_div10",
    "bnn_moe (KL/100, β=0.01)":"bnn_moe_kl_div100",
}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

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


def _available_labels() -> list[str]:
    return [label for label, folder in ALL_MODELS.items()
            if (MODELS_ROOT / folder).exists()]


def _available_seeds(label: str) -> list[str]:
    folder = ALL_MODELS[label]
    d = MODELS_ROOT / folder
    if not d.exists():
        return []
    return sorted([s.name for s in d.iterdir() if s.is_dir() and s.name.startswith("seed_")])


def _common_seeds(label_a: str, label_b: str) -> list[str]:
    return sorted(set(_available_seeds(label_a)) & set(_available_seeds(label_b)))


@st.cache_resource
def load_bert_model(device_str: str):
    from transformers import DistilBertModel, DistilBertTokenizer
    tok   = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval().to(torch.device(device_str))
    return tok, model


@st.cache_resource
def load_slda():
    if not (SLDA_DIR / "lda.pkl").exists():
        return None, None
    try:
        slda = AmortizedSLDA.load(SLDA_DIR, device="cpu")
        _, vec = load_bow()
        return slda, vec
    except Exception as e:
        st.warning(f"sLDA failed to load: {e}. Topic gating disabled.")
        return None, None


@st.cache_resource
def load_bnn_bundle(model_label: str, seed_name: str, device_str: str):
    folder   = ALL_MODELS[model_label]
    device   = torch.device(device_str)
    seed_dir = MODELS_ROOT / folder / seed_name
    if not seed_dir.exists():
        return None
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
    kl_beta        = meta.get("kl_beta", 1.0)
    return {
        "label": model_label, "cfg": cfg, "model": model, "guide": guide,
        "scaler": scaler, "use_theta": bool(meta["use_theta"]),
        "device": device, "_param_snapshot": param_snapshot, "kl_beta": kl_beta,
    }


def encode_text(text: str, tok, bert, device: torch.device) -> np.ndarray:
    text = clean_text(text) or "[UNK]"
    enc  = tok([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
    enc  = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        h = bert(**enc).last_hidden_state[:, 0, :]
    return h.detach().cpu().float().numpy()


def theta_for_text(text: str, slda, vec):
    if slda is None or vec is None:
        return None
    return slda.theta(vec.transform([clean_text(text) or "[UNK]"]))


def run_inference(bundle: dict, emb: np.ndarray, theta_np, mc_samples: int) -> dict:
    device  = bundle["device"]
    emb_std = bundle["scaler"].transform(torch.tensor(emb, dtype=torch.float32)).to(device)

    if bundle["use_theta"]:
        if theta_np is not None:
            theta_t = torch.tensor(theta_np, dtype=torch.float32, device=device)
        else:
            k       = bundle["cfg"].n_experts
            theta_t = torch.ones(1, k, device=device) / k
    else:
        theta_t = None

    pyro.clear_param_store()
    for k, v in bundle["_param_snapshot"].items():
        pyro.get_param_store()[k] = v.detach().clone()

    predictive = pyro.infer.Predictive(
        bundle["model"], guide=bundle["guide"],
        num_samples=mc_samples, return_sites=("_RETURN",),
    )
    logits = predictive(emb_std, theta_t)["_RETURN"] if theta_t is not None \
             else predictive(emb_std)["_RETURN"]
    mc     = torch.softmax(logits, -1)
    decomp = decompose_mc(mc)
    probs  = mc.mean(0).squeeze(0).detach().cpu().numpy()
    return {
        "probs":  probs,
        "H":      float(decomp["predictive_entropy"].item()),
        "H_ale":  float(decomp["aleatoric_entropy"].item()),
        "MI":     float(decomp["mutual_info"].item()),
        "n_samp": int(mc.shape[0]),
    }


def render_result(col, out: dict, title: str, tau: float):
    with col:
        pred     = int(out["probs"].argmax())
        H        = out["H"]
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        if H > tau:
            color = ":orange[ABSTAIN]"
        elif pred == 1:
            color = ":green[POSITIVE]"
        else:
            color = ":red[NEGATIVE]"

        st.markdown(f"### {title}")
        st.markdown(f"**Decision:** {color}")
        c1, c2 = st.columns(2)
        c1.metric("p(negative)", f"{out['probs'][0]:.3f}")
        c2.metric("p(positive)", f"{out['probs'][1]:.3f}")
        st.metric("Predictive entropy H", f"{H:.4f}")
        d1, d2 = st.columns(2)
        d1.metric("Aleatoric E[H(p)]", f"{out['H_ale']:.4f}")
        d2.metric("Epistemic MI",       f"{out['MI']:.4f}")
        if H > tau:
            st.warning(f"H={H:.3f} > τ={tau:.3f} → human review")
        else:
            st.success(f"H={H:.3f} ≤ τ={tau:.3f} → auto-classify")


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------

st.set_page_config(page_title="BNN Experiments", page_icon="🔬", layout="wide")
st.title("🔬 BNN Experiment Comparison")
st.caption("Warm-start · KL sweep · MUIA 2025/2026")

available_labels = _available_labels()
if len(available_labels) < 1:
    st.error("No trained models found. Run scripts 06 and/or 07 first.")
    st.stop()

with st.sidebar:
    st.header("Models")
    default_left  = available_labels[0]
    default_right = available_labels[1] if len(available_labels) > 1 else available_labels[0]

    label_left  = st.selectbox("Reference model (left)",     available_labels,
                               index=0)
    label_right = st.selectbox("Comparison model (right)",   available_labels,
                               index=min(1, len(available_labels) - 1))

    common = _common_seeds(label_left, label_right)
    if not common:
        all_seeds = sorted(set(_available_seeds(label_left)) | set(_available_seeds(label_right)))
        seed_choice = st.selectbox("Seed", all_seeds or ["none"], index=0)
        st.warning("Models don't share this seed — only the one that has it will load.")
    else:
        seed_choice = st.selectbox("Seed", common, index=0)

    st.divider()
    st.header("Sampling")
    mc_samples = st.slider("MC samples", 20, 200, 100, step=10)

    st.divider()
    st.header("Decision rule")
    tau = st.slider("Rejection threshold τ", 0.0, 0.693, 0.35, step=0.005)

    st.divider()
    st.caption(f"Device: **{DEVICE}**")

bundle_left  = load_bnn_bundle(label_left,  seed_choice, str(DEVICE))
bundle_right = load_bnn_bundle(label_right, seed_choice, str(DEVICE))

if bundle_left is None:
    st.error(f"Could not load {label_left} / {seed_choice}")
    st.stop()
if bundle_right is None:
    st.error(f"Could not load {label_right} / {seed_choice}")
    st.stop()

tok, bert = load_bert_model(str(DEVICE))
slda, vec  = load_slda()

example_choice = st.selectbox("Predefined example", list(EXAMPLES.keys()), index=0)
text = st.text_area("Review text", value=EXAMPLES[example_choice], height=180,
                    placeholder="Paste a movie review here...")

if st.button("Analyze both models", type="primary"):
    with st.spinner("Running inference..."):
        emb      = encode_text(text, tok, bert, DEVICE)
        theta_np = theta_for_text(text, slda, vec)
        out_left  = run_inference(bundle_left,  emb, theta_np, mc_samples)
        out_right = run_inference(bundle_right, emb, theta_np, mc_samples)

    col_l, col_r = st.columns(2)
    render_result(col_l, out_left,  label_left,  tau)
    render_result(col_r, out_right, label_right, tau)

    st.divider()
    delta_pp = out_right["probs"][1] - out_left["probs"][1]
    delta_H  = out_right["H"]        - out_left["H"]
    delta_MI = out_right["MI"]       - out_left["MI"]
    st.markdown(
        f"**Δ p(positive):** `{delta_pp:+.4f}` &nbsp;&nbsp; "
        f"**Δ H:** `{delta_H:+.4f}` &nbsp;&nbsp; "
        f"**Δ MI (epistemic):** `{delta_MI:+.4f}`"
    )

    if theta_np is not None and slda is not None:
        st.subheader("Topic mixture θ (sLDA) — shared input")
        try:
            with open(SLDA_DIR / "topic_words.json",     encoding="utf-8") as f: tw = json.load(f)
            with open(SLDA_DIR / "topic_sentiment.json", encoding="utf-8") as f: ts = json.load(f)
        except Exception:
            tw, ts = {}, {}
        theta = theta_np[0]
        for k in np.argsort(theta)[::-1][:3]:
            words = " ".join(tw.get(str(int(k)), [])[:8])
            sent  = float(ts.get(str(int(k)), 0.0))
            tag   = "POS" if sent > 0.1 else ("NEG" if sent < -0.1 else "neu")
            st.write(f"- **Topic {int(k)}** ({theta[k]*100:.1f}%, {tag} {sent:+.2f}): _{words}_")

    with st.expander("Raw output"):
        st.json({
            label_left:  {**out_left,  "probs": out_left["probs"].tolist()},
            label_right: {**out_right, "probs": out_right["probs"].tolist()},
        })
