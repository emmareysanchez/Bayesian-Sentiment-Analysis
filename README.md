# Bayesian Sentiment Analysis: Confidence Calibration & OOD Detection

**Jimena Monteagudo & Emma Rey**  
Master's in Artificial Intelligence — Probabilistic AI (2025/2026)

---

## 📌 Overview

This project implements a **probabilistic sentiment classifier** that goes beyond point predictions. Instead of forcing a hard Positive/Negative label, the system outputs a full predictive distribution p(y|x) and decomposes uncertainty into its aleatoric and epistemic components.

The pipeline combines two probabilistic stages:

1. **Soft Dirichlet Clustering (LDA)** — each review is represented as a soft topic distribution θ ~ Dir(α), capturing semantic structure and ambiguity before classification.
2. **Bayesian Neural Network (BNN) with Variational Inference** — weights are distributions, not fixed values. Trained by maximizing the ELBO. Uncertainty is estimated via Monte Carlo sampling.

If predictive entropy H[p(y|x)] exceeds a critical threshold τ, the system **abstains** and routes the sample to human review — reducing operational risk in production.

---

## 🗂️ Repository Structure

```
sentiment-bayesian/
├── data/
│   ├── raw/                # Original dataset (not tracked by git)
│   ├── processed/          # Cleaned and tokenized data
│   └── ood/                # Out-of-distribution samples for evaluation
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_lda_clustering.ipynb
│   ├── 04_bnn_training.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data/               # Loaders and preprocessing
│   ├── models/             # LDA, BNN, end-to-end pipeline
│   ├── inference/          # ELBO, variational guide, uncertainty decomposition
│   ├── evaluation/         # ECE, NLL, AUROC, rejection curves
│   └── utils/              # Visualization helpers
├── experiments/
│   ├── configs/            # YAML hyperparameter files
│   └── results/            # Saved metrics and plots
├── app/
│   └── streamlit_app.py    # Interactive demo
├── paper/
│   └── main.tex            # NeurIPS 2025 format report
├── Dockerfile
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/sentiment-bayesian.git
cd sentiment-bayesian
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or using Docker (recommended for full reproducibility):

```bash
docker build -t sentiment-bayesian .
docker run -p 8501:8501 sentiment-bayesian
```

### 3. Download the dataset

We use the [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). Download and place it under `data/raw/`:

```bash
# Example using the datasets library
python -c "from datasets import load_dataset; load_dataset('imdb').save_to_disk('data/raw/imdb')"
```

### 4. Run the pipeline

```bash
# Step by step
python src/data/preprocessing.py
python src/models/lda.py
python src/models/bnn.py

# Or end-to-end
python src/models/pipeline.py
```

### 5. Launch the interactive demo

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Methodology

### Stage 1 — Dirichlet Soft Clustering

We model each document as a mixture over K latent topics:

```
θ_d ~ Dir(α)          # Topic distribution for document d
z_n ~ Categorical(θ_d) # Topic assignment for word n
w_n ~ Categorical(β_z) # Word generation
```

The resulting topic vector θ_d (instead of a hard cluster label) is used as an additional feature for the BNN, enriching the input with soft semantic structure.

### Stage 2 — Bayesian Neural Network

Weights W follow distributions instead of fixed values:

```
Prior:     p(W) = N(0, I)
Posterior: q(W|φ) = N(μ, diag(σ²))   [learned via VI]
Objective: ELBO = E_q[log p(y|x,W)] - KL(q(W|φ) || p(W))
```

### Uncertainty Decomposition

Using T Monte Carlo samples from the posterior:

```
Total uncertainty     = H[E_q[p(y|x,W)]]           # Predictive entropy
Aleatoric uncertainty = E_q[H[p(y|x,W)]]            # Expected entropy
Epistemic uncertainty = Total - Aleatoric            # Mutual information
```

### Rejection Criterion

```
If H[p(y|x)] > τ  →  abstain, route to human review
```

The threshold τ is selected via ROC analysis on the validation set.

---

## 📊 Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| NLL | Offline | Negative Log-Likelihood — sharpness of predictive distribution |
| ECE | Offline | Expected Calibration Error — reliability of confidence scores |
| AUROC (rejection) | Offline | Quality of uncertainty as a rejection signal |
| Coverage / Precision | Offline | Trade-off curve for the abstention mechanism |
| Human override rate | Online | How often experts override the model based on shown uncertainty |

---

## 📦 Dependencies

Main libraries used:

- `pyro-ppl` — Probabilistic programming (BNN + VI)
- `gensim` — LDA topic modelling
- `transformers` — Text embeddings
- `streamlit` — Interactive demo
- `torchmetrics` — ECE, NLL, AUROC
- `mlflow` *(optional)* — Experiment tracking

See `requirements.txt` for full list with pinned versions.

---

## 📄 Paper

The full report follows the **NeurIPS 2025** template and is located in `paper/main.tex`. Maximum 9 pages (excluding references and appendices).

---

## ⚠️ Notes on Generative AI Usage

AI tools (Claude, ChatGPT) were used for coding assistance and writing support. All mathematical derivations and methodological justifications are the authors' own work, consistent with the implemented code.
