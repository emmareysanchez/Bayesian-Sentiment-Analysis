# Bayesian Sentiment Analysis — Proyecto Final IA Probabilística

Monteagudo & Rey — MUIA 2025/2026

Clasificador de sentimiento probabilístico con **tres capas acopladas**:

1. **sLDA supervisado amortizado** — LDA clásico cuya salida θ_d se acopla a la loss de clasificación a través de una cabeza supervisada bayesiana. Los temas pasan a ser predictivos de sentimiento, no de actores.
2. **BNN Mixture-of-Experts** — K expertos bayesianos con priors independientes, mezclados por un gate determinista g(θ_d, h_d). Captura heterogeneidad semántica.
3. **Cabeza de verosimilitud heterocedástica** — separa ruido aleatórico (σ² por input) de epistémico (varianza de los pesos) siguiendo Depeweg et al. (2018).

Evaluación: calibración (ECE, Brier, NLL), selective prediction (AURC, risk-coverage), OOD detection con shift real, análisis coste-beneficio del umbral de rechazo.

---

## Estructura del repositorio

```
.
├── src/
│   ├── data/
│   │   ├── loader.py              # DataLoaders para todos los modelos
│   │   └── preprocessing.py       # IMDb + ruido + OOD real (Amazon/AG-News)
│   ├── models/
│   │   ├── slda.py                # sLDA amortizado (LDA + cabeza supervisada)
│   │   ├── bnn_moe.py             # BNN Mixture-of-Experts con gating por temas
│   │   ├── heteroscedastic.py     # Cabeza de verosimilitud heterocedástica
│   │   ├── deterministic.py       # Baseline MLP determinista
│   │   └── mc_dropout.py          # Baseline MC-Dropout
│   ├── inference/
│   │   └── svi_trainer.py         # Wrapper SVI genérico con early stopping
│   ├── evaluation/
│   │   ├── metrics.py             # Accuracy, NLL, Brier, ECE, MCE
│   │   ├── uncertainty.py         # Descomposición Depeweg/Gal
│   │   ├── selective.py           # Risk-coverage, AURC, umbral óptimo
│   │   ├── ood.py                 # AUROC OOD, FPR@95TPR
│   │   └── business.py            # Análisis coste-beneficio
│   └── utils/
│       ├── seed.py                # Reproducibilidad
│       └── io.py                  # CSV/JSON helpers
│
├── scripts/
│   ├── 01_build_dataset.py        # Genera dataset completo (IMDb + OOD)
│   ├── 02_train_slda.py           # Entrena sLDA amortizado
│   ├── 03_train_all_models.py     # Entrena todos los baselines + MoE
│   ├── 04_evaluate.py             # E1-E7 del plan, genera tablas y figuras
│   └── 05_business_analysis.py    # τ* óptimo y curvas de coste
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploración
│   ├── 02_slda_topics.ipynb       # Visualización de temas supervisados
│   ├── 03_uncertainty_qualitative.ipynb   # Ejemplos cualitativos
│   └── 04_results_figures.ipynb   # Figuras para la memoria
│
├── app/
│   └── streamlit_app.py           # Demo interactiva
│
├── experiments/
│   └── results/                   # CSVs y modelos guardados
│
├── data/
│   ├── processed/                 # Embeddings, TF-IDF, labels, splits
│   └── ood/                       # Muestras OOD
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

O con Docker:

```bash
docker build -t bayesian-sentiment .
docker run --rm -it -v $(pwd):/app bayesian-sentiment bash
```

---

## Ejecución paso a paso

### Paso 1 — Construir el dataset (una vez)

Descarga IMDb + Amazon (libros) + AG-News, limpia, inyecta ruido controlado, extrae embeddings DistilBERT y construye splits.

```bash
python scripts/01_build_dataset.py --imdb-size 20000 --ood-size-per-source 2000
```

Tarda ~15-25 min en CPU, ~3-5 min en GPU. Genera:

- `data/processed/bert_embeddings.npy`
- `data/processed/tfidf_matrix.npz`
- `data/processed/labels.npy`
- `data/processed/splits.json`
- `data/ood/ood_embeddings.npy`
- `data/ood/ood_metadata.json`

### Paso 2 — Entrenar sLDA amortizado

```bash
python scripts/02_train_slda.py --n-topics 10 --epochs 30
```

Genera `experiments/results/slda/` con:
- `lda.pkl` (modelo LDA clásico entrenado)
- `supervised_head.pt` (pesos bayesianos de la cabeza supervisada, formato Pyro)
- `theta_{train,val,test,ood}.npy` (mezclas de temas para todos los splits)
- `topic_words.json` (top palabras por tema)
- `topic_sentiment.json` (correlación tema → sentimiento)

### Paso 3 — Entrenar todos los modelos a comparar

Entrena: determinista, MC-Dropout, BNN base, BNN+LDA concat, BNN-MoE (el nuestro), BNN-MoE+heteroscedástica.

```bash
python scripts/03_train_all_models.py --seeds 42 43 44
```

Guarda cada modelo en `experiments/results/models/<nombre>/seed_<S>/`.

### Paso 4 — Evaluación completa (E1-E7)

```bash
python scripts/04_evaluate.py
```

Genera en `experiments/results/evaluation/`:
- `summary_table.csv` — tabla principal de la memoria
- `risk_coverage.pdf`, `reliability_diagram.pdf`, `ood_auroc.pdf`
- `per_model_metrics.json`

### Paso 5 — Análisis de negocio

```bash
python scripts/05_business_analysis.py \
  --cost-fp 1.0 --cost-fn 5.0 --cost-review 0.2
```

Genera curva E[C(τ)] y umbral óptimo τ*.

### Paso 6 — Demo interactiva

```bash
streamlit run app/streamlit_app.py
```

Abre `http://localhost:8501`: textarea → predicción + descomposición de incertidumbre + decisión de derivación.

---

## Reproducibilidad

Todas las seeds se fijan en `src/utils/seed.py`. Los scripts aceptan `--seed` y `--seeds`. Los resultados del paper se obtienen con seeds `{42, 43, 44}`.

## Licencia

Proyecto académico. Código bajo MIT.
