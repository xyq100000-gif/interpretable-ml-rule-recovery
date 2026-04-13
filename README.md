# From Classification to Rule Recovery: An Interpretable ML Pipeline

This repository refactors a coursework submission into a compact, research-oriented case study on **interpretable structure discovery from synthetic tabular data**. Rather than treating the task as a standard prediction exercise, the project asks whether hidden decision rules can be recovered with simple, transparent models.

The pipeline is organised in three stages:

1. **Clan inference** using sparse feature selection and k-nearest neighbours.
2. **Within-clan rank rule discovery** using single-feature threshold rules.
3. **Prowess formula recovery** using sparse linear search with integer-constrained interpretability.

The emphasis is not on achieving the highest possible predictive score, but on showing how a sequence of lightweight models can reconstruct latent structure in a controlled setting.

## Why this project matters

Many machine learning portfolios are dominated by black-box prediction tasks. This project is different in two ways:

- it prioritises **recoverability and interpretability** over model complexity;
- it treats modelling as a process of **reverse-engineering latent rules**, not just fitting outputs.

That makes it a useful small-scale example of research training in interpretable modelling, feature selection, structured prediction, and sparse model search.

## Main results

Using the refactored pipeline on the provided synthetic dataset:

- the clan classifier selected **`avoidance`** and **`peril`** as the most discriminative pair of features;
- repeated stratified 5-fold cross-validation selected **`k = 9`** for KNN, with mean accuracy **0.9416**;
- the fitted clan classifier achieved **0.948** training accuracy;
- per-clan threshold rules achieved **perfect training separation (1.000)** for rank prediction;
- the sparse prowess model recovered the formula:

```text
prowess = -27796 + 6*charring + 887*sliminess
```

with training **R² = 0.9993**.

## Repository structure

```text
interpretable-ml-rule-recovery/
├── README.md
├── requirements.txt
├── data/
│   └── raw/machlearnia.csv
├── notebooks/
│   └── refactored_analysis.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── interpreterule/
│       ├── clan_classifier.py
│       ├── rank_rules.py
│       ├── prowess.py
│       ├── tournament.py
│       └── pipeline.py
├── outputs/
│   ├── metrics.json
│   ├── rank_rules.csv
│   ├── prowess_formula.csv
│   └── figures/
├── tests/
│   └── test_reproducibility.py
└── docs/
    ├── portfolio_positioning.md
    └── cv_project_entry.md
```

## Methods overview

### 1. Clan classification

A Fisher-score screening step identifies the most discriminative pair of features. A hand-implemented KNN classifier is then tuned using repeated stratified 5-fold cross-validation. The final model also reports vote-share confidence on test predictions.

### 2. Rank rule learning

For each clan, the code searches for a single explanatory variable and two thresholds that split rank into three ordered groups. This produces simple rules that are directly interpretable and easy to audit.

### 3. Prowess formula recovery

Candidate subsets of explanatory variables are evaluated using ordinary least squares and cross-validated MAE. The final selected formula is rounded to integer coefficients to favour transparent reconstruction over purely numeric optimisation.

## How to run

Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

This produces predictions, summary metrics, recovered rules, the recovered prowess formula, and diagnostic figures in `outputs/`.

To inspect the analysis in notebook form, open:

```text
notebooks/refactored_analysis.ipynb
```

## What is intentionally not claimed

This is a **synthetic** and highly controlled setting. The project does **not** claim robustness on noisy real-world data, nor does it present a novel algorithm. Its value lies in careful problem decomposition and transparent model design.

In particular:

- the hidden rules are recoverable by design;
- performance is reported in a low-noise setting;
- interpretability is prioritised over flexible function approximation;
- the project should be read as a **method-focused portfolio case study**, not as a research contribution in itself.

## Possible extensions

Natural next steps would include:

- replacing pairwise Fisher screening with sparse discriminant analysis or penalised multinomial models;
- evaluating robustness under injected noise and feature corruption;
- formalising the formula search as symbolic regression or constrained model selection;
- comparing threshold-rule recovery with decision trees or ordinal classification baselines.

## Portfolio positioning

This project is best presented as a compact example of:

- interpretable machine learning;
- structured tabular prediction;
- sparse model search;
- reverse-engineering latent decision rules.

It is suitable as a **supporting GitHub project** for applications in statistical machine learning, interpretable ML, or probabilistic modelling.
