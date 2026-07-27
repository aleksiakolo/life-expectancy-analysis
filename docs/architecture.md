# Architecture

## Overview

`life-expectancy-analysis` is a reproducible data science pipeline for analysing global life expectancy drivers. It is structured as a proper Python package with a CLI-driven workflow and clean separation between pipeline logic and exploratory notebooks.

---

## 1. Package Structure

**Package name:** `life_expectancy`

```text
life-expectancy-analysis/
│
├── life_expectancy/          # Main package
│   ├── data/                 # Loading, cleaning, standardisation, imputation, panel merging
│   ├── features/             # Feature engineering, selection, temporal (lag) features
│   ├── modeling/             # Pipelines, train/eval, splits, registries
│   │   ├── models/           # baseline, tree, boosting, lstm, neural
│   │   └── experiments/      # core, boosting, sequence, wdi experiment runners
│   ├── analysis/             # Diagnostics and error analysis
│   └── cli.py                # CLI entrypoint (Typer)
│
├── configs/
│   └── default.yaml          # All runtime parameters (data paths, model settings, seeds)
│
├── notebooks/                # Exploration, visualisation, reporting (00–12)
├── tests/                    # Unit tests mirroring package structure
├── data/
│   ├── raw/                  # Source CSVs (WHO, World Bank, WDI) — gitignored
│   ├── interim/              # Intermediate artifacts — gitignored
│   └── processed/            # Final model-ready datasets — gitignored
├── docs/
├── .github/workflows/ci.yml  # GitHub Actions: lint + test
├── pyproject.toml            # Package metadata, deps, tool config
├── Dockerfile
└── Makefile
```

---

## 2. Packaging

- **Build backend:** `setuptools` via `pyproject.toml`
- **Install:** `pip install -e ".[dev,advanced,sequence]"`
- **CLI entrypoint:** `lifeexp = "life_expectancy.cli:main"`
- **Dependency groups:**
  - `dev` — pytest, ruff, mypy, pre-commit, stubs
  - `advanced` — XGBoost, LightGBM, CatBoost, statsmodels
  - `sequence` — PyTorch (LSTM)
  - `interpretability` — shap, scikit-optimize

---

## 3. CLI

**Binary:** `lifeexp`

| Command | Description |
|---|---|
| `lifeexp info` | Show project/config information |
| `lifeexp preprocess` | Clean raw data and build the panel |
| `lifeexp features` | Build feature sets A/B/C from the panel |
| `lifeexp train-baselines` | Train baseline models (mean, Ridge, Lasso, trees) |
| `lifeexp train-advanced` | Train advanced sklearn models (HGB, RF, ExtraTrees, MLP) |
| `lifeexp train-boosting` | Train boosting models (XGBoost, LightGBM, CatBoost) |
| `lifeexp train-wdi` | Train lag models on the WDI feature set |
| `lifeexp tune` | Bayesian hyperparameter search (scikit-optimize) for one model |
| `lifeexp interpret` | SHAP + permutation feature importance for one model |
| `lifeexp all` | Run the core reproducible end-to-end workflow |

All commands accept `--config-path` (defaults to `configs/default.yaml`). The
`tune` and `interpret` commands additionally accept `--model-name` and
`--feature-set`, and read the `tuning` / `interpretability` config sections.

---

## 4. Configuration

- Single YAML file: `configs/default.yaml`
- Covers: data source paths, panel merging, imputation, cleaning, feature engineering, model splits, hyperparameters, random seeds
- Loaded at CLI startup and threaded through all downstream functions — no hardcoded values in pipeline code

---

## 5. Data Layer (`life_expectancy/data/`)

| Module | Responsibility |
|---|---|
| `loading.py` | CSV and WDI file ingestion |
| `cleaning.py` | WHO and World Bank schema cleaning |
| `standardization.py` | Column normalisation across sources |
| `preprocessing.py` | Panel construction and full preprocessing |
| `imputation.py` | Missing value strategies |
| `missingness.py` | Missingness analysis utilities |
| `panel.py` | Cross-source panel merging |
| `wdi.py` | World Development Indicators reshaping |
| `utils.py` | Shared IO helpers |

---

## 6. Feature Engineering (`life_expectancy/features/`)

| Module | Responsibility |
|---|---|
| `feature_engineering.py` | Log transforms, interaction terms, status flags |
| `feature_selection.py` | Correlation, VIF, manual selection; builds feature-set A/B/C |
| `temporal.py` | Country-level lag feature generation |

---

## 7. Modeling (`life_expectancy/modeling/`)

| Module | Responsibility |
|---|---|
| `pipelines.py` | scikit-learn preprocessing pipelines |
| `train_eval.py` | Training loop and evaluation metrics |
| `splits.py` | Time-aware train/val/test splitting |
| `registries.py` | Model registry (named model instances) |
| `models/baselines.py` | Ridge, Lasso, ElasticNet |
| `models/tree.py` | HistGradientBoosting, RandomForest, ExtraTrees |
| `models/boosting.py` | XGBoost, LightGBM, CatBoost wrappers |
| `models/lstm.py` | LSTM regression (PyTorch) |
| `models/neural.py` | MLP regression |
| `experiments/core.py` | Time-split experiment framework |
| `experiments/boosting.py` | Boosting experiment runner |
| `experiments/sequence.py` | LSTM sequence experiment runner |
| `experiments/wdi.py` | WDI panel experiment runner |
| `tuning.py` | Bayesian hyperparameter search (BayesSearchCV + predefined search spaces) |

---

## 8. Analysis (`life_expectancy/analysis/`)

| Module | Responsibility |
|---|---|
| `diagnostics.py` | Residual analysis, country-level error summaries, diagnostic plots |
| `interpretability.py` | SHAP feature importance, permutation importance, importance bar plots |

---

## 9. Notebooks

Notebooks are for exploration, visualisation, and reporting only. All reusable logic lives in the package.

| Range | Purpose |
|---|---|
| 00–02 | Data loading, exploration, cleaning |
| 03–05 | Descriptive, comparative, correlation analysis |
| 06–09 | Baseline modeling, diagnostics, visualisation, advanced models |
| 10–12 | Temporal features, boosting + LSTM, WDI modeling |
| 13 | Bayesian tuning + SHAP/permutation interpretability |

---

## 10. Testing

```text
tests/
├── data/          # Unit tests for data layer
├── features/      # Unit tests for feature engineering and selection
└── modeling/      # Unit tests for splits, pipelines, evaluation
```

Run with: `pytest`

---

## 11. CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
- **Lint job:** `ruff format --check` + `ruff check`
- **Test job:** `pip install -e ".[dev,advanced]"` → `pytest`
- Triggers on push and pull request to `main`

---

## 12. Dependency Management

- Source of truth: `pyproject.toml`
- `requirements.txt` installs all extras via `-e ".[dev,advanced,sequence]"`
- For reproducible environments, generate a lock file with `uv pip compile pyproject.toml -o requirements.lock`

---

## 13. Docker

```bash
make build   # builds image
make run     # mounts data/ and runs lifeexp
make panel   # runs lifeexp preprocess
```

Entrypoint: `lifeexp` (defaults to `--help`)
