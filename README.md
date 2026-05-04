# Life Expectancy Analysis

A reproducible, end-to-end machine learning pipeline for analyzing global life expectancy using WHO, World Bank, and WDI data.
The project supports data preprocessing, feature engineering, multiple model families, and experiment tracking through a unified CLI.

## Project Objective
This project builds a **country–year panel dataset** and trains models to predict life expectancy.

It includes:

- Data pipeline (WHO + World Bank + WDI)
- Feature engineering (A/B/C feature sets, lag features)
- Time-aware train/validation/test splits
- Multiple model families:
  - Baselines (Ridge, Lasso, etc.)
  - Tree models (Random Forest, Extra Trees, HistGBR)
  - External boosting (XGBoost, LightGBM, CatBoost)
  - WDI-based long-horizon models
- CLI for reproducible experimentation
- Docker support for environment reproducibility


**Explanation**
- Which factors are most strongly associated with life expectancy?
- How do health, economic, and social indicators interact?

**Prediction**
- How accurately can life expectancy be estimated from available indicators?
- Which variables contribute most to predictive performance?

---

## Data Sources

The analysis combines two international public datasets.

### 1) World Health Organization (WHO)
Contains health and demographic indicators.

Examples of variables:
- life expectancy at birth (target variable)
- adult mortality
- infant deaths
- immunization coverage
- disease indicators (e.g., HIV/AIDS)
- schooling
- BMI

File location: data/raw/world_health_organization/who.csv


Schema characteristics:
- One row per `(Country, Year)`
- Health-focused variables
- Main source of the **target variable**

---

### 2) World Bank Socio-Economic Panel
Contains economic and infrastructure indicators.

Examples of variables:
- sanitation access
- unemployment
- education expenditure
- CO₂ emissions
- undernourishment
- health expenditure
- income group and region

File location: data/raw/world_bank/wb.csv


Schema characteristics:
- One row per `(Country Name, Year)`
- Socio-economic predictors

---

### Comparison
World Bank WDI export: data/raw/world_bank/wdi.csv

---

## Project Structure

```text
life_expectancy/
  cli.py                 # Main CLI entrypoint
  data/                  # Data loading, cleaning, preprocessing
  features/              # Feature engineering and selection
  modeling/
    model/               # Model definitions (baselines, tree, etc.)
    experiments/         # Training + evaluation logic
    registries.py        # Model registry
    splits.py            # Time-based splits
    train_eval.py        # Evaluation utilities
````

Other folders:

```text
configs/                 # YAML config files
data/                    # Raw + processed data
reports/tables/          # Output metrics and results
reports/figures/         # Plots
notebooks/               # Analysis notebooks
tests/                   # Unit tests
```

---

## Reproducibility

This project is designed to be **fully reproducible**:

### 1. Config-driven

All experiments are controlled via:

```text
configs/default.yaml
```

This defines:

* data sources
* preprocessing logic
* model configurations
* splits (time-aware)

---

### 2. CLI-based pipeline

All major steps can be run via CLI:

```bash
lifeexp preprocess
lifeexp features
lifeexp train-baselines
lifeexp train-advanced
lifeexp train-boosting
lifeexp train-wdi
lifeexp all
```

The `all` command runs the full pipeline end-to-end.

---

### 3. Deterministic splits

* Time-aware splitting (no leakage)
* Fixed random seeds
* Consistent feature generation

---

### 4. Docker support

You can run the entire pipeline in a clean environment.

#### Build image:

```bash
docker build -t life-exp .
```

#### Run CLI:

```bash
docker run --rm life-exp --help
```

#### Run full pipeline:

```bash
docker run --rm -v "$(pwd)":/app life-exp all
```

---

## Installation

### Local (editable install)

```bash
pip install -e ".[dev]"
```

---

## Data Pipeline

1. Load raw data (WHO, World Bank, optional WDI)
2. Standardize schemas
3. Clean datasets
4. Merge into panel (country, year)
5. Add target (`life_expectancy`)
6. Impute missing values

---

## Modeling

### Feature Sets

* **A**: Minimal features
* **B**: Core features
* **C**: Full feature set

---

### Models

#### Baselines

* RidgeCV
* LassoCV

#### Tree Models

* HistGradientBoosting
* RandomForest
* ExtraTrees

#### External Boosting

* XGBoost
* LightGBM
* CatBoost

#### WDI Models

* Long time horizon
* Lag features
* Panel-aligned evaluation

---

## Outputs

Results are saved to:

```text
reports/tables/
```

Examples:

* `baseline_results.csv`
* `advanced_model_comparison.csv`
* `external_boosting_compare.csv`
* `wdi_panel_overlap_model_compare.csv`

---

## Testing

```bash
pytest
```

---

## Notes

* The pipeline uses **time-aware splits** to avoid leakage.
* WDI models extend the dataset across a longer time horizon.
* External boosting models are optional (installed via `[advanced]` dependencies).
* Some warnings (e.g., LightGBM feature names) are expected and non-breaking.

---

## Future Improvements (optional)

* Model artifact saving (pickle / joblib)
* Experiment tracking (MLflow or lightweight logging)
* Hyperparameter tuning automation
* Additional sequence models (LSTM)
