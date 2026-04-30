# Methodology

This document describes the **planned** methodology for the Life Expectancy Analysis project.
Sections will be updated with **final thresholds, dropped columns, and results** once Week 2 (cleaning/merge) and Week 3 (EDA/correlation) outputs are generated.

---

## Overview

The project builds a country–year panel dataset by combining:
- **WHO** life expectancy + health indicators (`data/raw/world_health_organization/who.csv`)
- **World Bank panel** socio-economic indicators (`data/raw/world_bank/wb.csv`)
- (Optional extension) **World Bank WDI export** (`data/raw/world_bank/wdi.csv`) for validation and additional features

The workflow is:

1. Data understanding + schema confirmation (Week 1)
2. Cleaning, harmonization, merge, and feature engineering (Week 2)
3. Advanced EDA, comparative analysis, and correlation study (Week 3)
4. Modeling, insights, reporting, and presentation (Week 4+)

---

## Data Preprocessing Plan

### 1) Standardization and Schema Alignment
**Goal:** ensure both datasets share consistent merge keys and interpretable column names.

Planned actions:
- Standardize column names to snake_case.
- Trim whitespace in raw WHO column headers (e.g., `Life expectancy ` → `life_expectancy`).
- Standardize country naming rules (strip whitespace, normalize punctuation/casing).
- Normalize `year` to integer and remove invalid/non-numeric years with explicit logging.

Artifacts:
- `data/external/country_name_map.csv` (manual mapping for country name mismatches)
- Notebook logs + summary tables in `docs/tables/`

---

### 2) Uniqueness and Duplicate Handling
**Goal:** enforce one row per `(country, year)`.

Checks:
- WHO: validate uniqueness on `(Country, Year)` and inspect duplicates.
- WB panel: validate uniqueness on `(Country Name, Year)`.

Decision rule (to finalize after inspection):
- If duplicates exist, choose one of:
  - keep-first (if exact duplicates),
  - aggregate (if duplicates differ only in a small subset of numeric fields),
  - drop (if clearly corrupted / inconsistent).

All decisions will be recorded with:
- counts removed/aggregated
- example rows
- justification

---

### 3) Plausibility and Range Checks
**Goal:** remove or correct values that are clearly invalid.

Planned checks:
- `life_expectancy`: flag values `<= 0` or `> 100` and decide drop vs clip.
- Percentage-like indicators (e.g., immunization rates): enforce valid bounds where appropriate (often `[0, 100]`).
- Mortality and expenditure fields: flag negative values and investigate.

All corrections will be:
- logged (how many values corrected, how many rows dropped)
- justified (why this threshold/rule is reasonable)

---

## Missing Value Diagnostics and Handling

### Diagnostics to Compute
To understand missingness structure before any imputation:
- % missing per column
- % missing per year (at least for target + key predictors)
- % missing per country (at least for target)
- Optional missingness heatmap (diagnostic only)

Artifacts:
- tables exported to `docs/tables/`
- optional figure(s) exported to `reports/figures/`

### Predefined Handling Rules (Subject to Final Threshold Selection)
**Target variable**
- Rows with missing `life_expectancy` are always dropped.

**Feature-level filtering**
- Drop features above a missingness threshold in the **40–60% band**.
- Final cutoff will be chosen after inspecting:
  - importance of the feature group
  - whether missingness is concentrated in specific periods/regions
  - merge-induced missingness post-join

**Country-level filtering**
- Drop countries with fewer than **5 observed years** of `life_expectancy` (minimum coverage rule).

**Imputation**
- Numeric predictors with moderate missingness: median imputation (global median) as baseline.
- Optional grouped imputation:
  - region median
  - income group median
  applied only when group labels exist and the variable is suitable.
- No blanket interpolation; see time-series rules below.

---

## Time-Series Gaps and Interpolation Policy

Not all variables should be interpolated.

**Interpolation allowed only when**
- gaps are short (1–2 years)
- indicator is slow-moving (e.g., life expectancy, schooling)

**Interpolation not allowed by default**
- variables prone to shocks/spikes (e.g., mortality-related indicators)

If interpolation is applied:
- record which columns were interpolated
- record how many values were filled
- keep traceability (before/after counts)

---

## Feature Transformations and Scaling

### Transformations
Some variables are expected to be heavy-tailed (e.g., GDP-like measures, population).
Planned options:
- log transforms for heavy-tailed predictors
  `x_log = log(x + c)` with small constant `c` if zeros exist.
- keep both original and transformed versions for interpretability.

### Scaling (Modeling Stage)
Scaling is applied **after** train/test split.

For predictor columns used in regression / ML models:

$$
x' = \frac{x - \mu}{\sigma}
$$

Rules:
- Fit scalers on training data only.
- Apply the same transform to validation/test data.

---

## Merge Strategy

### Target Structure
A clean country–year panel with:
- **Key:** `(country, year)`
- one row per country–year

### Join Logic
- Base merge: WHO cleaned ↔ WB panel cleaned on `(country, year)`
- Start with **inner join** for initial modeling reliability.
- Measure and report coverage loss:
  - rows lost vs WHO
  - rows lost vs WB
  - which countries/years are lost due to mismatches

Artifacts:
- `docs/tables/merge_coverage_loss.csv`

### Country Name Harmonization
Planned approach:
- mapping file: `data/external/country_name_map.csv`
- if mapping remains large, evaluate ISO3 alignment using WB `Country Code` as stable identifier

---

## Exploratory Data Analysis Plan

EDA is used to:
- validate distributions and coverage
- identify anomalies/outliers
- motivate modeling choices
- structure reporting around interpretable feature groups

### 1) Target Variable: Life Expectancy
Key questions:
- What is the global distribution of life expectancy?
- How does global life expectancy change over time?
- How does life expectancy differ by region/income group?

Planned plots:
- histogram/density of life expectancy
- global mean/median life expectancy over time
- boxplots by region/income group
- variability over time (variance or IQR by year)

### 2) Health System and Immunization
Planned focus:
- association between immunization coverage and life expectancy
- coverage and missingness patterns by country/year

Planned plots:
- scatter: immunization vs life expectancy
- time trends (global + by region)
- missingness tables for key health variables

### 3) Mortality and Disease Burden
Planned focus:
- inverse relationships with life expectancy
- nonlinear effects and outliers

Planned plots:
- scatter (possibly log-scale): mortality vs life expectancy
- correlation heatmap for mortality-related indicators
- example country trajectories for mortality + life expectancy

### 4) Economic Indicators
Planned focus:
- diminishing returns of income on life expectancy
- differences across income groups

Planned plots:
- life expectancy vs log(GDP-related) predictors
- time trends by income group

### 5) Social and Infrastructure
Planned focus:
- threshold effects
- bias from uneven coverage

Planned plots:
- life expectancy vs sanitation
- regional comparisons

### 6) Education and Demographics
Planned focus:
- structural demographic relationships
- long-run transitions

Planned plots:
- scatter with smoothing (e.g., LOWESS) where appropriate
- time series by region/income group

### 7) Multivariate Structure
Planned tools:
- correlation matrix (Pearson + Spearman)
- multicollinearity checks (VIF) on selected predictor subset
- PCA as descriptive structure (not required for final modeling)

---

## Correlation and Statistical Analysis Plan

Planned steps:
- compute Pearson and Spearman correlations with `life_expectancy`
- rank predictors by absolute correlation strength
- check stability across:
  - years (e.g., decade slices)
  - regions/income groups
- investigate multicollinearity among top predictors (VIF + correlation clustering)

Artifacts:
- `docs/tables/top_correlated_features.csv`
- correlation heatmap figure(s) in `reports/figures/`

---

## Predictive Modeling Plan

Baseline models:
- linear regression (baseline)
- Ridge / Lasso (regularized)

Modeling rules:
- train/test split or time-aware split (to decide based on panel structure)
- cross-validation where appropriate
- evaluate with:
  - RMSE
  - R²

Interpretability focus:
- keep a strong emphasis on explainability
- interpret results with public health / policy relevance
- report limitations (missingness, coverage bias, indicator definitions)

---
