# Bayesian Tuning & Interpretability — Change Report

**Date:** 2026-06-26
**Scope:** Life Expectancy project only (the Customer Churn / LLM items from the
worklog are out of scope here).

This report documents the work to finish the Life Expectancy modeling tasks logged
on 05/09–05/10:

> *Add Bayesian hyperparameter optimization (scikit-optimize) over the previous
> grid-search workflow; integrate it into the Life Expectancy modeling workflow; add
> SHAP-based explanations; add permutation importance; connect interpretability
> outputs to the existing evaluation/visualization workflows; update the
> modeling/report components.*

---

## 1 · Starting state vs. what was actually done

Two modules had been **created** but were **orphaned, untested, and partially broken**:

| Area | Found | Status |
|---|---|---|
| `life_expectancy/modeling/tuning.py` | `BayesSearchCV` wrapper + search spaces | Present, but **crashed on import** and had a **parameter bug** |
| `life_expectancy/analysis/interpretability.py` | SHAP + permutation helpers + plots | Present and correct, but **not wired into anything** |
| `configs/default.yaml` | `tuning` / `interpretability` sections | Present |
| `pyproject.toml` | `interpretability` extra (shap, scikit-optimize) | Present |
| `docs/architecture.md` | module tables | Present but CLI table was stale |
| Tests | — | **None** for either module |
| CLI / notebook integration | — | **None** — nothing called either module |

So the timesheet claims ("integrated into the modeling workflow", "connected to the
evaluation/visualization workflows") were **not yet true**. The work below makes them
true and fixes the defects.

---

## 2 · What changed, how, and why

### 2.1 Fixed a parameter bug that made Bayesian tuning unusable (`tuning.py`)

**What.** The default search spaces tune `model__alpha` (Ridge), `model__alpha` +
`model__l1_ratio` (ElasticNet), etc. But the model registry maps `ridge`/`lasso`/
`elasticnet` to the **cross-validated** estimators `RidgeCV`/`LassoCV`/`ElasticNetCV`,
which expose `alphas` (plural) and have **no tunable `alpha`**. Running a search on a
registry pipeline therefore raised `ValueError: Invalid parameter 'alpha' for
estimator RidgeCV`.

**How.** Added `get_tunable_estimator(model_name)`, returning **base** estimators
(`Ridge`, `Lasso`, `ElasticNet`, plus the tree/boosting factories) whose
hyperparameters line up exactly with the search spaces. Added a regression test
(`test_search_space_params_match_tunable_estimator`) asserting every search-space key
is a real parameter of the matching estimator.

**Why.** Bayesian search *replaces* the estimator's internal CV selection, so it needs
the plain estimator, not the `*CV` variant. This is the core correctness fix.

### 2.2 Made the module import-safe (`tuning.py`)

**What.** `tuning.py` did `from skopt import BayesSearchCV` at module top with a
`raise ImportError`, so simply importing the module failed when `scikit-optimize`
was absent.

**How.** Moved all `skopt` imports inside the functions that use them and build the
search-space dimension objects lazily (`_build_default_search_spaces`). A module-level
`AVAILABLE_MODELS` tuple lets callers enumerate/validate models without importing
`skopt`.

**Why.** CI installs only `[dev,advanced]` (no interpretability extras). With the old
top-level import, any test that imported the module would have errored the whole test
job. This matches the lazy-import pattern already used in `models/boosting.py` and
`analysis/interpretability.py`.

### 2.3 Added integration glue (`tuning.py`)

* `build_tunable_pipeline(model_name, x_train, …)` — builds the `prep` + `model`
  pipeline with the correct estimator and recommended scaling in one call.
* `tune_model(model_name, x_train, y_train, …)` — pairs the pipeline with its search
  space and runs the search, so the two can never drift apart.
* `run_bayes_search_from_config(...)` — reads the `tuning:` config block.

### 2.4 Wired both capabilities into the CLI (`cli.py`)

Two new commands integrate tuning and interpretability into the **same time-aware
workflow** (panel → feature sets A/B/C → last-3-years temporal holdout) used by the
existing `train-*` commands:

```bash
lifeexp tune       --model-name hgb --feature-set B
lifeexp interpret  --model-name hgb --feature-set B
```

* `tune` runs Bayesian search, evaluates the tuned model **and an untuned default**
  on the temporal holdout, and writes: all evaluated configs
  (`bayes_tuning_results.csv`), best params + metrics (`bayes_best_params.json`), and
  an appended default-vs-tuned log (`bayes_tuning_compare.csv`).
* `interpret` trains the model and writes SHAP + permutation importance tables to
  `reports/tables/` and bar charts to `reports/figures/` — i.e. interpretability
  outputs now flow into the existing reporting locations.

A shared `prepare_feature_split()` helper guarantees tuning, interpretability, and
training all use identical inputs.

### 2.5 Tests (CI-safe)

* `tests/modeling/test_tuning.py` — 10 tests. Pure-Python helpers (estimator factory,
  the bug regression test, scaling, pipeline shape) run without `skopt`; the
  search-execution tests use `pytest.importorskip("skopt")`.
* `tests/analysis/test_interpretability.py` — 5 tests. Permutation importance and
  plots run with scikit-learn/matplotlib only; the SHAP test uses
  `pytest.importorskip("shap")`.

Verified that with `skopt` and `shap` blocked (simulating the CI `[dev,advanced]`
env) the suite reports **11 passed, 4 skipped, 0 errors**.

### 2.6 Docs

`docs/architecture.md`: replaced the stale CLI table (it listed non-existent
`train` / `pipeline` commands) with the real command set including `tune` and
`interpret`, and added notebook 13 to the notebook index.

---

## 3 · Results (from `notebooks/13_tuning_and_interpretability.ipynb`)

Feature set **B** (31 features), temporal holdout = **2013–2015** (1,812 train /
453 test rows), `n_iter = 16`, 5-fold CV.

### 3.1 Default vs. Bayesian-tuned — test RMSE (lower is better)

| Model | Default RMSE | Tuned RMSE | Δ (improvement) | Default R² | Tuned R² |
|---|---|---|---|---|---|
| ridge | 3.1357 | 3.1342 | **+0.0015** | 0.8597 | 0.8598 |
| hgb | 1.5905 | 1.6442 | **−0.0537** | 0.9639 | 0.9614 |
| rf | 1.5973 | 1.6197 | **−0.0224** | 0.9636 | 0.9626 |

### 3.2 Feature importance (tuned HGB) — SHAP and permutation agree

| Rank | SHAP (mean \|value\|) | Permutation (Δ RMSE) |
|---|---|---|
| 1 | hiv_aids | income_composition_of_resources |
| 2 | income_composition_of_resources | hiv_aids |
| 3 | adult_mortality | adult_mortality |
| 4 | infant_deaths | infant_deaths |
| 5 | year | thinness_1_19_years |

Figures: `reports/figures/week13_tuning_default_vs_tuned_rmse.png`,
`week13_hgb_search_convergence.png`, `week13_hgb_importance.png`.

### 3.3 Key finding (honest, and the most useful part)

The tuning machinery works and reliably drives the **cross-validated** objective down,
**but on the out-of-time holdout the tuned tree models do not beat their defaults.**
The search optimises a *random* 5-fold CV on the training years and selects more
expressive configurations (e.g. tuned HGB: `max_depth=8`, `max_iter=500`,
`learning_rate=0.14`) that overfit the in-period folds relative to the time-shifted
test years. The library defaults are already well-regularised. `ridge` is essentially
unchanged, consistent with the existing `RidgeCV` already finding a good `alpha`.

**Recommendation / next step:** run `BayesSearchCV` with a **time-series CV** splitter
so the search objective matches the project's temporal evaluation. The interpretability
outputs, by contrast, are immediately useful and method-agreement gives confidence in
the identified drivers.

---

## 4 · How to run

```bash
pip install -e ".[dev,advanced,interpretability]"

# Bayesian tuning (writes results/best-params/compare under reports/tables/)
lifeexp tune --model-name hgb --feature-set B

# Interpretability (writes tables + figures under reports/)
lifeexp interpret --model-name hgb --feature-set B

# Full visual walkthrough
jupyter notebook notebooks/13_tuning_and_interpretability.ipynb
```

Search depth and importance settings are configured in `configs/default.yaml`
(`tuning:` and `interpretability:` blocks).

---

## 5 · Files added / changed

**Added**
- `notebooks/13_tuning_and_interpretability.ipynb` — executed, with figures
- `tests/modeling/test_tuning.py`, `tests/analysis/test_interpretability.py`
  (+ `tests/analysis/__init__.py`)
- `docs/interpretability_and_tuning_report.md` (this file)
- Generated artifacts (under the gitignored `reports/`, regenerated by the notebook
  and the `lifeexp tune` / `lifeexp interpret` commands): `reports/figures/week13_*.png`,
  `reports/tables/week13_*.csv`

**Changed**
- `life_expectancy/modeling/tuning.py` — bug fix, lazy imports, tunable estimators, glue
- `life_expectancy/cli.py` — `tune` and `interpret` commands + shared split helper
- `docs/architecture.md` — corrected CLI table, added notebook 13

**Already present (kept):** `life_expectancy/analysis/interpretability.py`,
`configs/default.yaml` (`tuning`/`interpretability`), `pyproject.toml`
(`interpretability` extra).
