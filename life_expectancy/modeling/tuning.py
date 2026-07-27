from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from life_expectancy.modeling.models.tree import (
    extra_trees_regressor,
    hist_gradient_boosting_regressor,
    random_forest_regressor,
)
from life_expectancy.modeling.pipelines import (
    ScaleMode,
    build_model_pipeline,
    build_preprocessor,
    infer_feature_types,
)

SearchSpace = dict[str, Any]

# Model names with a defined Bayesian search space. Listed here (rather than as
# keys of the space dict) so the module imports without scikit-optimize, which is
# an optional `interpretability` extra and is not installed in CI.
AVAILABLE_MODELS: tuple[str, ...] = (
    "ridge",
    "lasso",
    "elasticnet",
    "hgb",
    "rf",
    "extra_trees",
    "xgb",
    "lgbm",
)

# Recommended numeric scaling per tunable model (mirrors the model registry).
TUNABLE_SCALE_NUMERIC: dict[str, ScaleMode] = {
    "ridge": "standard",
    "lasso": "standard",
    "elasticnet": "standard",
    "hgb": "none",
    "rf": "none",
    "extra_trees": "none",
    "xgb": "none",
    "lgbm": "none",
}


def _build_default_search_spaces() -> dict[str, SearchSpace]:
    """Construct the default Bayesian search spaces.

    The skopt dimension objects are built lazily so importing this module does
    not require scikit-optimize. Parameter names are prefixed with ``model__``
    because the tuned pipeline's estimator step is named ``model`` (pipelines.py).

    Returns:
        Mapping of model name to a BayesSearchCV-compatible search space.
    """
    from skopt.space import Categorical, Integer, Real

    return {
        "ridge": {
            "model__alpha": Real(1e-2, 1e3, prior="log-uniform"),
        },
        "lasso": {
            "model__alpha": Real(1e-4, 1e2, prior="log-uniform"),
        },
        "elasticnet": {
            "model__alpha": Real(1e-4, 1e2, prior="log-uniform"),
            "model__l1_ratio": Real(0.05, 0.95),
        },
        "hgb": {
            "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "model__max_depth": Integer(3, 8),
            "model__min_samples_leaf": Integer(10, 50),
            "model__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
            "model__max_iter": Integer(100, 500),
        },
        "rf": {
            "model__n_estimators": Integer(50, 300),
            "model__max_depth": Integer(3, 15),
            "model__min_samples_leaf": Integer(2, 20),
            "model__max_features": Categorical(["sqrt", "log2"]),
        },
        "extra_trees": {
            "model__n_estimators": Integer(50, 300),
            "model__max_depth": Integer(3, 15),
            "model__min_samples_leaf": Integer(2, 20),
        },
        "xgb": {
            "model__n_estimators": Integer(100, 500),
            "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "model__max_depth": Integer(3, 8),
            "model__subsample": Real(0.6, 1.0),
            "model__colsample_bytree": Real(0.6, 1.0),
            "model__reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
        },
        "lgbm": {
            "model__n_estimators": Integer(100, 500),
            "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "model__num_leaves": Integer(15, 63),
            "model__subsample": Real(0.6, 1.0),
            "model__colsample_bytree": Real(0.6, 1.0),
            "model__reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
        },
    }


def get_search_space(model_name: str) -> SearchSpace:
    """Return the default Bayesian search space for a named model.

    Args:
        model_name: Model name key (e.g. "ridge", "hgb", "xgb").

    Returns:
        Search space dictionary compatible with BayesSearchCV.

    Raises:
        KeyError: If no default space is defined for the model name.
    """
    if model_name not in AVAILABLE_MODELS:
        raise KeyError(
            f"No search space defined for {model_name!r}. "
            f"Available: {sorted(AVAILABLE_MODELS)}"
        )
    return _build_default_search_spaces()[model_name]


def get_tunable_estimator(model_name: str, *, random_state: int = 42) -> Any:
    """Return a base estimator whose parameters match the model's search space.

    The default model registry uses cross-validated variants (``RidgeCV``,
    ``LassoCV``, ``ElasticNetCV``) that select regularization internally and do
    **not** expose a tunable ``alpha`` parameter. Bayesian search replaces that
    internal selection, so it needs the plain estimators (``Ridge``, ``Lasso``,
    ``ElasticNet``) whose hyperparameters line up with the search spaces above.

    Args:
        model_name: Model name key (see :data:`AVAILABLE_MODELS`).
        random_state: Random seed for stochastic estimators.

    Returns:
        Unfitted scikit-learn compatible estimator.

    Raises:
        KeyError: If the model name has no tunable estimator.
    """
    if model_name == "ridge":
        return Ridge()
    if model_name == "lasso":
        return Lasso(max_iter=5000, random_state=random_state)
    if model_name == "elasticnet":
        return ElasticNet(max_iter=5000, random_state=random_state)
    if model_name == "hgb":
        return hist_gradient_boosting_regressor(random_state=random_state)
    if model_name == "rf":
        return random_forest_regressor(random_state=random_state)
    if model_name == "extra_trees":
        return extra_trees_regressor(random_state=random_state)
    if model_name == "xgb":
        from life_expectancy.modeling.models.boosting import xgb_regressor

        # No early_stopping_rounds: BayesSearchCV.fit does not pass an eval_set.
        return xgb_regressor(
            {"random_state": random_state, "early_stopping_rounds": None}
        )
    if model_name == "lgbm":
        from life_expectancy.modeling.models.boosting import lgbm_regressor

        return lgbm_regressor({"random_state": random_state})

    raise KeyError(
        f"No tunable estimator for {model_name!r}. "
        f"Available: {sorted(AVAILABLE_MODELS)}"
    )


def get_tunable_scale_numeric(model_name: str) -> ScaleMode:
    """Return the recommended numeric scaling mode for a tunable model.

    Args:
        model_name: Model name key.

    Returns:
        Scaling mode string ("standard" or "none").
    """
    return TUNABLE_SCALE_NUMERIC.get(model_name, "standard")


def run_bayes_search(
    pipeline: Pipeline,
    search_space: SearchSpace,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 32,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    n_jobs: int = 1,
) -> Any:
    """Run Bayesian hyperparameter search over a pipeline.

    Args:
        pipeline: Scikit-learn Pipeline with a final step named "model".
        search_space: Parameter search space mapping param names to skopt dimensions.
        x_train: Training features.
        y_train: Training target.
        n_iter: Number of Bayesian search iterations.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for cross-validation.
        random_state: Random seed for reproducibility.
        n_jobs: Parallel jobs (-1 to use all cores).

    Returns:
        Fitted BayesSearchCV. Access best params via `.best_params_` and the
        best pipeline via `.best_estimator_`.
    """
    from skopt import BayesSearchCV

    searcher = BayesSearchCV(
        pipeline,
        search_space,
        n_iter=n_iter,
        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    searcher.fit(x_train, y_train)
    return searcher


def build_tunable_pipeline(
    model_name: str,
    x_train: pd.DataFrame,
    *,
    random_state: int = 42,
    scale_numeric: ScaleMode | bool | None = None,
) -> Pipeline:
    """Build a preprocessing-plus-estimator pipeline ready for Bayesian search.

    Args:
        model_name: Model name key (see :data:`AVAILABLE_MODELS`).
        x_train: Training features (used to infer numeric/categorical columns).
        random_state: Random seed for the estimator.
        scale_numeric: Override the recommended numeric scaling mode.

    Returns:
        Unfitted scikit-learn Pipeline with "prep" and "model" steps.
    """
    if scale_numeric is None:
        scale_numeric = get_tunable_scale_numeric(model_name)

    numeric_cols, categorical_cols = infer_feature_types(x_train)
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=scale_numeric,
    )
    estimator = get_tunable_estimator(model_name, random_state=random_state)

    return build_model_pipeline(model=estimator, preprocessor=preprocessor)


def bayes_search_results(searcher: Any) -> pd.DataFrame:
    """Extract CV results from a fitted BayesSearchCV into a DataFrame.

    Args:
        searcher: Fitted BayesSearchCV object.

    Returns:
        DataFrame of all evaluated configurations, sorted by mean test score descending.
    """
    results = pd.DataFrame(searcher.cv_results_)
    return results.sort_values("mean_test_score", ascending=False).reset_index(
        drop=True
    )


def tune_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 32,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    n_jobs: int = 1,
    scale_numeric: ScaleMode | bool | None = None,
) -> Any:
    """Build a tunable pipeline for ``model_name`` and run Bayesian search on it.

    Convenience wrapper that pairs :func:`build_tunable_pipeline` with
    :func:`run_bayes_search`, so callers do not have to keep the search space and
    estimator in sync by hand.

    Args:
        model_name: Model name key (see :data:`AVAILABLE_MODELS`).
        x_train: Training features.
        y_train: Training target.
        n_iter: Number of Bayesian search iterations.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for cross-validation.
        random_state: Random seed for reproducibility.
        n_jobs: Parallel jobs (-1 to use all cores).
        scale_numeric: Override the recommended numeric scaling mode.

    Returns:
        Fitted BayesSearchCV.
    """
    pipeline = build_tunable_pipeline(
        model_name,
        x_train,
        random_state=random_state,
        scale_numeric=scale_numeric,
    )

    return run_bayes_search(
        pipeline,
        get_search_space(model_name),
        x_train,
        y_train,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def run_bayes_search_from_config(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> Any:
    """Run Bayesian search using project configuration.

    Args:
        model_name: Model name key used to look up the default search space.
        x_train: Training features.
        y_train: Training target.
        config: Full project configuration dictionary containing `tuning`.

    Returns:
        Fitted BayesSearchCV.
    """
    tuning_config = config.get("tuning", {})

    return tune_model(
        model_name,
        x_train,
        y_train,
        n_iter=tuning_config.get("n_iter", 32),
        cv=tuning_config.get("cv", 5),
        scoring=tuning_config.get("scoring", "neg_root_mean_squared_error"),
        random_state=tuning_config.get("random_state", 42),
        n_jobs=tuning_config.get("n_jobs", 1),
    )
