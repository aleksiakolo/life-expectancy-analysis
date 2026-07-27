import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import RidgeCV as SklearnRidgeCV
from sklearn.pipeline import Pipeline

from life_expectancy.modeling.tuning import (
    AVAILABLE_MODELS,
    build_tunable_pipeline,
    get_search_space,
    get_tunable_estimator,
    get_tunable_scale_numeric,
    tune_model,
)


def make_regression_frame(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(2.0 * x["a"] - 1.0 * x["b"] + rng.normal(scale=0.1, size=n))
    return x, y


# --- pure helpers: importable and testable without scikit-optimize ---


def test_available_models_includes_core_models() -> None:
    for name in ["ridge", "lasso", "elasticnet", "hgb", "rf", "extra_trees"]:
        assert name in AVAILABLE_MODELS


def test_get_tunable_estimator_returns_base_not_cv() -> None:
    # Regression test: registry uses RidgeCV (no tunable `alpha`); the tunable
    # estimator must be a plain Ridge whose `alpha` matches the search space.
    estimator = get_tunable_estimator("ridge")

    assert isinstance(estimator, Ridge)
    assert not isinstance(estimator, SklearnRidgeCV)
    assert "alpha" in estimator.get_params()


def test_get_tunable_estimator_lasso() -> None:
    estimator = get_tunable_estimator("lasso")

    assert isinstance(estimator, Lasso)
    assert "alpha" in estimator.get_params()


def test_get_tunable_estimator_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_tunable_estimator("not_a_model")


def test_get_tunable_scale_numeric() -> None:
    assert get_tunable_scale_numeric("ridge") == "standard"
    assert get_tunable_scale_numeric("hgb") == "none"


def test_get_search_space_unknown_raises() -> None:
    # Unknown names are rejected before any scikit-optimize import.
    with pytest.raises(KeyError):
        get_search_space("not_a_model")


def test_build_tunable_pipeline_steps() -> None:
    x, _ = make_regression_frame()

    pipeline = build_tunable_pipeline("ridge", x, scale_numeric="standard")

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["prep", "model"]
    assert isinstance(pipeline.named_steps["model"], Ridge)


# --- search-space construction and execution: require scikit-optimize ---


def test_search_space_params_match_tunable_estimator() -> None:
    # Core regression test for the integration bug: every search-space parameter
    # must be a real, settable parameter of the matching tunable estimator.
    pytest.importorskip("skopt")

    for name in ["ridge", "lasso", "elasticnet", "hgb", "rf", "extra_trees"]:
        space = get_search_space(name)
        estimator_params = get_tunable_estimator(name).get_params()

        for param_name in space:
            assert param_name.startswith("model__")
            attr = param_name.split("__", 1)[1]
            assert attr in estimator_params, f"{name}: {attr} not in estimator"


def test_tune_model_ridge_runs_and_selects_alpha() -> None:
    pytest.importorskip("skopt")
    x, y = make_regression_frame()

    searcher = tune_model("ridge", x, y, n_iter=6, cv=3)

    assert "model__alpha" in searcher.best_params_
    # Tuned pipeline is refit and can predict.
    preds = searcher.best_estimator_.predict(x)
    assert len(preds) == len(y)


def test_tune_model_hgb_runs() -> None:
    pytest.importorskip("skopt")
    x, y = make_regression_frame()

    searcher = tune_model("hgb", x, y, n_iter=5, cv=3)

    assert searcher.best_params_
    assert searcher.best_score_ <= 0.0  # neg RMSE
