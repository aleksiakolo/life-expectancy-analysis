import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestRegressor

from life_expectancy.analysis.interpretability import (
    permutation_importance_df,
    plot_permutation_bar,
    plot_shap_bar,
    shap_importance,
)
from life_expectancy.modeling.pipelines import build_model_pipeline, build_preprocessor


def make_fitted_pipeline(n: int = 150):  # noqa: ANN201 - test helper
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    # f0 strongly informative, f3 pure noise.
    y = pd.Series(3.0 * x["f0"] - 1.0 * x["f1"] + rng.normal(scale=0.1, size=n))

    preprocessor = build_preprocessor(
        numeric_cols=list(x.columns),
        categorical_cols=[],
        scale_numeric="none",
    )
    pipeline = build_model_pipeline(
        model=RandomForestRegressor(n_estimators=40, random_state=0),
        preprocessor=preprocessor,
    )
    pipeline.fit(x, y)
    return pipeline, x, y


# --- permutation importance: scikit-learn only ---


def test_permutation_importance_columns_and_sorting() -> None:
    pipeline, x, y = make_fitted_pipeline()

    result = permutation_importance_df(pipeline, x, y, n_repeats=5)

    assert list(result.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(result["feature"]) == set(x.columns)
    # Sorted descending by mean importance.
    assert result["importance_mean"].is_monotonic_decreasing


def test_permutation_importance_ranks_informative_feature_first() -> None:
    pipeline, x, y = make_fitted_pipeline()

    result = permutation_importance_df(pipeline, x, y, n_repeats=5)

    assert result.iloc[0]["feature"] == "f0"


# --- plots: matplotlib only ---


def test_plot_permutation_bar_returns_axes() -> None:
    pipeline, x, y = make_fitted_pipeline()
    result = permutation_importance_df(pipeline, x, y, n_repeats=3)

    ax = plot_permutation_bar(result, n_features=4)

    assert isinstance(ax, Axes)


def test_plot_shap_bar_returns_axes() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "mean_abs_shap": [0.5, 0.3, 0.1],
        }
    )

    ax = plot_shap_bar(importance, n_features=3)

    assert isinstance(ax, Axes)


# --- SHAP importance: requires the optional `shap` package ---


def test_shap_importance_columns_and_sorting() -> None:
    pytest.importorskip("shap")
    pipeline, x, _ = make_fitted_pipeline()

    result = shap_importance(pipeline, x)

    assert list(result.columns) == ["feature", "mean_abs_shap"]
    assert result["mean_abs_shap"].is_monotonic_decreasing
    assert (result["mean_abs_shap"] >= 0).all()
