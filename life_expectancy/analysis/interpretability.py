from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.pipeline import Pipeline


def _get_estimator_and_transformed(
    pipeline: Pipeline,
    x: pd.DataFrame,
) -> tuple[Any, np.ndarray, list[str]]:
    """Extract the final estimator, transformed X, and feature names from a pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline built by build_model_pipeline()
            (steps: "prep", "model").
        x: Input feature DataFrame.

    Returns:
        Tuple of (estimator, x_transformed, feature_names).
    """
    prep = pipeline.named_steps["prep"]
    estimator = pipeline.named_steps["model"]

    x_transformed = prep.transform(x)

    try:
        feature_names = list(prep.get_feature_names_out())
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(x_transformed.shape[1])]

    return estimator, x_transformed, feature_names


def shap_importance(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    *,
    max_background_samples: int = 200,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values for each feature.

    Uses TreeExplainer for tree-based models and falls back to KernelExplainer
    for others (slower, uses a background sample).

    Args:
        pipeline: Fitted sklearn Pipeline (steps: "prep", "model").
        x_test: Test feature DataFrame.
        max_background_samples: Max rows used as KernelExplainer background.

    Returns:
        DataFrame with columns ["feature", "mean_abs_shap"], sorted descending.
    """
    try:
        # importlib is used to avoid static analysis tools complaining when
        # 'shap' is not installed in the environment used for linting.
        import importlib

        shap = importlib.import_module("shap")
    except ImportError as exc:
        raise ImportError("shap is required. Install with: pip install shap") from exc

    estimator, x_transformed, feature_names = _get_estimator_and_transformed(
        pipeline, x_test
    )

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(x_transformed)
    except Exception:
        n_bg = min(max_background_samples, len(x_transformed))
        explainer = shap.KernelExplainer(estimator.predict, x_transformed[:n_bg])
        shap_values = explainer.shap_values(x_transformed, nsamples=100)

    mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)

    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_df(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_repeats: int = 10,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance for a fitted pipeline.

    Permutation importance measures how much the score drops when a feature's
    values are randomly shuffled. Works on the original (pre-transform) features,
    so column names match the input DataFrame directly.

    Args:
        pipeline: Fitted sklearn Pipeline.
        x_test: Test feature DataFrame.
        y_test: Test target series.
        n_repeats: Number of shuffle repeats per feature.
        scoring: Scoring metric (higher is better, e.g. neg_root_mean_squared_error).
        random_state: Random seed.

    Returns:
        DataFrame with columns ["feature", "importance_mean", "importance_std"],
        sorted by descending importance.
    """
    result = sklearn_permutation_importance(
        pipeline,
        x_test,
        y_test,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
    )

    return (
        pd.DataFrame(
            {
                "feature": list(x_test.columns),
                # sklearn.permutation_importance may return a Bunch or a dict-like
                # object depending on sklearn version; use item access to be safe
                "importance_mean": result["importances_mean"],
                "importance_std": result["importances_std"],
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_shap_bar(
    importance_df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    n_features: int = 20,
    title: str = "Mean |SHAP| Feature Importance",
) -> Axes:
    """Plot a horizontal bar chart of SHAP feature importance.

    Args:
        importance_df: DataFrame from shap_importance() with
            columns ["feature", "mean_abs_shap"].
        ax: Optional matplotlib axes.
        n_features: Number of top features to show.
        title: Plot title.

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, n_features * 0.35)))

    top = importance_df.head(n_features).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)

    return ax


def plot_permutation_bar(
    importance_df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    n_features: int = 20,
    title: str = "Permutation Feature Importance",
) -> Axes:
    """Plot a horizontal bar chart of permutation importance with error bars.

    Args:
        importance_df: DataFrame from permutation_importance_df() with
            columns ["feature", "importance_mean", "importance_std"].
        ax: Optional matplotlib axes.
        n_features: Number of top features to show.
        title: Plot title.

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, n_features * 0.35)))

    top = importance_df.head(n_features).iloc[::-1]
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        capsize=3,
    )
    ax.set_xlabel("Importance (mean decrease in score)")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)

    return ax
