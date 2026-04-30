from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ID_COLS = ["country", "year", "region", "income_group"]


def attach_predictions(
    df: pd.DataFrame,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Attach prediction, error, and absolute-error columns.

    Args:
        df: Base DataFrame to copy.
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        y_true_col: Name of the true-value output column.
        y_pred_col: Name of the prediction output column.

    Returns:
        DataFrame with true values, predictions, errors, and absolute errors.

    Raises:
        ValueError: If true and predicted values have different lengths.
    """
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)

    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(df) != len(true_values):
        raise ValueError("df, y_true, and y_pred must have the same length.")

    out = df.copy()
    out[y_true_col] = true_values
    out[y_pred_col] = pred_values
    out["error"] = out[y_pred_col] - out[y_true_col]
    out["abs_error"] = np.abs(out["error"])

    return out


def worst_errors_table(
    pred_df: pd.DataFrame,
    *,
    n: int = 10,
    cols_to_show: list[str] | None = None,
) -> pd.DataFrame:
    """Return rows with the largest absolute errors.

    Args:
        pred_df: Prediction DataFrame containing `abs_error`.
        n: Number of worst-error rows to return.
        cols_to_show: Optional columns to include in the output.

    Returns:
        DataFrame sorted by descending absolute error.

    Raises:
        KeyError: If `abs_error` is missing.
    """
    require_columns(pred_df, ["abs_error"])

    worst = pred_df.sort_values("abs_error", ascending=False).head(n).copy()

    if cols_to_show is None:
        id_cols = [col for col in DEFAULT_ID_COLS if col in worst.columns]
        cols_to_show = id_cols + ["y_true", "y_pred", "error", "abs_error"]

    available_cols = [col for col in cols_to_show if col in worst.columns]

    return worst[available_cols].reset_index(drop=True)


def group_error_table(
    pred_df: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    """Compute MAE and RMSE by group.

    Args:
        pred_df: Prediction DataFrame containing `error` and `abs_error`.
        group_col: Column used for grouping, such as region or income group.

    Returns:
        Group-level error table sorted by descending MAE.

    Raises:
        KeyError: If required columns are missing.
    """
    require_columns(pred_df, [group_col, "error", "abs_error"])

    grouped = (
        pred_df.groupby(group_col)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("error", root_mean_square),
        )
        .reset_index()
    )

    return grouped.sort_values("mae", ascending=False).reset_index(drop=True)


def time_slice_error_table(
    pred_df: pd.DataFrame,
    *,
    year_col: str = "year",
) -> pd.DataFrame:
    """Compute MAE and RMSE by year.

    Args:
        pred_df: Prediction DataFrame containing `error`, `abs_error`, and year.
        year_col: Year column name.

    Returns:
        Year-level error table sorted by year.

    Raises:
        KeyError: If required columns are missing.
    """
    require_columns(pred_df, [year_col, "error", "abs_error"])

    grouped = (
        pred_df.groupby(year_col)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("error", root_mean_square),
        )
        .reset_index()
    )

    return grouped.sort_values(year_col).reset_index(drop=True)


def root_mean_square(values: pd.Series) -> float:
    """Compute root mean square for a numeric series.

    Args:
        values: Numeric values.

    Returns:
        Root mean square value.
    """
    array = np.asarray(values, dtype=float)

    return float(np.sqrt(np.mean(np.square(array))))


def plot_predicted_vs_actual(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: plt.Axes | None = None,
    title: str = "Predicted vs Actual",
) -> plt.Axes:
    """Plot predicted values against actual values.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        ax: Optional matplotlib axes.
        title: Plot title.

    Returns:
        Matplotlib axes object.
    """
    ax = get_axes(ax)
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)

    ax.scatter(true_values, pred_values, alpha=0.5)

    lower = float(min(true_values.min(), pred_values.min()))
    upper = float(max(true_values.max(), pred_values.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    return ax


def plot_residuals_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: plt.Axes | None = None,
    title: str = "Residuals vs Predicted",
) -> plt.Axes:
    """Plot residuals against predicted values.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        ax: Optional matplotlib axes.
        title: Plot title.

    Returns:
        Matplotlib axes object.
    """
    ax = get_axes(ax)
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    residuals = pred_values - true_values

    ax.scatter(pred_values, residuals, alpha=0.5)
    ax.axhline(0.0, linestyle="--")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred - actual)")
    ax.set_title(title)

    return ax


def plot_residual_hist(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: plt.Axes | None = None,
    bins: int = 30,
    title: str = "Residual Distribution",
) -> plt.Axes:
    """Plot residual distribution.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        ax: Optional matplotlib axes.
        bins: Number of histogram bins.
        title: Plot title.

    Returns:
        Matplotlib axes object.
    """
    ax = get_axes(ax)
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    residuals = pred_values - true_values

    ax.hist(residuals, bins=bins, alpha=0.8)
    ax.axvline(0.0, linestyle="--")

    ax.set_xlabel("Residual (pred - actual)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    return ax


def get_axes(ax: plt.Axes | None) -> plt.Axes:
    """Return existing axes or create new axes.

    Args:
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib axes object.
    """
    if ax is not None:
        return ax

    _, new_ax = plt.subplots()

    return new_ax


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate required columns.

    Args:
        df: DataFrame to validate.
        columns: Required column names.

    Raises:
        KeyError: If any required columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
