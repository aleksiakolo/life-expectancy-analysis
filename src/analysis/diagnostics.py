from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def attach_predictions(
    df: pd.DataFrame,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Return df copy with prediction columns + errors."""
    out = df.copy()
    out[y_true_col] = np.asarray(y_true)
    out[y_pred_col] = np.asarray(y_pred)
    out["error"] = out[y_pred_col] - out[y_true_col]
    out["abs_error"] = np.abs(out["error"])
    return out


def worst_errors_table(pred_df: pd.DataFrame, n: int = 10, cols_to_show: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    pred_df should contain: y_true, y_pred, abs_error, and (optionally) country/year/etc.
    """
    if "abs_error" not in pred_df.columns:
        raise KeyError("pred_df must have an 'abs_error' column. Use attach_predictions() first.")
    worst = pred_df.sort_values("abs_error", ascending=False).head(n).copy()

    if cols_to_show is None:
        # show whatever ids exist + core columns
        base_cols = [c for c in ["country", "year", "region", "income_group"] if c in worst.columns]
        cols_to_show = base_cols + ["y_true", "y_pred", "error", "abs_error"]

    cols_to_show = [c for c in cols_to_show if c in worst.columns]
    return worst[cols_to_show]


def group_error_table(pred_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute MAE/RMSE by group (e.g., region)."""
    if group_col not in pred_df.columns:
        raise KeyError(f"group_col='{group_col}' not found in pred_df.columns")

    def rmse(x):
        return float(np.sqrt(np.mean(np.square(x))))

    grouped = pred_df.groupby(group_col).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        rmse=("error", rmse),
    ).reset_index()

    return grouped.sort_values("mae", ascending=False)


def time_slice_error_table(pred_df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """Compute MAE/RMSE by year (good for time-split sanity checks)."""
    if year_col not in pred_df.columns:
        raise KeyError(f"year_col='{year_col}' not found in pred_df.columns")

    def rmse(x):
        return float(np.sqrt(np.mean(np.square(x))))

    grouped = pred_df.groupby(year_col).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        rmse=("error", rmse),
    ).reset_index()

    return grouped.sort_values(year_col)


# -----------------------------
# Plot helpers 
# -----------------------------

def plot_predicted_vs_actual(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Predicted vs Actual",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ax.scatter(y_true, y_pred, alpha=0.5)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")  # y=x reference

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return ax


def plot_residuals_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Residuals vs Predicted",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resid = y_pred - y_true

    ax.scatter(y_pred, resid, alpha=0.5)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred - actual)")
    ax.set_title(title)
    return ax


def plot_residual_hist(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
    title: str = "Residual Distribution",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resid = y_pred - y_true

    ax.hist(resid, bins=bins, alpha=0.8)
    ax.axvline(0.0, linestyle="--")
    ax.set_xlabel("Residual (pred - actual)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax
