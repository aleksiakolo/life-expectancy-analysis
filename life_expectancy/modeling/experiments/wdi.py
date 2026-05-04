from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from life_expectancy.modeling.train_eval import regression_metrics

Summary = dict[str, Any]


def make_panel_overlap_split(
    df: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    year_col: str,
    val_years: int,
    test_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Summary]:
    """Split WDI data so test years overlap with the processed panel."""
    wdi_years = set(pd.to_numeric(df[year_col], errors="raise").astype(int))
    panel_years = set(pd.to_numeric(panel[year_col], errors="raise").astype(int))

    overlap_years = sorted(wdi_years & panel_years)

    if len(overlap_years) < test_years + val_years:
        raise ValueError(
            "Not enough overlapping years to create validation and test split."
        )

    test_year_values = overlap_years[-test_years:]
    remaining_overlap = [year for year in overlap_years if year < test_year_values[0]]
    val_year_values = remaining_overlap[-val_years:]

    val_start = min(val_year_values)

    train_df = df[df[year_col] < val_start].copy()
    val_df = df[df[year_col].isin(val_year_values)].copy()
    test_df = df[df[year_col].isin(test_year_values)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("WDI split produced an empty train, validation, or test set.")

    summary: Summary = {
        "train_year_min": int(train_df[year_col].min()),
        "train_year_max": int(train_df[year_col].max()),
        "val_years": list(map(int, val_year_values)),
        "test_years": list(map(int, test_year_values)),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }

    return train_df, val_df, test_df, summary


def build_wdi_preprocessor(
    X: pd.DataFrame,
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build preprocessing pipeline for WDI models."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = [
        ("num", Pipeline(numeric_steps), numeric_cols),
    ]

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_prediction_frame(
    base_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    year_col: str,
    metadata_cols: list[str],
) -> pd.DataFrame:
    """Create prediction DataFrame with metadata and errors."""
    keep_cols = [
        col
        for col in ["country", "country_code", year_col, *metadata_cols]
        if col in base_df.columns
    ]

    out = base_df[keep_cols].copy()
    out["y_true"] = np.asarray(y_true)
    out["y_pred"] = np.asarray(y_pred)
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = np.abs(out["error"])

    return out.reset_index(drop=True)


def fit_evaluate_wdi_model(
    *,
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    year_col: str,
    metadata_cols: list[str],
    scale_numeric: bool,
) -> tuple[Summary, pd.DataFrame, Pipeline]:
    """Fit one WDI model and evaluate on validation and test sets."""
    preprocessor = build_wdi_preprocessor(
        X_train,
        scale_numeric=scale_numeric,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", clone(model)),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    row: Summary = {
        "model_name": model_name,
        "dataset": "WDI",
        "split_name": "panel_overlap_test",
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "r2": test_metrics["r2"],
    }

    pred_df = make_prediction_frame(
        test_df,
        y_true=y_test,
        y_pred=test_pred,
        year_col=year_col,
        metadata_cols=metadata_cols,
    )

    return row, pred_df, pipeline
