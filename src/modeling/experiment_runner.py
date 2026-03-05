from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
import pandas as pd
from sklearn.pipeline import Pipeline
from .pipelines import infer_feature_types, build_preprocessor_extended
from .splits import make_time_split
from .train_eval import train_eval


def save_time_split_metadata(split_info, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "split_type": split_info.split_type,
        "n_train": split_info.n_train,
        "n_test": split_info.n_test,
        "extra": split_info.extra,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


def append_run_log(row: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row])
    if out_path.exists():
        old = pd.read_csv(out_path)
        combined = pd.concat([old, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(out_path, index=False)
    return out_path


def run_time_experiment(
    df: pd.DataFrame,
    feature_list: Sequence[str],
    target_col: str,
    year_col: str,
    model_name: str,
    model,
    scale_mode: str = "standard",
    test_years: int = 3,
    run_log_path: str | Path | None = None,
    split_label: str | None = None,
    add_numeric_missing_indicators: bool = False,
) -> tuple[dict, pd.DataFrame, object, Pipeline]:
    keep_cols = list(dict.fromkeys([*feature_list, target_col, year_col]))
    extra_meta_cols = [c for c in ["country", "country_code", "region", "income_group"] if c in df.columns]
    keep_cols = list(dict.fromkeys([*keep_cols, *extra_meta_cols]))

    work_df = df[keep_cols].copy()

    X_train, X_test, y_train, y_test, split_info = make_time_split(
        work_df,
        target_col=target_col,
        year_col=year_col,
        test_years=test_years,
    )

    Xtr = X_train[list(feature_list)].copy()
    Xte = X_test[list(feature_list)].copy()

    numeric_cols, categorical_cols = infer_feature_types(Xtr)

    preprocessor = build_preprocessor_extended(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=scale_mode,
        add_numeric_missing_indicators=add_numeric_missing_indicators,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    split_name = split_label if split_label is not None else "time"

    result, pred_df = train_eval(
        pipe,
        Xtr,
        y_train,
        Xte,
        y_test,
        model_name=model_name,
        split_name=split_name,
        return_predictions_df=True,
        id_df=work_df,
        id_cols=[c for c in ["country", year_col] if c in work_df.columns],
    )

    row = {
        "model_name": result.model_name,
        "split_name": result.split_name,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "rmse": result.rmse,
        "mae": result.mae,
        "r2": result.r2,
    }

    if run_log_path is not None:
        append_run_log(row, run_log_path)

    return row, pred_df, split_info, pipe


def top_n_worst_predictions(pred_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = [c for c in ["country", "country_code", "region", "income_group", "year", "y_true", "y_pred", "error", "abs_error"] if c in pred_df.columns]
    return pred_df.sort_values("abs_error", ascending=False)[cols].head(n).reset_index(drop=True)


def grouped_mae(pred_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in pred_df.columns:
        raise KeyError(f"{group_col!r} not found in prediction dataframe")
    out = (
        pred_df.groupby(group_col, dropna=False)["abs_error"]
        .mean()
        .reset_index(name="mae")
        .sort_values("mae", ascending=False)
        .reset_index(drop=True)
    )
    return out