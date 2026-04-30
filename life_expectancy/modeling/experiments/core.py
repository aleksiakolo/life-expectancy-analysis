from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from life_expectancy.modeling.pipelines import (
    build_model_pipeline,
    build_preprocessor,
    infer_feature_types,
)
from life_expectancy.modeling.splits import SplitInfo, make_time_split
from life_expectancy.modeling.train_eval import train_eval

Summary = dict[str, Any]


def save_time_split_metadata(split_info: SplitInfo, out_path: str | Path) -> Path:
    """Save time-split metadata to JSON.

    Args:
        split_info: Split metadata object.
        out_path: Output JSON path.

    Returns:
        Resolved output path.
    """
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(split_info), file, indent=2)

    return output_path


def append_run_log(row: Summary, out_path: str | Path) -> Path:
    """Append one experiment result row to a CSV log.

    Args:
        row: Experiment result row.
        out_path: Output CSV path.

    Returns:
        Resolved output path.
    """
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row])

    if output_path.exists():
        old_df = pd.read_csv(output_path)
        row_df = pd.concat([old_df, row_df], ignore_index=True)

    row_df.to_csv(output_path, index=False)

    return output_path


def run_time_experiment(
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    year_col: str,
    model_name: str,
    model: Any,
    *,
    scale_numeric: str | bool | None = "standard",
    test_years: int = 3,
    run_log_path: str | Path | None = None,
    split_label: str | None = None,
    add_numeric_missing_indicators: bool = False,
    id_cols: list[str] | None = None,
) -> tuple[Summary, pd.DataFrame, SplitInfo, Pipeline]:
    """Run one time-aware model experiment.

    Args:
        df: Modeling DataFrame.
        feature_list: Feature columns used by the model.
        target_col: Target column name.
        year_col: Year column name.
        model_name: Human-readable model name.
        model: Scikit-learn compatible estimator.
        scale_numeric: Numeric scaling mode passed to the preprocessor.
        test_years: Number of latest years used as the test set.
        run_log_path: Optional CSV path for appending result rows.
        split_label: Optional split name. Defaults to `"time"`.
        add_numeric_missing_indicators: Whether to add numeric missing indicators.
        id_cols: Optional ID columns to attach to prediction output.

    Returns:
        Tuple containing result row, prediction DataFrame, split metadata, and
        fitted pipeline.

    Raises:
        KeyError: If required columns are missing.
    """
    id_cols = id_cols or ["country", year_col]
    keep_cols = collect_required_columns(
        df=df,
        feature_list=feature_list,
        target_col=target_col,
        year_col=year_col,
        id_cols=id_cols,
    )

    work_df = df[keep_cols].copy()

    x_train, x_test, y_train, y_test, split_info = make_time_split(
        work_df,
        target_col=target_col,
        year_col=year_col,
        test_years=test_years,
    )

    x_train_model = x_train[feature_list].copy()
    x_test_model = x_test[feature_list].copy()

    numeric_cols, categorical_cols = infer_feature_types(x_train_model)

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=scale_numeric,
        add_numeric_missing_indicators=add_numeric_missing_indicators,
    )

    pipeline = build_model_pipeline(
        model=model,
        preprocessor=preprocessor,
    )

    split_name = split_label or "time"
    available_id_cols = [col for col in id_cols if col in work_df.columns]

    result, pred_df = train_eval(
        pipeline,
        x_train_model,
        y_train,
        x_test_model,
        y_test,
        model_name=model_name,
        split_name=split_name,
        return_predictions_df=True,
        id_df=work_df,
        id_cols=available_id_cols,
    )

    if pred_df is None:
        raise RuntimeError("Expected prediction DataFrame, but got None.")

    row: Summary = {
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

    return row, pred_df, split_info, pipeline


def run_time_experiment_from_config(
    df: pd.DataFrame,
    feature_list: list[str],
    model_name: str,
    model: Any,
    config: dict[str, Any],
) -> tuple[Summary, pd.DataFrame, SplitInfo, Pipeline]:
    """Run one time-aware experiment using project configuration.

    Args:
        df: Modeling DataFrame.
        feature_list: Feature columns used by the model.
        model_name: Human-readable model name.
        model: Scikit-learn compatible estimator.
        config: Full project configuration dictionary containing `modeling`.

    Returns:
        Tuple containing result row, prediction DataFrame, split metadata, and
        fitted pipeline.
    """
    modeling_config = config.get("modeling", {})
    split_config = modeling_config.get("split", {})
    pipeline_config = modeling_config.get("pipeline", {})
    experiment_config = modeling_config.get("experiment", {})

    return run_time_experiment(
        df=df,
        feature_list=feature_list,
        target_col=split_config.get("target_col", "life_expectancy"),
        year_col=split_config.get("year_col", "year"),
        model_name=model_name,
        model=model,
        scale_numeric=pipeline_config.get("scale_numeric", "standard"),
        test_years=split_config.get("test_years", 3),
        run_log_path=experiment_config.get("run_log_path"),
        split_label=experiment_config.get("split_label"),
        add_numeric_missing_indicators=pipeline_config.get(
            "add_numeric_missing_indicators",
            False,
        ),
        id_cols=experiment_config.get("id_cols", ["country", "year"]),
    )


def collect_required_columns(
    *,
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    year_col: str,
    id_cols: list[str],
) -> list[str]:
    """Collect columns needed for one experiment.

    Args:
        df: Input DataFrame.
        feature_list: Feature columns.
        target_col: Target column name.
        year_col: Year column name.
        id_cols: Optional metadata columns.

    Returns:
        Ordered list of available required columns.

    Raises:
        KeyError: If required feature, target, or year columns are missing.
    """
    required_cols = list(dict.fromkeys([*feature_list, target_col, year_col]))
    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        raise KeyError(f"Missing required experiment columns: {missing_required}")

    optional_cols = [col for col in id_cols if col in df.columns]

    return list(dict.fromkeys([*required_cols, *optional_cols]))
