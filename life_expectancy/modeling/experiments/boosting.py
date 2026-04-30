from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from life_expectancy.modeling.experiments.core import append_run_log
from life_expectancy.modeling.model.boosting import build_boosting_model
from life_expectancy.modeling.pipelines import build_preprocessor
from life_expectancy.modeling.train_eval import regression_metrics

Summary = dict[str, Any]


@dataclass(frozen=True)
class BoostingEvalResult:
    """Evaluation result for an external boosting model.

    Attributes:
        model_name: Model name.
        split_name: Split name.
        n_train: Number of training rows.
        n_val: Number of validation rows.
        n_test: Number of test rows.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        r2: R-squared score.
        best_iteration: Best boosting iteration, if available.
    """

    model_name: str
    split_name: str
    n_train: int
    n_val: int
    n_test: int
    rmse: float
    mae: float
    r2: float
    best_iteration: int | None = None


def make_time_train_val_test(
    df: pd.DataFrame,
    *,
    target_col: str,
    year_col: str = "year",
    test_years: int = 3,
    val_years: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create time-aware train, validation, and test sets.

    Args:
        df: Input modeling DataFrame.
        target_col: Target column name.
        year_col: Year column name.
        test_years: Number of latest years used for testing.
        val_years: Number of latest pre-test years used for validation.

    Returns:
        Tuple containing train, validation, and test DataFrames.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If the split would produce an empty block.
    """
    require_columns(df, [target_col, year_col])

    if test_years <= 0:
        raise ValueError("test_years must be positive.")

    if val_years <= 0:
        raise ValueError("val_years must be positive.")

    data = df.dropna(subset=[target_col]).copy()
    data[year_col] = pd.to_numeric(data[year_col], errors="raise").astype(int)

    max_year = int(data[year_col].max())
    test_cutoff = max_year - test_years + 1

    train_val_df = data[data[year_col] < test_cutoff].copy()
    test_df = data[data[year_col] >= test_cutoff].copy()

    train_years = sorted(train_val_df[year_col].unique())

    if len(train_years) <= val_years:
        raise ValueError("Not enough pre-test years to create validation set.")

    val_year_values = train_years[-val_years:]
    train_df = train_val_df[~train_val_df[year_col].isin(val_year_values)].copy()
    val_df = train_val_df[train_val_df[year_col].isin(val_year_values)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Time split produced empty train, validation, or test set.")

    return train_df, val_df, test_df


def run_boosting_time_experiment(
    df: pd.DataFrame,
    *,
    feature_list: list[str],
    target_col: str,
    model_family: str,
    model_name: str,
    year_col: str = "year",
    test_years: int = 3,
    val_years: int = 1,
    scale_numeric: str | bool | None = "none",
    model_params: dict[str, Any] | None = None,
    run_log_path: str | Path | None = None,
    id_cols: list[str] | None = None,
) -> tuple[Summary, pd.DataFrame, Any, Pipeline]:
    """Run one time-aware external boosting experiment.

    Args:
        df: Modeling DataFrame.
        feature_list: Feature columns used for modeling.
        target_col: Target column name.
        model_family: One of `xgb`, `lgbm`, or `catboost`.
        model_name: Human-readable experiment name.
        year_col: Year column name.
        test_years: Number of latest years used for testing.
        val_years: Number of latest pre-test years used for validation.
        scale_numeric: Numeric scaling mode.
        model_params: Optional model parameters.
        run_log_path: Optional CSV path for appending result rows.
        id_cols: Optional metadata columns attached to predictions.

    Returns:
        Tuple containing result row, prediction DataFrame, fitted model, and
        fitted preprocessor.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If model family or split settings are invalid.
    """
    model_params = model_params or {}
    id_cols = id_cols or ["country", "country_code", "region", "income_group", year_col]

    keep_cols = collect_experiment_columns(
        df=df,
        feature_list=feature_list,
        target_col=target_col,
        year_col=year_col,
        id_cols=id_cols,
    )
    work_df = df[keep_cols].copy()

    train_df, val_df, test_df = make_time_train_val_test(
        work_df,
        target_col=target_col,
        year_col=year_col,
        test_years=test_years,
        val_years=val_years,
    )

    x_train = train_df[feature_list].copy()
    y_train = train_df[target_col].copy()
    x_val = val_df[feature_list].copy()
    y_val = val_df[target_col].copy()
    x_test = test_df[feature_list].copy()
    y_test = test_df[target_col].copy()

    preprocessor = build_preprocessor(
        numeric_cols=feature_list,
        categorical_cols=[],
        scale_numeric=scale_numeric,
    )

    x_train_t = preprocessor.fit_transform(x_train)
    x_val_t = preprocessor.transform(x_val)
    x_test_t = preprocessor.transform(x_test)

    model = build_boosting_model(
        model_family,
        params=model_params,
    )

    best_iteration = fit_boosting_model(
        model=model,
        model_family=model_family,
        x_train=x_train_t,
        y_train=y_train,
        x_val=x_val_t,
        y_val=y_val,
    )

    predictions = model.predict(x_test_t)
    metrics = regression_metrics(y_test, predictions)

    result = BoostingEvalResult(
        model_name=model_name,
        split_name="time",
        n_train=len(x_train),
        n_val=len(x_val),
        n_test=len(x_test),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        best_iteration=best_iteration,
    )

    pred_df = build_boosting_prediction_df(
        y_true=y_test,
        y_pred=predictions,
        test_df=test_df,
        id_cols=id_cols,
    )

    row = asdict(result)

    if run_log_path is not None:
        append_run_log(row, run_log_path)

    return row, pred_df, model, preprocessor


def fit_boosting_model(
    *,
    model: Any,
    model_family: str,
    x_train: np.ndarray,
    y_train: pd.Series,
    x_val: np.ndarray,
    y_val: pd.Series,
) -> int | None:
    """Fit one external boosting model with validation data.

    Args:
        model: External boosting estimator.
        model_family: One of `xgb`, `lgbm`, or `catboost`.
        x_train: Transformed training features.
        y_train: Training target.
        x_val: Transformed validation features.
        y_val: Validation target.

    Returns:
        Best iteration if available, otherwise None.

    Raises:
        ValueError: If model family is unsupported.
    """
    if model_family == "xgb":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        return getattr(model, "best_iteration", None)

    if model_family == "lgbm":
        import lightgbm as lgb

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        return getattr(model, "best_iteration_", None)

    if model_family == "catboost":
        model.fit(
            x_train,
            y_train,
            eval_set=(x_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        return get_catboost_best_iteration(model)

    raise ValueError("model_family must be one of: xgb, lgbm, catboost.")


def get_catboost_best_iteration(model: Any) -> int | None:
    """Return CatBoost best iteration when available.

    Args:
        model: Fitted CatBoost model.

    Returns:
        Best iteration or None.
    """
    try:
        return model.get_best_iteration()
    except AttributeError:
        return None


def build_boosting_prediction_df(
    *,
    y_true: pd.Series,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    """Create prediction DataFrame for a boosting experiment.

    Args:
        y_true: Test target values.
        y_pred: Test predictions.
        test_df: Test DataFrame containing metadata columns.
        id_cols: Metadata columns to attach when available.

    Returns:
        Prediction DataFrame.
    """
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    errors = pred_values - true_values

    pred_df = pd.DataFrame(
        {
            "y_true": true_values,
            "y_pred": pred_values,
            "error": errors,
            "abs_error": np.abs(errors),
        },
        index=test_df.index,
    )

    for col in id_cols:
        if col in test_df.columns:
            pred_df[col] = test_df[col].values

    return pred_df.reset_index(drop=True)


def collect_experiment_columns(
    *,
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    year_col: str,
    id_cols: list[str],
) -> list[str]:
    """Collect required and optional columns for a boosting experiment.

    Args:
        df: Input DataFrame.
        feature_list: Model feature columns.
        target_col: Target column name.
        year_col: Year column name.
        id_cols: Optional metadata columns.

    Returns:
        Ordered list of columns to keep.

    Raises:
        KeyError: If required columns are missing.
    """
    required_cols = list(dict.fromkeys([*feature_list, target_col, year_col]))
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    optional_cols = [col for col in id_cols if col in df.columns]

    return list(dict.fromkeys([*required_cols, *optional_cols]))


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate required columns.

    Args:
        df: DataFrame to validate.
        columns: Required column names.

    Raises:
        KeyError: If required columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
