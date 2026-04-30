from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

Metrics = dict[str, float]


@dataclass(frozen=True)
class EvalResult:
    """Regression evaluation result.

    Attributes:
        model_name: Name of the evaluated model.
        split_name: Name of the train/test split.
        n_train: Number of training rows.
        n_test: Number of test rows.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        r2: R-squared score.
    """

    model_name: str
    split_name: str
    n_train: int
    n_test: int
    rmse: float
    mae: float
    r2: float


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> Metrics:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing RMSE, MAE, and R-squared.
    """
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)

    return {
        "rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "r2": float(r2_score(true_values, pred_values)),
    }


def fit_predict(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> np.ndarray:
    """Fit a pipeline and predict on a test set.

    Args:
        pipeline: Scikit-learn Pipeline.
        x_train: Training features.
        y_train: Training target.
        x_test: Test features.

    Returns:
        Prediction array for the test rows.
    """
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    return np.asarray(predictions)


def train_eval(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str = "model",
    split_name: str = "split",
    return_predictions_df: bool = True,
    id_df: pd.DataFrame | None = None,
    id_cols: list[str] | None = None,
) -> tuple[EvalResult, pd.DataFrame | None]:
    """Train and evaluate a regression pipeline.

    Args:
        pipeline: Scikit-learn Pipeline.
        x_train: Training features.
        y_train: Training target.
        x_test: Test features.
        y_test: Test target.
        model_name: Human-readable model name.
        split_name: Human-readable split name.
        return_predictions_df: Whether to return row-level predictions.
        id_df: Optional DataFrame containing identifier columns.
        id_cols: Optional identifier columns to attach to prediction rows.

    Returns:
        Tuple containing evaluation metrics and an optional prediction DataFrame.

    Raises:
        KeyError: If requested ID columns are missing.
    """
    predictions = fit_predict(
        pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
    )
    metrics = regression_metrics(y_test, predictions)

    result = EvalResult(
        model_name=model_name,
        split_name=split_name,
        n_train=len(x_train),
        n_test=len(x_test),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
    )

    prediction_df = None

    if return_predictions_df:
        prediction_df = build_prediction_dataframe(
            y_true=y_test,
            y_pred=predictions,
            x_test=x_test,
            id_df=id_df,
            id_cols=id_cols,
        )

    return result, prediction_df


def build_prediction_dataframe(
    *,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    id_df: pd.DataFrame | None = None,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a row-level prediction DataFrame.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        x_test: Test feature DataFrame used for index alignment.
        id_df: Optional DataFrame containing identifier columns.
        id_cols: Optional identifier columns to attach.

    Returns:
        Prediction DataFrame with true values, predictions, errors, and IDs.

    Raises:
        KeyError: If requested ID columns are missing from `id_df`.
    """
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    errors = pred_values - true_values

    prediction_df = pd.DataFrame(
        {
            "y_true": true_values,
            "y_pred": pred_values,
            "error": errors,
            "abs_error": np.abs(errors),
        },
        index=x_test.index,
    )

    if id_df is not None and id_cols:
        missing_cols = [col for col in id_cols if col not in id_df.columns]

        if missing_cols:
            raise KeyError(f"Missing ID columns: {missing_cols}")

        for col in id_cols:
            prediction_df[col] = id_df.loc[x_test.index, col].values

    return prediction_df.reset_index(drop=True)


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Convert evaluation results to a DataFrame.

    Args:
        results: List of EvalResult objects.

    Returns:
        DataFrame with one row per evaluation result.
    """
    return pd.DataFrame([asdict(result) for result in results])


def train_eval_from_config(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
) -> tuple[EvalResult, pd.DataFrame | None]:
    """Train and evaluate using modeling evaluation config.

    Args:
        pipeline: Scikit-learn Pipeline.
        x_train: Training features.
        y_train: Training target.
        x_test: Test features.
        y_test: Test target.
        config: Full project configuration dictionary containing
            `modeling.evaluation`.

    Returns:
        Tuple containing evaluation metrics and optional prediction DataFrame.
    """
    eval_config = config.get("modeling", {}).get("evaluation", {})

    return train_eval(
        pipeline,
        x_train,
        y_train,
        x_test,
        y_test,
        model_name=eval_config.get("model_name", "model"),
        split_name=eval_config.get("split_name", "split"),
        return_predictions_df=eval_config.get("return_predictions_df", True),
    )
