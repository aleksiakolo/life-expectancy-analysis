from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class EvalResult:
    """Metrics and a few helpful counts."""
    model_name: str
    split_name: str
    n_train: int
    n_test: int
    rmse: float
    mae: float
    r2: float


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Computes basic regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def fit_predict(pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Fits pipeline on train and return predictions on test."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return preds


def train_eval(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    split_name: str = "split",
    return_predictions_df: bool = True,
    id_df: Optional[pd.DataFrame] = None,
    id_cols: Optional[list[str]] = None,
) -> Tuple[EvalResult, Optional[pd.DataFrame]]:
    """
    Trains and evaluates a pipeline. Optionally returns a predictions dataframe for diagnostics.
    If you pass id_df + id_cols, those columns are attached to the prediction output.
    """
    preds = fit_predict(pipeline, X_train, y_train, X_test)
    m = regression_metrics(y_test, preds)

    result = EvalResult(
        model_name=model_name,
        split_name=split_name,
        n_train=len(X_train),
        n_test=len(X_test),
        rmse=m["rmse"],
        mae=m["mae"],
        r2=m["r2"],
    )

    pred_df = None
    if return_predictions_df:
        pred_df = pd.DataFrame(
            {
                "y_true": np.asarray(y_test),
                "y_pred": np.asarray(preds),
                "error": np.asarray(preds) - np.asarray(y_test),
                "abs_error": np.abs(np.asarray(preds) - np.asarray(y_test)),
            }
        )
        if id_df is not None and id_cols:
            # align by index
            for c in id_cols:
                pred_df[c] = id_df.loc[X_test.index, c].values

    return result, pred_df


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Turns list of EvalResult into a nice DataFrame."""
    return pd.DataFrame([r.__dict__ for r in results])
