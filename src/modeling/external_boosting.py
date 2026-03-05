from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class ExternalEvalResult:
    model_name: str
    split_name: str
    n_train: int
    n_val: int
    n_test: int
    rmse: float
    mae: float
    r2: float
    best_iteration: int | None = None


def _numeric_preprocessor(scale_mode: str = "none") -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_mode == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scale_mode == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scale_mode == "none":
        pass
    else:
        raise ValueError("scale_mode must be one of {'none', 'standard', 'robust'}")

    return Pipeline(steps=steps)


def make_time_train_val_test(
    df: pd.DataFrame,
    target_col: str,
    year_col: str = "year",
    test_years: int = 3,
    val_years: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy()

    if target_col not in data.columns:
        raise KeyError(f"{target_col!r} not found")
    if year_col not in data.columns:
        raise KeyError(f"{year_col!r} not found")

    data = data.dropna(subset=[target_col]).copy()
    data[year_col] = pd.to_numeric(data[year_col], errors="raise").astype(int)

    max_year = int(data[year_col].max())
    test_cutoff = max_year - (test_years - 1)

    trainval_df = data[data[year_col] < test_cutoff].copy()
    test_df = data[data[year_col] >= test_cutoff].copy()

    train_years = sorted(trainval_df[year_col].unique())
    if len(train_years) <= val_years:
        raise ValueError("Not enough pre-test years to carve out a validation block.")

    val_year_values = train_years[-val_years:]
    train_df = trainval_df[~trainval_df[year_col].isin(val_year_values)].copy()
    val_df = trainval_df[trainval_df[year_col].isin(val_year_values)].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Time split produced empty train / val / test.")

    return train_df, val_df, test_df


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def append_run_log(row: dict[str, Any], out_path: str) -> None:
    out_df = pd.DataFrame([row])
    try:
        old = pd.read_csv(out_path)
        out_df = pd.concat([old, out_df], ignore_index=True)
    except FileNotFoundError:
        pass
    out_df.to_csv(out_path, index=False)


def get_xgb_trial_grid(random_state: int = 42) -> list[dict[str, Any]]:
    trials = []

    # max_depth sweep
    for depth in [3, 5, 7]:
        trials.append(
            {
                "trial_name": f"xgb_depth_{depth}",
                "params": {
                    "n_estimators": 1200,
                    "learning_rate": 0.05,
                    "max_depth": depth,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 1.0,
                    "random_state": random_state,
                },
            }
        )

    # learning_rate sweep
    for lr in [0.03, 0.05, 0.10]:
        trials.append(
            {
                "trial_name": f"xgb_lr_{lr}",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": lr,
                    "max_depth": 5,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 1.0,
                    "random_state": random_state,
                },
            }
        )

    # n_estimators / subsample / colsample sweep
    for n_estimators, subsample, colsample in [
        (800, 0.8, 0.8),
        (1200, 0.8, 0.8),
        (1500, 1.0, 0.8),
    ]:
        trials.append(
            {
                "trial_name": f"xgb_cap_{n_estimators}_sub{subsample}_col{colsample}",
                "params": {
                    "n_estimators": n_estimators,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "subsample": subsample,
                    "colsample_bytree": colsample,
                    "reg_lambda": 1.0,
                    "random_state": random_state,
                },
            }
        )

    return trials


def _build_xgb(params: dict[str, Any]):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=50,
        **params,
    )


def _build_lgbm(random_state: int = 42):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
    )


def _build_catboost(random_state: int = 42):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        od_type="Iter",
        od_wait=50,
        random_seed=random_state,
        verbose=False,
    )


def run_external_timeaware_trial(
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    model_family: str,
    model_name: str,
    year_col: str = "year",
    test_years: int = 3,
    val_years: int = 1,
    scale_mode: str = "none",
    model_params: dict[str, Any] | None = None,
    run_log_path: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Any, Pipeline]:
    model_params = model_params or {}

    meta_cols = [c for c in ["country", "country_code", "region", "income_group", year_col] if c in df.columns]
    keep_cols = list(dict.fromkeys([*feature_list, target_col, *meta_cols]))

    work_df = df[keep_cols].copy()
    train_df, val_df, test_df = make_time_train_val_test(
        work_df,
        target_col=target_col,
        year_col=year_col,
        test_years=test_years,
        val_years=val_years,
    )

    X_train = train_df[feature_list].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[feature_list].copy()
    y_val = val_df[target_col].copy()

    X_test = test_df[feature_list].copy()
    y_test = test_df[target_col].copy()

    prep = _numeric_preprocessor(scale_mode=scale_mode)
    X_train_t = prep.fit_transform(X_train)
    X_val_t = prep.transform(X_val)
    X_test_t = prep.transform(X_test)

    if model_family == "xgb":
        model = _build_xgb(model_params)
        model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
        best_iteration = getattr(model, "best_iteration", None)

    elif model_family == "lgbm":
        import lightgbm as lgb

        model = _build_lgbm(model_params.get("random_state", 42))
        model.set_params(**{k: v for k, v in model_params.items() if k != "random_state"})
        model.fit(
            X_train_t,
            y_train,
            eval_set=[(X_val_t, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        best_iteration = getattr(model, "best_iteration_", None)

    elif model_family == "catboost":
        model = _build_catboost(model_params.get("random_state", 42))
        model.set_params(**{k: v for k, v in model_params.items() if k != "random_state"})
        model.fit(X_train_t, y_train, eval_set=(X_val_t, y_val), use_best_model=True, verbose=False)
        try:
            best_iteration = model.get_best_iteration()
        except Exception:
            best_iteration = None

    else:
        raise ValueError("model_family must be one of {'xgb', 'lgbm', 'catboost'}")

    preds = model.predict(X_test_t)
    m = regression_metrics(y_test, preds)

    result = ExternalEvalResult(
        model_name=model_name,
        split_name="time",
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        rmse=m["rmse"],
        mae=m["mae"],
        r2=m["r2"],
        best_iteration=best_iteration,
    )

    pred_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_test),
            "y_pred": np.asarray(preds),
            "error": np.asarray(preds) - np.asarray(y_test),
            "abs_error": np.abs(np.asarray(preds) - np.asarray(y_test)),
        },
        index=test_df.index,
    )

    for c in meta_cols:
        pred_df[c] = test_df[c].values

    row = {
        "model_name": result.model_name,
        "split_name": result.split_name,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "n_test": result.n_test,
        "rmse": result.rmse,
        "mae": result.mae,
        "r2": result.r2,
        "best_iteration": result.best_iteration,
    }

    if run_log_path is not None:
        append_run_log(row, run_log_path)

    return row, pred_df.reset_index(drop=True), model, prep