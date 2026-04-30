from __future__ import annotations

from typing import Any

Trial = dict[str, Any]


def xgb_regressor(params: dict[str, Any] | None = None):
    """Create an XGBoost regressor.

    Args:
        params: Optional XGBoost parameters.

    Returns:
        Configured XGBRegressor.

    Raises:
        ImportError: If xgboost is not installed.
    """
    from xgboost import XGBRegressor

    params = params or {}

    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=params.pop("early_stopping_rounds", 50),
        **params,
    )


def lgbm_regressor(params: dict[str, Any] | None = None):
    """Create a LightGBM regressor.

    Args:
        params: Optional LightGBM parameters.

    Returns:
        Configured LGBMRegressor.

    Raises:
        ImportError: If lightgbm is not installed.
    """
    from lightgbm import LGBMRegressor

    default_params: dict[str, Any] = {
        "objective": "regression",
        "n_estimators": 1500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbose": -1,
    }

    if params:
        default_params.update(params)

    return LGBMRegressor(**default_params)


def catboost_regressor(params: dict[str, Any] | None = None):
    """Create a CatBoost regressor.

    Args:
        params: Optional CatBoost parameters.

    Returns:
        Configured CatBoostRegressor.

    Raises:
        ImportError: If catboost is not installed.
    """
    from catboost import CatBoostRegressor

    default_params: dict[str, Any] = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.05,
        "depth": 6,
        "od_type": "Iter",
        "od_wait": 50,
        "random_seed": 42,
        "verbose": False,
    }

    if params:
        default_params.update(params)

    return CatBoostRegressor(**default_params)


def get_xgb_trial_grid(
    *,
    random_state: int = 42,
) -> list[Trial]:
    """Create a small XGBoost trial grid.

    Args:
        random_state: Random seed.

    Returns:
        List of trial dictionaries with `trial_name` and `params`.
    """
    trials: list[Trial] = []

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

    for learning_rate in [0.03, 0.05, 0.10]:
        trials.append(
            {
                "trial_name": f"xgb_lr_{learning_rate}",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": learning_rate,
                    "max_depth": 5,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 1.0,
                    "random_state": random_state,
                },
            }
        )

    for n_estimators, subsample, colsample in [
        (800, 0.8, 0.8),
        (1200, 0.8, 0.8),
        (1500, 1.0, 0.8),
    ]:
        trials.append(
            {
                "trial_name": (
                    f"xgb_cap_{n_estimators}_" f"sub{subsample}_col{colsample}"
                ),
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


def get_default_boosting_registry(
    *,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Create default external boosting model specs.

    Args:
        random_state: Random seed.

    Returns:
        Dictionary mapping model names to model specs.
    """
    return {
        "xgb": {
            "builder": xgb_regressor,
            "params": {
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "random_state": random_state,
            },
            "scale_numeric": "none",
        },
        "lgbm": {
            "builder": lgbm_regressor,
            "params": {
                "random_state": random_state,
            },
            "scale_numeric": "none",
        },
        "catboost": {
            "builder": catboost_regressor,
            "params": {
                "random_seed": random_state,
            },
            "scale_numeric": "none",
        },
    }


def build_boosting_model(
    model_name: str,
    *,
    params: dict[str, Any] | None = None,
):
    """Build one external boosting model by name.

    Args:
        model_name: One of `xgb`, `lgbm`, or `catboost`.
        params: Optional model parameters.

    Returns:
        External boosting regressor.

    Raises:
        ValueError: If the model name is unsupported.
    """
    if model_name == "xgb":
        return xgb_regressor(params)

    if model_name == "lgbm":
        return lgbm_regressor(params)

    if model_name == "catboost":
        return catboost_regressor(params)

    raise ValueError(f"Unsupported boosting model: {model_name}")
