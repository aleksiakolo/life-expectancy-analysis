from __future__ import annotations
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from .baselines import lasso_cv, ridge_cv


def get_timeaware_model_registry(random_state: int = 42) -> dict[str, dict]:
    """
    Returns:
        model_name -> {"model": estimator, "scale_mode": "standard"/"robust"/"none"}
    """
    return {
        "RidgeCV": {
            "model": ridge_cv(),
            "scale_mode": "standard",
        },
        "LassoCV": {
            "model": lasso_cv(random_state=random_state),
            "scale_mode": "standard",
        },
        "HistGBR": {
            "model": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=None,
                max_leaf_nodes=31,
                max_iter=400,
                random_state=random_state,
            ),
            "scale_mode": "none",
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            "scale_mode": "none",
        },
        "ExtraTrees": {
            "model": ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            "scale_mode": "none",
        },
        "MLP": {
            "model": MLPRegressor(
                hidden_layer_sizes=(128, 64),
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=random_state,
            ),
            "scale_mode": "robust",
        },
    }


def get_hgbr_sweep(random_state: int = 42) -> dict[str, dict]:
    configs = [
        {"learning_rate": 0.03, "max_depth": 3, "max_leaf_nodes": 15},
        {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 31},
        {"learning_rate": 0.05, "max_depth": 5, "max_leaf_nodes": 31},
        {"learning_rate": 0.10, "max_depth": 3, "max_leaf_nodes": 31},
        {"learning_rate": 0.05, "max_depth": None, "max_leaf_nodes": 63},
    ]

    sweep = {}
    for cfg in configs:
        name = (
            f"HGB_lr{cfg['learning_rate']}_"
            f"d{cfg['max_depth']}_"
            f"leaf{cfg['max_leaf_nodes']}"
        )
        sweep[name] = {
            "model": HistGradientBoostingRegressor(
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                max_leaf_nodes=cfg["max_leaf_nodes"],
                max_iter=400,
                random_state=random_state,
            ),
            "scale_mode": "none",
        }
    return sweep