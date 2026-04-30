from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

ModelSpec = dict[str, Any]
ModelRegistry = dict[str, ModelSpec]


def hist_gradient_boosting_regressor(
    *,
    learning_rate: float = 0.05,
    max_depth: int | None = None,
    max_leaf_nodes: int = 31,
    max_iter: int = 400,
    l2_regularization: float = 0.0,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """Create a HistGradientBoostingRegressor.

    Args:
        learning_rate: Boosting learning rate.
        max_depth: Maximum tree depth.
        max_leaf_nodes: Maximum leaf nodes per tree.
        max_iter: Maximum boosting iterations.
        l2_regularization: L2 regularization strength.
        random_state: Random seed.

    Returns:
        Configured HistGradientBoostingRegressor.
    """
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )


def random_forest_regressor(
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    max_features: str | float | int | None = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Create a RandomForestRegressor.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        max_features: Number or fraction of features considered per split.
        random_state: Random seed.
        n_jobs: Number of parallel jobs.

    Returns:
        Configured RandomForestRegressor.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def extra_trees_regressor(
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    max_features: str | float | int | None = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> ExtraTreesRegressor:
    """Create an ExtraTreesRegressor.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        max_features: Number or fraction of features considered per split.
        random_state: Random seed.
        n_jobs: Number of parallel jobs.

    Returns:
        Configured ExtraTreesRegressor.
    """
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_hgb_sweep(
    *,
    random_state: int = 42,
) -> ModelRegistry:
    """Create a small HistGradientBoosting hyperparameter sweep.

    Args:
        random_state: Random seed.

    Returns:
        Dictionary mapping trial names to model specs.
    """
    configs: list[dict[str, Any]] = [
        {"learning_rate": 0.03, "max_depth": 3, "max_leaf_nodes": 15},
        {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 31},
        {"learning_rate": 0.05, "max_depth": 5, "max_leaf_nodes": 31},
        {"learning_rate": 0.10, "max_depth": 3, "max_leaf_nodes": 31},
        {"learning_rate": 0.05, "max_depth": None, "max_leaf_nodes": 63},
    ]

    sweep: ModelRegistry = {}

    for config in configs:
        name = build_hgb_trial_name(config)
        sweep[name] = {
            "model": hist_gradient_boosting_regressor(
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                max_leaf_nodes=config["max_leaf_nodes"],
                random_state=random_state,
            ),
            "scale_numeric": "none",
        }

    return sweep


def build_hgb_trial_name(config: dict[str, Any]) -> str:
    """Build a readable HGB trial name.

    Args:
        config: HGB parameter dictionary.

    Returns:
        Trial name string.
    """
    return (
        f"hgb_lr{config['learning_rate']}_"
        f"d{config['max_depth']}_"
        f"leaf{config['max_leaf_nodes']}"
    )
