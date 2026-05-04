from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin

from life_expectancy.modeling.models.baselines import (
    baseline_mean,
    elasticnet_cv,
    lasso_cv,
    linear_regression,
    ridge_cv,
)
from life_expectancy.modeling.models.neural import mlp_regressor
from life_expectancy.modeling.models.tree import (
    extra_trees_regressor,
    hist_gradient_boosting_regressor,
    random_forest_regressor,
)

ModelSpec = dict[str, Any]
ModelRegistry = dict[str, ModelSpec]


def get_default_model_registry(
    *,
    random_state: int = 42,
) -> ModelRegistry:
    """Return the default model registry.

    Args:
        random_state: Random seed used by stochastic models.

    Returns:
        Dictionary mapping model names to model specifications. Each specification
        contains an estimator and the recommended numeric scaling mode.
    """
    return {
        "mean": {
            "model": baseline_mean(),
            "scale_numeric": "none",
        },
        "linear": {
            "model": linear_regression(),
            "scale_numeric": "standard",
        },
        "ridge": {
            "model": ridge_cv(),
            "scale_numeric": "standard",
        },
        "lasso": {
            "model": lasso_cv(random_state=random_state),
            "scale_numeric": "standard",
        },
        "elasticnet": {
            "model": elasticnet_cv(random_state=random_state),
            "scale_numeric": "standard",
        },
        "hgb": {
            "model": hist_gradient_boosting_regressor(
                random_state=random_state,
            ),
            "scale_numeric": "none",
        },
        "rf": {
            "model": random_forest_regressor(
                random_state=random_state,
            ),
            "scale_numeric": "none",
        },
        "extra_trees": {
            "model": extra_trees_regressor(
                random_state=random_state,
            ),
            "scale_numeric": "none",
        },
        "mlp": {
            "model": mlp_regressor(
                random_state=random_state,
            ),
            "scale_numeric": "robust",
        },
    }


def get_model_spec(
    model_name: str,
    *,
    registry: ModelRegistry | None = None,
    random_state: int = 42,
) -> ModelSpec:
    """Return one model specification by name.

    Args:
        model_name: Name of the model to retrieve.
        registry: Optional existing model registry.
        random_state: Random seed used if a registry is created internally.

    Returns:
        Model specification dictionary.

    Raises:
        KeyError: If the model name is not in the registry.
    """
    if registry is None:
        registry = get_default_model_registry(random_state=random_state)

    if model_name not in registry:
        available = sorted(registry)
        raise KeyError(f"Unknown model {model_name!r}. Available models: {available}")

    return registry[model_name]


def get_model(
    model_name: str,
    *,
    registry: ModelRegistry | None = None,
    random_state: int = 42,
) -> RegressorMixin:
    """Return only the estimator for one model.

    Args:
        model_name: Name of the model to retrieve.
        registry: Optional existing model registry.
        random_state: Random seed used if a registry is created internally.

    Returns:
        Scikit-learn compatible regressor.
    """
    return get_model_spec(
        model_name,
        registry=registry,
        random_state=random_state,
    )["model"]


def get_scale_numeric(
    model_name: str,
    *,
    registry: ModelRegistry | None = None,
    random_state: int = 42,
) -> str:
    """Return the recommended numeric scaling mode for one model.

    Args:
        model_name: Name of the model to retrieve.
        registry: Optional existing model registry.
        random_state: Random seed used if a registry is created internally.

    Returns:
        Scaling mode string.
    """
    return get_model_spec(
        model_name,
        registry=registry,
        random_state=random_state,
    )["scale_numeric"]


def get_selected_model_registry(
    model_names: list[str],
    *,
    random_state: int = 42,
) -> ModelRegistry:
    """Return a registry containing only selected models.

    Args:
        model_names: Model names to keep.
        random_state: Random seed used by stochastic models.

    Returns:
        Filtered model registry.

    Raises:
        KeyError: If any requested model is unavailable.
    """
    registry = get_default_model_registry(random_state=random_state)

    missing = [name for name in model_names if name not in registry]
    if missing:
        available = sorted(registry)
        raise KeyError(f"Unknown models {missing}. Available models: {available}")

    return {name: registry[name] for name in model_names}


def get_model_registry_from_config(
    config: dict[str, Any],
) -> ModelRegistry:
    """Build a model registry from project configuration.

    Args:
        config: Full project configuration dictionary. Expected optional section:
            `modeling.registry`.

    Returns:
        Model registry, optionally filtered to configured model names.
    """
    registry_config = config.get("modeling", {}).get("registry", {})
    random_state = registry_config.get("random_state", 42)
    model_names = registry_config.get("model_names")

    if model_names is None:
        return get_default_model_registry(random_state=random_state)

    return get_selected_model_registry(
        model_names=list(model_names),
        random_state=random_state,
    )
