from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV

ModelConfig = dict[str, Any]


def baseline_mean() -> DummyRegressor:
    """Create a mean-prediction baseline model.

    Returns:
        DummyRegressor that always predicts the training target mean.
    """
    return DummyRegressor(strategy="mean")


def linear_regression() -> LinearRegression:
    """Create an ordinary least squares linear regression model.

    Returns:
        LinearRegression estimator.
    """
    return LinearRegression()


def ridge_cv(
    *,
    alphas: list[float] | None = None,
    cv: int = 5,
) -> RidgeCV:
    """Create a RidgeCV regression model.

    Args:
        alphas: Candidate regularization strengths.
        cv: Number of cross-validation folds.

    Returns:
        RidgeCV estimator.
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    return RidgeCV(
        alphas=alphas,
        cv=cv,
    )


def lasso_cv(
    *,
    alphas: list[float] | None = None,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 5000,
) -> LassoCV:
    """Create a LassoCV regression model.

    Args:
        alphas: Candidate regularization strengths.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducible coordinate descent.
        max_iter: Maximum optimization iterations.

    Returns:
        LassoCV estimator.
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    return LassoCV(
        alphas=alphas,
        cv=cv,
        random_state=random_state,
        max_iter=max_iter,
    )


def elasticnet_cv(
    *,
    l1_ratio: list[float] | None = None,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 5000,
) -> ElasticNetCV:
    """Create an ElasticNetCV regression model.

    Args:
        l1_ratio: Candidate L1/L2 mixing values.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducible coordinate descent.
        max_iter: Maximum optimization iterations.

    Returns:
        ElasticNetCV estimator.
    """
    if l1_ratio is None:
        l1_ratio = [0.1, 0.25, 0.5, 0.75, 0.9]

    return ElasticNetCV(
        l1_ratio=l1_ratio,
        cv=cv,
        random_state=random_state,
        max_iter=max_iter,
    )


def get_baseline_model(
    model_name: str,
    config: ModelConfig | None = None,
):
    """Create a baseline model by name.

    Args:
        model_name: Model name. Supported values are `mean`, `linear`,
            `ridge`, `lasso`, and `elasticnet`.
        config: Optional model-specific configuration.

    Returns:
        Scikit-learn regression estimator.

    Raises:
        ValueError: If the model name is unsupported.
    """
    config = config or {}

    if model_name == "mean":
        return baseline_mean()

    if model_name == "linear":
        return linear_regression()

    if model_name == "ridge":
        return ridge_cv(
            alphas=config.get("alphas"),
            cv=config.get("cv", 5),
        )

    if model_name == "lasso":
        return lasso_cv(
            alphas=config.get("alphas"),
            cv=config.get("cv", 5),
            random_state=config.get("random_state", 42),
            max_iter=config.get("max_iter", 5000),
        )

    if model_name == "elasticnet":
        return elasticnet_cv(
            l1_ratio=config.get("l1_ratio"),
            cv=config.get("cv", 5),
            random_state=config.get("random_state", 42),
            max_iter=config.get("max_iter", 5000),
        )

    raise ValueError(f"Unsupported baseline model: {model_name}")
