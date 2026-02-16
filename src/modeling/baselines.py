from __future__ import annotations
from typing import Optional, Sequence
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV


def baseline_mean() -> DummyRegressor:
    """Predicts the training mean (super simple baseline)."""
    return DummyRegressor(strategy="mean")


def linear_regression() -> LinearRegression:
    """Plain Linear Regression baseline."""
    return LinearRegression()


def ridge_cv(
    alphas: Optional[Sequence[float]] = None,
    cv: int = 5,
) -> RidgeCV:
    """Ridge regression with built-in CV for alpha. Defaults are fine for a starter baseline."""
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    return RidgeCV(alphas=list(alphas), cv=cv)


def lasso_cv(
    alphas: Optional[Sequence[float]] = None,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 5000,
) -> LassoCV:
    """Lasso regression with CV. Good for feature selection but can be unstable on correlated features."""
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    return LassoCV(alphas=list(alphas), cv=cv, random_state=random_state, max_iter=max_iter)


def elasticnet_cv(
    l1_ratio: Optional[Sequence[float]] = None,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 5000,
) -> ElasticNetCV:
    """ElasticNet regression with CV. Compromise between Ridge and Lasso. Tunes both alpha and l1_ratio."""
    if l1_ratio is None:
        l1_ratio = [0.1, 0.25, 0.5, 0.75, 0.9]
    return ElasticNetCV(
        l1_ratio=list(l1_ratio),
        cv=cv,
        random_state=random_state,
        max_iter=max_iter,
    )
