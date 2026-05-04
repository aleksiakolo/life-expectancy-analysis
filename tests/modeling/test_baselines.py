import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV

from life_expectancy.modeling.models.baselines import (
    baseline_mean,
    elasticnet_cv,
    get_baseline_model,
    lasso_cv,
    linear_regression,
    ridge_cv,
)


def test_baseline_mean() -> None:
    model = baseline_mean()

    assert isinstance(model, DummyRegressor)
    assert model.strategy == "mean"


def test_linear_regression() -> None:
    model = linear_regression()

    assert isinstance(model, LinearRegression)


def test_ridge_cv_defaults() -> None:
    model = ridge_cv()

    assert isinstance(model, RidgeCV)
    assert list(model.alphas) == [0.1, 1.0, 10.0, 100.0, 1000.0]


def test_ridge_cv_custom_config() -> None:
    model = ridge_cv(alphas=[1.0, 2.0], cv=3)

    assert list(model.alphas) == [1.0, 2.0]
    assert model.cv == 3


def test_lasso_cv_defaults() -> None:
    model = lasso_cv()

    assert isinstance(model, LassoCV)
    assert list(model.alphas) == [0.1, 1.0, 10.0, 100.0, 1000.0]
    assert model.random_state == 42
    assert model.max_iter == 5000


def test_elasticnet_cv_defaults() -> None:
    model = elasticnet_cv()

    assert isinstance(model, ElasticNetCV)
    assert list(model.l1_ratio) == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert model.random_state == 42
    assert model.max_iter == 5000


def test_get_baseline_model() -> None:
    assert isinstance(get_baseline_model("mean"), DummyRegressor)
    assert isinstance(get_baseline_model("linear"), LinearRegression)
    assert isinstance(get_baseline_model("ridge"), RidgeCV)
    assert isinstance(get_baseline_model("lasso"), LassoCV)
    assert isinstance(get_baseline_model("elasticnet"), ElasticNetCV)


def test_get_baseline_model_with_config() -> None:
    model = get_baseline_model(
        "ridge",
        {
            "alphas": [1.0, 10.0],
            "cv": 3,
        },
    )

    assert isinstance(model, RidgeCV)
    assert list(model.alphas) == [1.0, 10.0]
    assert model.cv == 3


def test_get_baseline_model_unsupported_raises() -> None:
    with pytest.raises(ValueError):
        get_baseline_model("unknown")
