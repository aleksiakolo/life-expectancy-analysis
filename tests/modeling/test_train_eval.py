import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from life_expectancy.modeling.pipelines import build_model_pipeline, build_preprocessor
from life_expectancy.modeling.train_eval import (
    EvalResult,
    build_prediction_dataframe,
    fit_predict,
    regression_metrics,
    results_to_dataframe,
    train_eval,
    train_eval_from_config,
)


def make_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([2.0, 4.0, 6.0, 8.0])

    x_test = pd.DataFrame({"x": [5.0, 6.0]}, index=[10, 11])
    y_test = pd.Series([10.0, 12.0], index=[10, 11])

    return x_train, x_test, y_train, y_test


def make_pipeline() -> Pipeline:
    preprocessor = build_preprocessor(
        numeric_cols=["x"],
        categorical_cols=[],
        scale_numeric="none",
    )

    return build_model_pipeline(
        model=LinearRegression(),
        preprocessor=preprocessor,
    )


def test_regression_metrics_perfect_predictions() -> None:
    metrics = regression_metrics(
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.0, 2.0, 3.0]),
    )

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_fit_predict() -> None:
    x_train, x_test, y_train, _ = make_train_test_data()

    predictions = fit_predict(
        make_pipeline(),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
    )

    assert predictions.shape == (2,)
    assert np.allclose(predictions, [10.0, 12.0])


def test_build_prediction_dataframe_without_ids() -> None:
    _, x_test, _, y_test = make_train_test_data()

    result = build_prediction_dataframe(
        y_true=y_test,
        y_pred=np.array([9.0, 13.0]),
        x_test=x_test,
    )

    assert result["y_true"].tolist() == [10.0, 12.0]
    assert result["y_pred"].tolist() == [9.0, 13.0]
    assert result["error"].tolist() == [-1.0, 1.0]
    assert result["abs_error"].tolist() == [1.0, 1.0]


def test_build_prediction_dataframe_with_ids() -> None:
    _, x_test, _, y_test = make_train_test_data()
    id_df = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2012, 2012],
        },
        index=[10, 11],
    )

    result = build_prediction_dataframe(
        y_true=y_test,
        y_pred=np.array([10.0, 12.0]),
        x_test=x_test,
        id_df=id_df,
        id_cols=["country", "year"],
    )

    assert result["country"].tolist() == ["A", "B"]
    assert result["year"].tolist() == [2012, 2012]


def test_build_prediction_dataframe_missing_id_column_raises() -> None:
    _, x_test, _, y_test = make_train_test_data()
    id_df = pd.DataFrame({"country": ["A", "B"]}, index=[10, 11])

    with pytest.raises(KeyError):
        build_prediction_dataframe(
            y_true=y_test,
            y_pred=np.array([10.0, 12.0]),
            x_test=x_test,
            id_df=id_df,
            id_cols=["country", "year"],
        )


def test_train_eval_with_predictions() -> None:
    x_train, x_test, y_train, y_test = make_train_test_data()

    result, pred_df = train_eval(
        make_pipeline(),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="LinearRegression",
        split_name="test_split",
    )

    assert isinstance(result, EvalResult)
    assert result.model_name == "LinearRegression"
    assert result.split_name == "test_split"
    assert result.n_train == 4
    assert result.n_test == 2
    assert result.rmse == pytest.approx(0.0)
    assert pred_df is not None
    assert pred_df.shape[0] == 2


def test_train_eval_without_predictions() -> None:
    x_train, x_test, y_train, y_test = make_train_test_data()

    _, pred_df = train_eval(
        make_pipeline(),
        x_train,
        y_train,
        x_test,
        y_test,
        return_predictions_df=False,
    )

    assert pred_df is None


def test_results_to_dataframe() -> None:
    results = [
        EvalResult(
            model_name="m1",
            split_name="s1",
            n_train=10,
            n_test=5,
            rmse=1.0,
            mae=0.5,
            r2=0.9,
        )
    ]

    df = results_to_dataframe(results)

    assert df.loc[0, "model_name"] == "m1"
    assert df.loc[0, "rmse"] == 1.0


def test_train_eval_from_config() -> None:
    x_train, x_test, y_train, y_test = make_train_test_data()
    config = {
        "modeling": {
            "evaluation": {
                "model_name": "ConfiguredModel",
                "split_name": "configured_split",
                "return_predictions_df": False,
            }
        }
    }

    result, pred_df = train_eval_from_config(
        make_pipeline(),
        x_train,
        y_train,
        x_test,
        y_test,
        config,
    )

    assert result.model_name == "ConfiguredModel"
    assert result.split_name == "configured_split"
    assert pred_df is None
