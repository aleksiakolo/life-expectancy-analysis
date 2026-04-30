import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from life_expectancy.modeling.pipelines import (
    build_model_pipeline,
    build_numeric_steps,
    build_preprocessor,
    build_preprocessor_from_config,
    infer_feature_types,
)


def make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2010, 2011, 2012],
            "gdp": [100.0, None, 300.0],
            "region": ["A", "B", "A"],
            "status_flag": [1, 0, 1],
        }
    )


def test_infer_feature_types() -> None:
    df = make_features()

    numeric_cols, categorical_cols = infer_feature_types(df)

    assert numeric_cols == ["year", "gdp", "status_flag"]
    assert categorical_cols == ["region"]


def test_infer_feature_types_removes_overlap() -> None:
    df = make_features()

    numeric_cols, categorical_cols = infer_feature_types(
        df,
        numeric_cols=["year", "region"],
        categorical_cols=["region"],
    )

    assert numeric_cols == ["year", "region"]
    assert categorical_cols == []


def test_build_numeric_steps_standard_scaler() -> None:
    steps = build_numeric_steps(
        impute_strategy="median",
        scale_numeric="standard",
        add_missing_indicators=False,
    )

    assert [name for name, _ in steps] == ["imputer", "scaler"]


def test_build_numeric_steps_no_scaler() -> None:
    steps = build_numeric_steps(
        impute_strategy="median",
        scale_numeric="none",
        add_missing_indicators=False,
    )

    assert [name for name, _ in steps] == ["imputer"]


def test_build_numeric_steps_invalid_scaler_raises() -> None:
    with pytest.raises(ValueError):
        build_numeric_steps(
            impute_strategy="median",
            scale_numeric="bad",
            add_missing_indicators=False,
        )


def test_build_preprocessor_numeric_only() -> None:
    preprocessor = build_preprocessor(
        numeric_cols=["year", "gdp"],
        categorical_cols=[],
    )

    transformed = preprocessor.fit_transform(make_features())

    assert transformed.shape[0] == 3
    assert transformed.shape[1] == 2


def test_build_preprocessor_with_categorical() -> None:
    preprocessor = build_preprocessor(
        numeric_cols=["year", "gdp"],
        categorical_cols=["region"],
        scale_numeric="none",
    )

    transformed = preprocessor.fit_transform(make_features())

    assert transformed.shape[0] == 3
    assert transformed.shape[1] == 4


def test_build_preprocessor_from_config() -> None:
    df = make_features()
    config = {
        "modeling": {
            "pipeline": {
                "numeric_cols": ["year", "gdp"],
                "categorical_cols": ["region"],
                "scale_numeric": "robust",
            }
        }
    }

    preprocessor = build_preprocessor_from_config(df, config)
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] == 4


def test_build_model_pipeline() -> None:
    preprocessor = build_preprocessor(
        numeric_cols=["year", "gdp"],
        categorical_cols=[],
    )
    pipeline = build_model_pipeline(
        model=LinearRegression(),
        preprocessor=preprocessor,
    )

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["prep", "model"]
