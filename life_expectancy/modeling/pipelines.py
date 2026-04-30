from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

ScaleMode = Literal["standard", "robust", "none"]
FeatureTypes = tuple[list[str], list[str]]


def infer_feature_types(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> FeatureTypes:
    """Infer numeric and categorical feature columns.

    Args:
        df: Input feature DataFrame.
        numeric_cols: Optional explicit numeric columns.
        categorical_cols: Optional explicit categorical columns.

    Returns:
        Tuple containing numeric columns and categorical columns.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool", "str"]
        ).columns.tolist()

    numeric_set = set(numeric_cols)
    categorical_cols = [col for col in categorical_cols if col not in numeric_set]

    return list(numeric_cols), list(categorical_cols)


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    *,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    scale_numeric: ScaleMode | bool | None = "standard",
    add_numeric_missing_indicators: bool = False,
) -> ColumnTransformer:
    """Build a scikit-learn column preprocessor.

    Numeric features are imputed and optionally scaled. Categorical features are
    imputed and one-hot encoded.

    Args:
        numeric_cols: Numeric feature columns.
        categorical_cols: Optional categorical feature columns.
        numeric_impute_strategy: Strategy for numeric imputation.
        categorical_impute_strategy: Strategy for categorical imputation.
        scale_numeric: Scaling mode: `"standard"`, `"robust"`, `"none"`,
            `True`, `False`, or `None`.
        add_numeric_missing_indicators: Whether numeric imputers should add
            missingness indicators.

    Returns:
        ColumnTransformer ready to be used inside a scikit-learn Pipeline.

    Raises:
        ValueError: If `scale_numeric` is unsupported.
    """
    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols or [])

    numeric_pipe = Pipeline(
        steps=build_numeric_steps(
            impute_strategy=numeric_impute_strategy,
            scale_numeric=scale_numeric,
            add_missing_indicators=add_numeric_missing_indicators,
        )
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("num", numeric_pipe, numeric_cols)
    ]

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy=categorical_impute_strategy),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_preprocessor_from_config(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> ColumnTransformer:
    """Build a preprocessor using modeling pipeline configuration.

    Args:
        df: Input feature DataFrame.
        config: Full project configuration dictionary containing
            `modeling.pipeline`.

    Returns:
        Configured ColumnTransformer.
    """
    pipeline_config = config.get("modeling", {}).get("pipeline", {})

    numeric_cols, categorical_cols = infer_feature_types(
        df,
        numeric_cols=pipeline_config.get("numeric_cols"),
        categorical_cols=pipeline_config.get("categorical_cols"),
    )

    return build_preprocessor(
        numeric_cols,
        categorical_cols,
        numeric_impute_strategy=pipeline_config.get(
            "numeric_impute_strategy",
            "median",
        ),
        categorical_impute_strategy=pipeline_config.get(
            "categorical_impute_strategy",
            "most_frequent",
        ),
        scale_numeric=pipeline_config.get("scale_numeric", "standard"),
        add_numeric_missing_indicators=pipeline_config.get(
            "add_numeric_missing_indicators",
            False,
        ),
    )


def build_model_pipeline(
    model: Any,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Build a standard preprocessing-plus-model pipeline.

    Args:
        model: Scikit-learn compatible estimator.
        preprocessor: ColumnTransformer for preprocessing features.

    Returns:
        Scikit-learn Pipeline with `prep` and `model` steps.
    """
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )


def build_numeric_steps(
    *,
    impute_strategy: str,
    scale_numeric: ScaleMode | bool | None,
    add_missing_indicators: bool,
) -> list[tuple[str, Any]]:
    """Build preprocessing steps for numeric columns.

    Args:
        impute_strategy: Strategy passed to SimpleImputer.
        scale_numeric: Scaling mode.
        add_missing_indicators: Whether to add numeric missingness indicators.

    Returns:
        List of scikit-learn pipeline steps.

    Raises:
        ValueError: If `scale_numeric` is unsupported.
    """
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy=impute_strategy,
                add_indicator=add_missing_indicators,
            ),
        )
    ]

    if scale_numeric in ("standard", True):
        steps.append(("scaler", StandardScaler()))
    elif scale_numeric == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scale_numeric in ("none", None, False):
        pass
    else:
        raise ValueError(
            "scale_numeric must be one of: 'standard', 'robust', "
            "'none', True, False, or None."
        )

    return steps
