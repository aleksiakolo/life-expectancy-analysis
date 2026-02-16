from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Optional[Sequence[str]] = None,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build a simple sklearn ColumnTransformer:
      numeric: impute (median) + optional StandardScaler
      categorical: impute (most_frequent) + OneHotEncoder
    """
    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols) if categorical_cols is not None else []

    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_impute_strategy))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)

    transformers = [("num", numeric_pipe, numeric_cols)]

    if len(categorical_cols) > 0:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def build_model_pipeline(model, preprocessor: ColumnTransformer) -> Pipeline:
    """Standard pattern: preprocessor -> model."""
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def infer_feature_types(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Convenience helper: infer numeric/categorical columns if not provided.
    Keeps things simple: numeric = number dtype, categorical = object/category/bool.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # avoid overlap
    numeric_set = set(numeric_cols)
    categorical_cols = [c for c in categorical_cols if c not in numeric_set]
    return numeric_cols, categorical_cols
