from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from life_expectancy.data.utils import coerce_year_to_int

Summary = dict[str, Any]


def add_log_features(df: pd.DataFrame, *, cols: list[str]) -> pd.DataFrame:
    """Add log1p-transformed versions of selected nonnegative columns.

    Args:
        df: Input DataFrame.
        cols: Columns to transform.

    Returns:
        DataFrame with added `{column}_log1p` features.
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        values = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
        out[f"{col}_log1p"] = np.log1p(values)

    return out


def add_status_flag(
    df: pd.DataFrame, *, status_col: str, output_col: str
) -> pd.DataFrame:
    """Create a binary developed-status flag.

    Args:
        df: Input DataFrame.
        status_col: Column containing development status labels.
        output_col: Name of the binary output flag.

    Returns:
        DataFrame with the status flag added when the status column exists.
    """
    out = df.copy()

    if status_col in out.columns and output_col not in out.columns:
        out[output_col] = (
            out[status_col].astype(str).str.lower().str.contains("developed")
        ).astype(int)

    return out


def add_missingness_flags(df: pd.DataFrame, *, cols: list[str]) -> pd.DataFrame:
    """Add binary missingness flags for selected columns.

    Args:
        df: Input DataFrame.
        cols: Columns for which to create missingness flags.

    Returns:
        DataFrame with added `{column}_missing_flag` columns.
    """
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[f"{col}_missing_flag"] = out[col].isna().astype(int)

    return out


def add_interaction_features(
    df: pd.DataFrame, *, interaction_pairs: list[list[str]]
) -> pd.DataFrame:
    """Add pairwise multiplicative interaction features.

    Args:
        df: Input DataFrame.
        interaction_pairs: Pairs of columns to multiply.

    Returns:
        DataFrame with added `{a}__x__{b}` interaction columns.
    """
    out = df.copy()

    for pair in interaction_pairs:
        if len(pair) != 2:
            continue

        left_col, right_col = pair

        if left_col in out.columns and right_col in out.columns:
            left = pd.to_numeric(out[left_col], errors="coerce")
            right = pd.to_numeric(out[right_col], errors="coerce")
            out[f"{left_col}__x__{right_col}"] = left * right

    return out


def prepare_numeric_model_frame(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Build a numeric-only modeling DataFrame using feature config.

    This function creates configured engineered features, removes leakage and
    identifier columns, keeps only numeric predictors, and appends the target column.

    Args:
        df: Processed analytical panel.
        config: Full project configuration dictionary containing a `features` section.

    Returns:
        Numeric model-ready DataFrame containing predictors and target.

    Raises:
        KeyError: If the configured target column is missing.
    """
    feature_config = config["features"]

    target_col = feature_config.get("target_col", "life_expectancy")
    year_col = feature_config.get("year_col", "year")
    status_col = feature_config.get("status_col", "status")
    status_flag_col = feature_config.get("status_flag_col", "status_flag")

    log_cols = feature_config.get("log_candidates", [])
    missingness_flag_cols = feature_config.get("missingness_flag_candidates", [])
    interaction_pairs = feature_config.get("interactions", [])

    leakage_cols = set(feature_config.get("leakage_cols", []))
    id_cols = set(feature_config.get("drop_id_cols", []))

    out = df.copy()

    if target_col not in out.columns:
        raise KeyError(f"Target column {target_col!r} not found in DataFrame.")

    if year_col in out.columns:
        out[year_col] = coerce_year_to_int(out[year_col])

    out = add_status_flag(
        out,
        status_col=status_col,
        output_col=status_flag_col,
    )
    out = add_log_features(out, cols=log_cols)
    out = add_missingness_flags(out, cols=missingness_flag_cols)
    out = add_interaction_features(out, interaction_pairs=interaction_pairs)

    drop_cols = {target_col, status_col, *leakage_cols, *id_cols}
    candidate_cols = [col for col in out.columns if col not in drop_cols]

    numeric_predictors = [
        col for col in candidate_cols if pd.api.types.is_numeric_dtype(out[col])
    ]

    final_cols = list(dict.fromkeys([*numeric_predictors, target_col]))
    model_df = out[final_cols].copy()
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    if year_col in model_df.columns:
        model_df[year_col] = pd.to_numeric(
            model_df[year_col],
            errors="raise",
        ).astype(int)

    return model_df
