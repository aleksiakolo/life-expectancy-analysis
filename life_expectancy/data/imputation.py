from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from life_expectancy.data.utils import coerce_year_to_int, require_columns

JoinKey = tuple[str, str]
Summary = dict[str, Any]


def impute_panel(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, Summary]:
    """Impute missing values in the merged country-year panel.

    This function reads imputation settings from `config["imputation"]`. It applies
    imputation in three optional stages: country-level interpolation, group-median
    filling, and global-median filling. The configured target column is never
    imputed.

    Args:
        df: Merged country-year panel DataFrame.
        config: Full project configuration dictionary containing an `imputation`
            section.

    Returns:
        Tuple containing:
            - Imputed panel DataFrame.
            - Summary dictionary with filled-value counts and output metadata.

    Raises:
        KeyError: If required key columns, target column, or group columns are missing.
    """
    imputation_config = config["imputation"]

    key: JoinKey = tuple(imputation_config.get("key", ["country", "year"]))
    country_col, year_col = key

    target_col = imputation_config.get("target_col", "life_expectancy")
    numeric_only = imputation_config.get("numeric_only", True)
    interpolate_cols = imputation_config.get("interpolate_cols")
    interpolate_limit = imputation_config.get("interpolate_limit", 2)
    interpolate_method = imputation_config.get("interpolate_method", "linear")
    group_cols = imputation_config.get("group_cols")
    group_median_cols = imputation_config.get("group_median_cols")
    median_impute_cols = imputation_config.get("median_impute_cols")

    out = df.copy()

    require_columns(out, [country_col, year_col, target_col], name="panel")
    out[year_col] = coerce_year_to_int(out[year_col])

    summary: Summary = {
        "input_rows": len(out),
        "interpolate_cols": interpolate_cols or [],
        "median_impute_cols": median_impute_cols or [],
        "group_cols": group_cols or [],
        "group_median_cols": group_median_cols or [],
    }

    eligible_numeric = eligible_numeric_cols(
        out,
        target_col=target_col,
        numeric_only=numeric_only,
    )

    if interpolate_cols:
        summary["interpolate_filled_counts"] = interpolate_by_country(
            out,
            country_col=country_col,
            year_col=year_col,
            cols=interpolate_cols,
            eligible_numeric=eligible_numeric,
            method=interpolate_method,
            limit=interpolate_limit,
        )
    else:
        summary["interpolate_filled_counts"] = {}

    if group_cols and group_median_cols:
        summary["group_median_filled_counts"] = group_median_impute(
            out,
            group_cols=group_cols,
            cols=group_median_cols,
            eligible_numeric=eligible_numeric,
        )
    else:
        summary["group_median_filled_counts"] = {}

    if median_impute_cols:
        summary["median_filled_counts"] = global_median_impute(
            out,
            cols=median_impute_cols,
            eligible_numeric=eligible_numeric,
        )
    else:
        summary["median_filled_counts"] = {}

    summary["output_rows"] = len(out)
    summary["target_missing_count_after"] = int(out[target_col].isna().sum())

    return out, summary


def eligible_numeric_cols(
    df: pd.DataFrame, *, target_col: str, numeric_only: bool
) -> list[str]:
    """Return columns eligible for imputation.

    Args:
        df: Input DataFrame.
        target_col: Target column that must not be imputed.
        numeric_only: Whether to restrict imputation to numeric columns.

    Returns:
        List of eligible column names.
    """
    if not numeric_only:
        return [col for col in df.columns if col != target_col]

    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col in cols:
        cols.remove(target_col)

    return cols


def interpolate_by_country(
    df: pd.DataFrame,
    *,
    country_col: str,
    year_col: str,
    cols: list[str],
    eligible_numeric: list[str],
    method: str,
    limit: int,
) -> dict[str, int]:
    """Interpolate selected columns within each country over time.

    This function mutates the input DataFrame in place.

    Args:
        df: Input panel DataFrame.
        country_col: Country column name.
        year_col: Year column name.
        cols: Columns requested for interpolation.
        eligible_numeric: Columns allowed to be imputed.
        method: Interpolation method passed to pandas.
        limit: Maximum number of consecutive missing values to fill.

    Returns:
        Dictionary mapping each filled column to the number of values filled.
    """
    filled_counts: dict[str, int] = {}

    for col in cols:
        if col not in df.columns or col not in eligible_numeric:
            continue

        before_missing = int(df[col].isna().sum())

        sorted_df = df.sort_values([country_col, year_col])
        interpolated = sorted_df.groupby(country_col, group_keys=False)[col].apply(
            lambda series: series.interpolate(
                method=method,
                limit=limit,
                limit_direction="both",
            )
        )

        df.loc[interpolated.index, col] = interpolated

        after_missing = int(df[col].isna().sum())
        filled_counts[col] = before_missing - after_missing

    return filled_counts


def group_median_impute(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    cols: list[str],
    eligible_numeric: list[str],
) -> dict[str, int]:
    """Fill selected columns using medians within configured groups.

    This function mutates the input DataFrame in place.

    Args:
        df: Input panel DataFrame.
        group_cols: Columns used to define groups, such as region/income group.
        cols: Columns requested for group-median imputation.
        eligible_numeric: Columns allowed to be imputed.

    Returns:
        Dictionary mapping each filled column to the number of values filled.

    Raises:
        KeyError: If any group columns are missing.
    """
    require_columns(df, group_cols, name="panel")

    filled_counts: dict[str, int] = {}

    for col in cols:
        if col not in df.columns or col not in eligible_numeric:
            continue

        before_missing = int(df[col].isna().sum())
        group_medians = df.groupby(group_cols)[col].transform("median")
        df[col] = df[col].fillna(group_medians)
        after_missing = int(df[col].isna().sum())

        filled_counts[col] = before_missing - after_missing

    return filled_counts


def global_median_impute(
    df: pd.DataFrame, *, cols: list[str], eligible_numeric: list[str]
) -> dict[str, int]:
    """Fill selected columns using global column medians.

    This function mutates the input DataFrame in place.

    Args:
        df: Input panel DataFrame.
        cols: Columns requested for median imputation.
        eligible_numeric: Columns allowed to be imputed.

    Returns:
        Dictionary mapping each filled column to the number of values filled.
    """
    filled_counts: dict[str, int] = {}

    for col in cols:
        if col not in df.columns or col not in eligible_numeric:
            continue

        before_missing = int(df[col].isna().sum())
        median_value = df[col].median(skipna=True)
        df[col] = df[col].fillna(median_value)
        after_missing = int(df[col].isna().sum())

        filled_counts[col] = before_missing - after_missing

    return filled_counts
