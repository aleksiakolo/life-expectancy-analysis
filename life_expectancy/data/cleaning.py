from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from life_expectancy.data.missingness import drop_features_by_missingness
from life_expectancy.data.panel import panel_output_summary
from life_expectancy.data.utils import prepare_panel_keys, require_columns

JoinKey = tuple[str, str]
DuplicatePolicy = Literal["keep_first", "drop_all", "mean_aggregate"]
Summary = dict[str, Any]


def clean_who(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, Summary]:
    """Clean the standardized WHO country-year panel using config settings.

    This function applies the WHO-specific cleaning strategy defined under
    `config["cleaning"]["who"]`. It assumes the input DataFrame has already been
    standardized, so country/year columns and the WHO life expectancy target use
    the configured standardized names.

    Args:
        df: Standardized WHO DataFrame.
        config: Full project configuration dictionary containing a `cleaning.who`
            section.

    Returns:
        Tuple containing:
            - Cleaned WHO DataFrame.
            - Summary dictionary with row counts, duplicate handling information,
              target-cleaning counts, missingness-based column drops, country
              coverage filtering details, and output shape/year range.

    Raises:
        KeyError: If required configured columns are missing.
        ValueError: If an unsupported duplicate policy or invalid missingness
            threshold is provided.
    """

    cfg = config["cleaning"]["who"]

    key = tuple(cfg.get("key", ["country", "year"]))
    country_col, year_col = key

    target_col = cfg.get("target_col", "life_expectancy_who")
    drop_missing_target = cfg.get("drop_missing_target", True)
    duplicate_policy = cfg.get("duplicate_policy", "keep_first")
    min_years_per_country = cfg.get("min_years_per_country", 5)
    bounds = tuple(cfg.get("life_expectancy_bounds", [0.0, 100.0]))
    clip = cfg.get("clip_life_expectancy", False)
    immun_clip = cfg.get("immunization_clip_0_100", True)
    negative_cols = cfg.get("negative_to_na_cols")
    missing_thresh = cfg.get("feature_missingness_drop_threshold")

    out = prepare_panel_keys(df, country_col, year_col)
    require_columns(out, [country_col, year_col, target_col], name="who_df")

    summary: Summary = {
        "input_rows": len(out),
        "input_cols": out.shape[1],
    }

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    out, s = handle_duplicates(
        out,
        country_col=country_col,
        year_col=year_col,
        duplicate_policy=duplicate_policy,
    )
    summary.update(s)

    out, s = clean_life_expectancy(
        out,
        target_col=target_col,
        bounds=bounds,
        clip=clip,
    )
    summary.update(s)

    if negative_cols:
        out = convert_negative_values_to_na(out, negative_cols)

    if immun_clip:
        out, s = clip_immunization_columns(out)
        summary.update(s)

    out, s = drop_missing_target_rows(
        out,
        target_col=target_col,
        enabled=drop_missing_target,
    )
    summary.update(s)

    out, s = filter_country_coverage(
        out,
        country_col=country_col,
        target_col=target_col,
        min_years=min_years_per_country,
    )
    summary.update(s)

    out, s = drop_features_by_missingness(
        out,
        threshold=missing_thresh,
        protected_cols={country_col, year_col, target_col},
    )
    summary.update(s)

    summary.update(panel_output_summary(out, year_col))

    return out, summary


def clean_wb(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, Summary]:
    """Clean the standardized World Bank country-year panel using config settings.

    This function applies the World Bank-specific cleaning strategy defined under
    `config["cleaning"]["wb"]`. It assumes the input DataFrame has already been
    standardized, so country/year columns and renamed indicators use the configured
    standardized names.

    Args:
        df: Standardized World Bank DataFrame.
        config: Full project configuration dictionary containing a `cleaning.wb`
            section.

    Returns:
        Tuple containing:
            - Cleaned World Bank DataFrame.
            - Summary dictionary with row counts, selected-column information,
              winsorization/log-transform settings, missingness-based column drops,
              duplicate counts, and output shape/year range.

    Raises:
        KeyError: If required configured columns are missing.
        ValueError: If an invalid missingness threshold is provided.
    """
    cfg = config["cleaning"]["wb"]

    key = tuple(cfg.get("key", ["country", "year"]))
    country_col, year_col = key

    indicator_subset = cfg.get("indicator_subset")
    winsorize_cols = cfg.get("winsorize_cols")
    winsorize_limits = tuple(cfg.get("winsorize_limits", [0.01, 0.99]))
    log_cols = cfg.get("log_transform_cols")
    log_eps = cfg.get("log_epsilon", 1e-9)
    missing_thresh = cfg.get("feature_missingness_drop_threshold")

    out = prepare_panel_keys(df, country_col, year_col)
    require_columns(out, [country_col, year_col], name="wb_df")

    summary: Summary = {
        "input_rows": len(out),
        "input_cols": out.shape[1],
        "duplicate_rows": int(
            out.duplicated(subset=[country_col, year_col], keep=False).sum()
        ),
    }

    if indicator_subset:
        out, s = apply_wb_subset(
            out,
            country_col=country_col,
            year_col=year_col,
            indicator_subset=indicator_subset,
        )
        summary.update(s)

    out = convert_wb_numeric_columns(out, country_col=country_col, year_col=year_col)

    if winsorize_cols:
        out = winsorize_columns(out, cols=winsorize_cols, limits=winsorize_limits)

    if log_cols:
        out = add_log_columns(out, cols=log_cols, epsilon=log_eps)

    out, s = drop_features_by_missingness(
        out,
        threshold=missing_thresh,
        protected_cols={country_col, year_col},
    )
    summary.update(s)

    summary.update(panel_output_summary(out, year_col))

    return out, summary


def handle_duplicates(
    df: pd.DataFrame,
    *,
    country_col: str,
    year_col: str,
    duplicate_policy: DuplicatePolicy,
) -> tuple[pd.DataFrame, Summary]:
    """Handle duplicate country-year rows.

    Args:
        df: Input panel DataFrame.
        country_col: Country column name.
        year_col: Year column name.
        duplicate_policy: Duplicate handling strategy.

    Returns:
        Deduplicated DataFrame and duplicate summary dictionary.
    """
    duplicate_mask = df.duplicated(subset=[country_col, year_col], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    summary: Summary = {"duplicate_rows": duplicate_rows}

    if duplicate_rows == 0:
        summary["rows_after_dedup"] = len(df)
        return df, summary

    if duplicate_policy == "keep_first":
        out = (
            df.sort_values([country_col, year_col])
            .drop_duplicates([country_col, year_col], keep="first")
            .copy()
        )
    elif duplicate_policy == "drop_all":
        out = df.loc[~duplicate_mask].copy()
    elif duplicate_policy == "mean_aggregate":
        out = mean_aggregate_duplicates(df, country_col, year_col)
    else:
        raise ValueError(f"Unknown duplicate_policy: {duplicate_policy}")

    summary["rows_after_dedup"] = len(out)
    return out, summary


def mean_aggregate_duplicates(
    df: pd.DataFrame, country_col: str, year_col: str
) -> pd.DataFrame:
    """Aggregate duplicate country-year rows using mean for numeric columns.

    Args:
        df: Input panel DataFrame.
        country_col: Country column name.
        year_col: Year column name.

    Returns:
        Aggregated DataFrame with one row per country-year.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    aggregate_rules: dict[str, str] = {}

    for col in df.columns:
        if col in {country_col, year_col}:
            continue
        aggregate_rules[col] = "mean" if col in numeric_cols else "first"

    return df.groupby([country_col, year_col], as_index=False).agg(aggregate_rules)


def clean_life_expectancy(
    df: pd.DataFrame, *, target_col: str, bounds: tuple[float, float], clip: bool
) -> tuple[pd.DataFrame, Summary]:
    """Clean invalid life expectancy values.

    Args:
        df: Input DataFrame.
        target_col: Life expectancy column.
        bounds: Inclusive lower and upper validity bounds.
        clip: Whether to clip invalid values instead of setting them to missing.

    Returns:
        Cleaned DataFrame and target-cleaning summary.
    """
    lower_bound, upper_bound = bounds
    out = df.copy()

    invalid_mask = (out[target_col] <= lower_bound) | (out[target_col] > upper_bound)
    invalid_mask = invalid_mask.fillna(False)

    if clip:
        out[target_col] = out[target_col].clip(lower=lower_bound, upper=upper_bound)
    else:
        out.loc[invalid_mask, target_col] = np.nan

    return out, {"life_expectancy_invalid_count": int(invalid_mask.sum())}


def convert_negative_values_to_na(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert negative values to missing values in selected columns.

    Args:
        df: Input DataFrame.
        cols: Columns to clean.

    Returns:
        DataFrame with negative values replaced by missing values.
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] < 0, col] = np.nan

    return out


def clip_immunization_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Summary]:
    """Clip immunization-like columns to the [0, 100] range.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with clipped immunization columns and summary dictionary.
    """
    out = df.copy()
    immunization_cols = [
        col for col in out.columns if looks_like_immunization_col(str(col))
    ]

    for col in immunization_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(0, 100)

    return out, {"immunization_cols_clipped": immunization_cols}


def drop_missing_target_rows(
    df: pd.DataFrame, *, target_col: str, enabled: bool
) -> tuple[pd.DataFrame, Summary]:
    """Drop rows with missing target values.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        enabled: Whether dropping is enabled.

    Returns:
        Filtered DataFrame and summary dictionary.
    """
    if not enabled:
        return df, {"rows_dropped_missing_target": 0}

    before_rows = len(df)
    out = df.loc[df[target_col].notna()].copy()

    return out, {"rows_dropped_missing_target": before_rows - len(out)}


def filter_country_coverage(
    df: pd.DataFrame, *, country_col: str, target_col: str, min_years: int
) -> tuple[pd.DataFrame, Summary]:
    """Filter countries with insufficient target observations.

    Args:
        df: Input panel DataFrame.
        country_col: Country column name.
        target_col: Target column used to count coverage.
        min_years: Minimum number of non-missing target observations.

    Returns:
        Filtered DataFrame and coverage summary dictionary.
    """
    if min_years <= 0:
        return df, {
            "rows_dropped_low_coverage": 0,
            "countries_kept": int(df[country_col].nunique()),
        }

    counts = df.groupby(country_col)[target_col].count()
    keep_countries = counts[counts >= min_years].index

    before_rows = len(df)
    out = df[df[country_col].isin(keep_countries)].copy()

    return out, {
        "rows_dropped_low_coverage": before_rows - len(out),
        "countries_kept": len(keep_countries),
    }


def apply_wb_subset(
    df: pd.DataFrame, *, country_col: str, year_col: str, indicator_subset: list[str]
) -> tuple[pd.DataFrame, Summary]:
    """Keep selected World Bank indicator columns and key metadata.

    Args:
        df: Input World Bank DataFrame.
        country_col: Country column name.
        year_col: Year column name.
        indicator_subset: Indicator columns to keep.

    Returns:
        Subset DataFrame and subset summary dictionary.
    """
    keep_cols = {country_col, year_col}

    metadata_cols = [
        "country_code",
        "region",
        "income_group",
        "Country Code",
        "Region",
        "IncomeGroup",
    ]

    for col in metadata_cols:
        if col in df.columns:
            keep_cols.add(col)

    keep_cols.update(indicator_subset)
    missing_requested_cols = [col for col in indicator_subset if col not in df.columns]

    out = df[[col for col in df.columns if col in keep_cols]].copy()

    return out, {
        "cols_after_subset": out.shape[1],
        "missing_requested_cols": missing_requested_cols,
    }


def convert_wb_numeric_columns(
    df: pd.DataFrame, *, country_col: str, year_col: str
) -> pd.DataFrame:
    """Convert World Bank indicator columns to numeric when possible.

    Args:
        df: Input World Bank DataFrame.
        country_col: Country column name.
        year_col: Year column name.

    Returns:
        DataFrame with numeric-like columns converted.
    """
    out = df.copy()
    categorical_cols = {
        country_col,
        year_col,
        "country_code",
        "region",
        "income_group",
        "Country Code",
        "Region",
        "IncomeGroup",
    }

    for col in out.columns:
        if col in categorical_cols:
            continue

        converted = pd.to_numeric(out[col], errors="coerce")

        if converted.notna().sum() > 0:
            out[col] = converted

    return out


def winsorize_columns(
    df: pd.DataFrame, *, cols: list[str], limits: tuple[float, float]
) -> pd.DataFrame:
    """Winsorize selected columns using quantile limits.

    Args:
        df: Input DataFrame.
        cols: Columns to winsorize.
        limits: Lower and upper quantiles.

    Returns:
        DataFrame with selected columns clipped.
    """
    out = df.copy()
    lower_quantile, upper_quantile = limits

    for col in cols:
        if col not in out.columns:
            continue

        values = pd.to_numeric(out[col], errors="coerce")
        lower = values.quantile(lower_quantile)
        upper = values.quantile(upper_quantile)
        out[col] = values.clip(lower=lower, upper=upper)

    return out


def add_log_columns(
    df: pd.DataFrame, *, cols: list[str], epsilon: float
) -> pd.DataFrame:
    """Add log-transformed versions of selected nonnegative columns.

    Args:
        df: Input DataFrame.
        cols: Columns to transform.
        epsilon: Small positive value added before taking the log.

    Returns:
        DataFrame with new `log_` columns.
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        values = pd.to_numeric(out[col], errors="coerce")
        values = values.where(values >= 0)
        out[f"log_{col}"] = np.log(values + epsilon)

    return out


def looks_like_immunization_col(col: str) -> bool:
    """Check whether a column name appears to represent immunization coverage.

    Args:
        col: Column name.

    Returns:
        True if the column appears immunization-related.
    """
    cleaned = col.lower().strip()
    keywords = ["polio", "diphtheria", "hepatitis", "measles", "immun", "vacc"]

    return any(keyword in cleaned for keyword in keywords)
