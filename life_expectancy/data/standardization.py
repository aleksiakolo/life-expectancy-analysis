from __future__ import annotations

from typing import Any

import pandas as pd

from life_expectancy.data.utils import (
    clean_column_name,
    clean_string_series,
    to_nullable_year,
    validate_columns,
)

SourceConfig = dict[str, Any]


def standardize(df: pd.DataFrame, source_config: SourceConfig) -> pd.DataFrame:
    """Standardize a raw dataset using configuration metadata.

    Args:
        df: Raw input DataFrame.
        source_config: Configuration for one data source. Expected keys include
            `name`, and optionally `country_col`, `year_col`, and `column_renames`.

    Returns:
        Standardized DataFrame with consistent column names and key columns.

    Raises:
        ValueError: If the configured source name is unsupported.
    """
    source_name = source_config["name"]
    column_renames = source_config.get("column_renames", {})

    if source_name in {"who", "wb"}:
        return standardize_panel_source(
            df=df,
            country_col=source_config.get("country_col", "Country"),
            year_col=source_config.get("year_col", "Year"),
            column_renames=column_renames,
        )

    if source_name == "wdi":
        return standardize_wdi(df=df, column_renames=column_renames)

    raise ValueError(f"Unsupported source name: {source_name}")


def standardize_panel_source(
    df: pd.DataFrame, country_col: str, year_col: str, column_renames: dict[str, str]
) -> pd.DataFrame:
    """Standardize a country-year panel dataset.

    Args:
        df: Raw country-year panel DataFrame.
        country_col: Raw column name containing country names.
        year_col: Raw column name containing years.
        column_renames: Mapping from cleaned column names to final names.

    Returns:
        Standardized panel DataFrame.

    Raises:
        KeyError: If the configured country or year column is missing.
    """
    validate_columns(df, required_columns=[country_col, year_col])

    out = df.copy()
    cleaned_column_map = {col: clean_column_name(col) for col in out.columns}
    out = out.rename(columns=cleaned_column_map)

    cleaned_country_col = clean_column_name(country_col)
    cleaned_year_col = clean_column_name(year_col)

    out = out.rename(
        columns={
            cleaned_country_col: "country",
            cleaned_year_col: "year",
        }
    )

    out = out.rename(columns=column_renames)

    out["country"] = clean_string_series(out["country"])
    out["year"] = to_nullable_year(out["year"])

    return out


def standardize_wdi(df: pd.DataFrame, column_renames: dict[str, str]) -> pd.DataFrame:
    """Standardize a World Development Indicators export.

    Args:
        df: Raw WDI export DataFrame.
        column_renames: Mapping from cleaned column names to final names.

    Returns:
        Standardized WDI DataFrame.

    Raises:
        KeyError: If required WDI identifier columns are missing.
    """
    required_columns = [
        "Country Name",
        "Country Code",
        "Indicator Name",
        "Indicator Code",
    ]
    validate_columns(df, required_columns=required_columns)

    out = df.copy()
    out = out.loc[:, ~out.columns.astype(str).str.contains(r"^Unnamed")]

    cleaned_column_map = {col: clean_column_name(col) for col in out.columns}
    out = out.rename(columns=cleaned_column_map)

    default_renames = {
        "country_name": "country",
        "country_code": "country_code",
        "indicator_name": "indicator_name",
        "indicator_code": "indicator_code",
    }

    out = out.rename(columns=default_renames)
    out = out.rename(columns=column_renames)

    for col in ["country", "country_code", "indicator_name", "indicator_code"]:
        out[col] = clean_string_series(out[col])

    return out
