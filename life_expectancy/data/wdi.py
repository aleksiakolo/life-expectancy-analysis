from __future__ import annotations

from typing import Any

import pandas as pd

from life_expectancy.data.utils import coerce_year_to_int, require_columns

Summary = dict[str, Any]


def pivot_wdi(
    wdi_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Summary]:
    """Pivot standardized WDI data into a country-year panel.

    This function reads its options from `config["wdi"]`. It assumes the input WDI
    DataFrame has already been standardized and contains country identifier columns,
    an indicator-code column, and wide year columns such as `1960`, `1961`, etc.

    Args:
        wdi_df: Standardized WDI DataFrame.
        config: Full project configuration dictionary containing a `wdi` section.

    Returns:
        Tuple containing:
            - Pivoted WDI panel with columns `country`, `year`, and one column per
              selected indicator code.
            - Summary dictionary describing filtering, dropped rows, and output shape.

    Raises:
        KeyError: If required WDI columns are missing.
        ValueError: If no year columns are detected.
    """
    wdi_config = config["wdi"]

    indicator_codes = wdi_config.get("indicator_codes")
    indicator_code_col = wdi_config.get("indicator_code_col", "indicator_code")
    country_col = wdi_config.get("country_col", "country")
    country_code_col = wdi_config.get("country_code_col", "country_code")
    value_name = wdi_config.get("value_name", "value")
    year_min = wdi_config.get("year_min")
    year_max = wdi_config.get("year_max")
    wide_prefix = wdi_config.get("wide_prefix", "")
    drop_all_nan_rows = wdi_config.get("drop_all_nan_rows", True)

    required_columns = [country_col, country_code_col, indicator_code_col]
    require_columns(wdi_df, required_columns, name="wdi_df")

    df = wdi_df.copy()
    year_cols = find_year_columns(df)

    if not year_cols:
        raise ValueError(
            "No year columns detected in WDI data. Expected columns like "
            "'1960', '1961', etc."
        )

    if indicator_codes is not None:
        df = df[df[indicator_code_col].isin(indicator_codes)].copy()

    summary: Summary = {
        "input_rows": len(wdi_df),
        "rows_after_indicator_filter": len(df),
        "n_year_columns": len(year_cols),
        "indicator_codes": indicator_codes,
    }

    long_df = wdi_to_long(
        df,
        year_cols=year_cols,
        country_col=country_col,
        country_code_col=country_code_col,
        indicator_code_col=indicator_code_col,
        value_name=value_name,
    )

    long_df = filter_year_range(
        long_df,
        year_min=year_min,
        year_max=year_max,
    )

    panel = wdi_long_to_panel(
        long_df,
        country_col=country_col,
        indicator_code_col=indicator_code_col,
        value_name=value_name,
    )

    panel = prefix_indicator_columns(
        panel,
        country_col=country_col,
        year_col="year",
        prefix=wide_prefix,
    )

    if drop_all_nan_rows:
        panel, drop_summary = drop_all_nan_indicator_rows(
            panel,
            country_col=country_col,
            year_col="year",
        )
        summary.update(drop_summary)
    else:
        summary["rows_dropped_all_nan"] = 0

    summary["output_rows"] = len(panel)
    summary["output_cols"] = panel.shape[1]

    return panel, summary


def wdi_to_long(
    df: pd.DataFrame,
    *,
    year_cols: list[str],
    country_col: str,
    country_code_col: str,
    indicator_code_col: str,
    value_name: str,
) -> pd.DataFrame:
    """Convert standardized WDI data from wide year columns to long format.

    Args:
        df: Standardized WDI DataFrame.
        year_cols: Year columns to melt.
        country_col: Country column name.
        country_code_col: Country code column name.
        indicator_code_col: Indicator code column name.
        value_name: Name for the melted value column.

    Returns:
        Long WDI DataFrame with columns for country, country code, indicator code,
        year, and value.
    """
    long_df = df.melt(
        id_vars=[country_col, country_code_col, indicator_code_col],
        value_vars=year_cols,
        var_name="year",
        value_name=value_name,
    )

    long_df["year"] = coerce_year_to_int(long_df["year"])
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")

    return long_df


def filter_year_range(
    df: pd.DataFrame, *, year_min: int | None, year_max: int | None
) -> pd.DataFrame:
    """Filter a WDI long-format DataFrame by year range.

    Args:
        df: Long WDI DataFrame containing a `year` column.
        year_min: Optional minimum year to keep.
        year_max: Optional maximum year to keep.

    Returns:
        Filtered DataFrame.
    """
    out = df.copy()

    if year_min is not None:
        out = out[out["year"] >= year_min]

    if year_max is not None:
        out = out[out["year"] <= year_max]

    return out


def wdi_long_to_panel(
    long_df: pd.DataFrame,
    *,
    country_col: str,
    indicator_code_col: str,
    value_name: str,
) -> pd.DataFrame:
    """Pivot long WDI data into a country-year panel.

    Args:
        long_df: Long WDI DataFrame.
        country_col: Country column name.
        indicator_code_col: Indicator code column name.
        value_name: Column containing indicator values.

    Returns:
        Country-year panel with one column per indicator code.
    """
    panel = long_df.pivot_table(
        index=[country_col, "year"],
        columns=indicator_code_col,
        values=value_name,
        aggfunc="first",
    ).reset_index()

    panel.columns = [str(col) for col in panel.columns]

    return panel


def prefix_indicator_columns(
    panel: pd.DataFrame, *, country_col: str, year_col: str, prefix: str
) -> pd.DataFrame:
    """Add a prefix to WDI indicator columns.

    Args:
        panel: Pivoted WDI panel.
        country_col: Country column name.
        year_col: Year column name.
        prefix: Prefix to add to indicator columns.

    Returns:
        Panel with renamed indicator columns.
    """
    if not prefix:
        return panel

    rename_map = {
        col: f"{prefix}{col}"
        for col in panel.columns
        if col not in {country_col, year_col}
    }

    return panel.rename(columns=rename_map)


def drop_all_nan_indicator_rows(
    panel: pd.DataFrame, *, country_col: str, year_col: str
) -> tuple[pd.DataFrame, Summary]:
    """Drop panel rows where all indicator values are missing.

    Args:
        panel: Pivoted WDI panel.
        country_col: Country column name.
        year_col: Year column name.

    Returns:
        Filtered panel and summary dictionary.
    """
    value_cols = [col for col in panel.columns if col not in {country_col, year_col}]
    before_rows = len(panel)
    out = panel.dropna(subset=value_cols, how="all")

    return out, {"rows_dropped_all_nan": before_rows - len(out)}


def find_year_columns(df: pd.DataFrame) -> list[str]:
    """Find WDI year columns.

    Args:
        df: WDI DataFrame with year columns represented as four-digit strings.

    Returns:
        List of columns that look like years.
    """
    return [str(col) for col in df.columns if is_year_column(col)]


def is_year_column(col: object) -> bool:
    """Check whether a column name looks like a four-digit year.

    Args:
        col: Column name.

    Returns:
        True if the column looks like a year between 1950 and 2050.
    """
    col_str = str(col).strip()

    if len(col_str) != 4:
        return False

    try:
        year = int(col_str)
    except ValueError:
        return False

    return 1950 <= year <= 2050
