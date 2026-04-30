from __future__ import annotations

from typing import Any

import pandas as pd

from life_expectancy.data.utils import require_columns

Summary = dict[str, Any]


def missingness_table(
    df: pd.DataFrame,
    *,
    sort_desc: bool = True,
    round_to: int = 4,
) -> pd.DataFrame:
    """Create an overall missingness table for each column.

    Args:
        df: Input DataFrame.
        sort_desc: Whether to sort by missing fraction in descending order.
        round_to: Number of decimal places for missing fractions.

    Returns:
        DataFrame with column name, missing fraction, missing count, and
        non-missing count.
    """
    row_count = len(df)
    missing_fraction = df.isna().mean()
    missing_count = df.isna().sum()
    non_missing_count = row_count - missing_count

    out = pd.DataFrame(
        {
            "column": missing_fraction.index,
            "missing_fraction": missing_fraction.values,
            "missing_count": missing_count.values,
            "non_missing_count": non_missing_count.values,
        }
    )

    out["missing_fraction"] = out["missing_fraction"].round(round_to)

    if sort_desc:
        out = out.sort_values("missing_fraction", ascending=False)

    return out.reset_index(drop=True)


def missingness_by_year(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
    cols: list[str],
    round_to: int = 4,
) -> pd.DataFrame:
    """Calculate missingness fractions by year for selected columns.

    Args:
        df: Input DataFrame.
        year_col: Year column name.
        cols: Columns for which to calculate missingness.
        round_to: Number of decimal places for missing fractions.

    Returns:
        DataFrame with one row per year and missingness fraction columns.

    Raises:
        KeyError: If the year column or selected columns are missing.
    """
    require_columns(df, [year_col, *cols], name="df")

    rows: list[dict[str, Any]] = []

    for year, group in df.groupby(year_col):
        row: dict[str, Any] = {"year": year, "n_rows": len(group)}

        for col in cols:
            row[f"{col}_missing_fraction"] = float(group[col].isna().mean())

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    for col in cols:
        out[f"{col}_missing_fraction"] = out[f"{col}_missing_fraction"].round(round_to)

    return out


def missingness_by_country(
    df: pd.DataFrame,
    *,
    country_col: str = "country",
    cols: list[str],
    top_n: int | None = 15,
    round_to: int = 4,
) -> pd.DataFrame:
    """Calculate missingness fractions by country for selected columns.

    Args:
        df: Input DataFrame.
        country_col: Country column name.
        cols: Columns for which to calculate missingness.
        top_n: Optional number of rows to keep after sorting by missingness.
        round_to: Number of decimal places for missing fractions.

    Returns:
        DataFrame with one row per country and missingness fraction columns.

    Raises:
        KeyError: If the country column or selected columns are missing.
    """
    require_columns(df, [country_col, *cols], name="df")

    rows: list[dict[str, Any]] = []

    for country, group in df.groupby(country_col):
        row: dict[str, Any] = {"country": country, "n_rows": len(group)}

        for col in cols:
            row[f"{col}_missing_fraction"] = float(group[col].isna().mean())

        rows.append(row)

    sort_col = f"{cols[0]}_missing_fraction"
    out = (
        pd.DataFrame(rows).sort_values(sort_col, ascending=False).reset_index(drop=True)
    )

    for col in cols:
        out[f"{col}_missing_fraction"] = out[f"{col}_missing_fraction"].round(round_to)

    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)

    return out


def drop_features_by_missingness(
    df: pd.DataFrame,
    *,
    threshold: float | None,
    protected_cols: set[str],
) -> tuple[pd.DataFrame, Summary]:
    """Drop columns whose missingness fraction exceeds a threshold.

    Args:
        df: Input DataFrame.
        threshold: Missingness threshold in the interval (0, 1). If None, no
            columns are dropped.
        protected_cols: Columns that should never be dropped.

    Returns:
        Tuple containing:
            - DataFrame with high-missingness columns removed.
            - Summary dictionary listing dropped columns and the number dropped.

    Raises:
        ValueError: If threshold is not None and is not in the interval (0, 1).
    """
    if threshold is None:
        return df, {"dropped_feature_cols": [], "n_dropped_feature_cols": 0}

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "feature_missingness_drop_threshold must be a fraction in (0, 1)."
        )

    missing_fraction = df.isna().mean()
    drop_cols = [
        col
        for col, fraction in missing_fraction.items()
        if fraction > threshold and col not in protected_cols
    ]

    out = df.drop(columns=drop_cols)

    return out, {
        "dropped_feature_cols": drop_cols,
        "n_dropped_feature_cols": len(drop_cols),
    }
