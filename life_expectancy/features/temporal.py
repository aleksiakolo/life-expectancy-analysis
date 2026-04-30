from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def make_country_lag_features(
    df: pd.DataFrame,
    *,
    target_col: str = "life_expectancy",
    country_col: str = "country",
    year_col: str = "year",
    feature_cols: Sequence[str] | None = None,
    lags: Sequence[int] = (1, 2, 3),
    rolling_windows: Sequence[int] = (3,),
    dropna_lagged: bool = True,
) -> pd.DataFrame:
    """Create flat lag and rolling-mean features by country.

    Args:
        df: Input country-year panel.
        target_col: Target column name.
        country_col: Country column name.
        year_col: Year column name.
        feature_cols: Feature columns to lag. If None, numeric-like non-ID
            columns are inferred.
        lags: Lag values to create.
        rolling_windows: Rolling windows for shifted target rolling means.
        dropna_lagged: Whether to drop rows with missing created lag features.

    Returns:
        DataFrame with added lag and rolling target features.

    Raises:
        KeyError: If required columns are missing.
    """
    require_columns(df, [country_col, year_col, target_col])

    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="raise").astype(int)
    out = out.sort_values([country_col, year_col]).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = infer_lag_feature_columns(
            out,
            target_col=target_col,
            country_col=country_col,
            year_col=year_col,
        )

    grouped = out.groupby(country_col, group_keys=False)
    created_cols: list[str] = []

    for col in [target_col, *feature_cols]:
        for lag in sorted(set(lags)):
            new_col = f"{col}_lag{lag}"
            out[new_col] = grouped[col].shift(lag)
            created_cols.append(new_col)

    for window in rolling_windows:
        new_col = f"{target_col}_rollmean_{window}"
        out[new_col] = grouped[target_col].transform(
            lambda series: series.shift(1).rolling(window).mean()
        )
        created_cols.append(new_col)

    if dropna_lagged:
        out = out.dropna(subset=created_cols).reset_index(drop=True)

    return out


def make_country_lag_features_from_config(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create temporal features using project configuration.

    Args:
        df: Input country-year panel.
        config: Full project configuration dictionary containing
            `features.temporal`.

    Returns:
        DataFrame with temporal lag features.
    """
    temporal_config = config.get("features", {}).get("temporal", {})

    return make_country_lag_features(
        df,
        target_col=temporal_config.get("target_col", "life_expectancy"),
        country_col=temporal_config.get("country_col", "country"),
        year_col=temporal_config.get("year_col", "year"),
        feature_cols=temporal_config.get("feature_cols"),
        lags=tuple(temporal_config.get("lags", [1, 2, 3])),
        rolling_windows=tuple(temporal_config.get("rolling_windows", [3])),
        dropna_lagged=temporal_config.get("dropna_lagged", True),
    )


def infer_lag_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str,
    country_col: str,
    year_col: str,
) -> list[str]:
    """Infer columns that are reasonable to lag.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        country_col: Country column name.
        year_col: Year column name.

    Returns:
        List of feature columns to lag.
    """
    excluded_cols = {
        target_col,
        country_col,
        "country_code",
        "region",
        "income_group",
        year_col,
    }

    return [
        col
        for col in df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
    ]


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate required columns.

    Args:
        df: DataFrame to validate.
        columns: Required columns.

    Raises:
        KeyError: If required columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
