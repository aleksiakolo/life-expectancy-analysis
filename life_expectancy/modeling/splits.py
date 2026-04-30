from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

SplitResult = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, "SplitInfo"]
Summary = dict[str, Any]


@dataclass(frozen=True)
class SplitInfo:
    """Metadata describing a train/test split.

    Attributes:
        split_type: Name of the split strategy.
        n_train: Number of training rows.
        n_test: Number of test rows.
        extra: Additional split-specific metadata.
    """

    split_type: str
    n_train: int
    n_test: int
    extra: Summary


def filter_countries_min_years(
    df: pd.DataFrame,
    *,
    country_col: str = "country",
    year_col: str = "year",
    min_years: int = 5,
) -> pd.DataFrame:
    """Keep countries with at least a minimum number of distinct years.

    Args:
        df: Input country-year panel.
        country_col: Country column name.
        year_col: Year column name.
        min_years: Minimum number of distinct years required per country.

    Returns:
        Filtered DataFrame containing only countries with enough year coverage.

    Raises:
        KeyError: If required columns are missing.
    """
    require_columns(df, [country_col, year_col])

    year_counts = df.groupby(country_col)[year_col].nunique()
    keep_countries = year_counts[year_counts >= min_years].index

    return df[df[country_col].isin(keep_countries)].copy()


def split_xy(
    df: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into predictors and target.

    Args:
        df: Input modeling DataFrame.
        target_col: Target column name.

    Returns:
        Tuple containing predictor DataFrame and target Series.

    Raises:
        KeyError: If the target column is missing.
    """
    require_columns(df, [target_col])

    x = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return x, y


def make_random_split(
    df: pd.DataFrame,
    *,
    target_col: str,
    test_size: float = 0.2,
    seed: int = 42,
    dropna_target: bool = True,
) -> SplitResult:
    """Create a random train/test split.

    Args:
        df: Input modeling DataFrame.
        target_col: Target column name.
        test_size: Fraction of rows to use for the test set.
        seed: Random seed used by scikit-learn.
        dropna_target: Whether to drop rows with missing target values.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test, and split metadata.

    Raises:
        KeyError: If the target column is missing.
    """
    data = df.copy()
    require_columns(data, [target_col])

    if dropna_target:
        data = data.dropna(subset=[target_col])

    x, y = split_xy(data, target_col=target_col)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
    )

    info = SplitInfo(
        split_type="random",
        n_train=len(x_train),
        n_test=len(x_test),
        extra={
            "test_size": test_size,
            "seed": seed,
            "dropna_target": dropna_target,
        },
    )

    return x_train, x_test, y_train, y_test, info


def make_random_split_from_config(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> SplitResult:
    """Create a random split using modeling split configuration.

    Args:
        df: Input modeling DataFrame.
        config: Full project configuration dictionary containing `modeling.split`.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test, and split metadata.
    """
    split_config = config.get("modeling", {}).get("split", {})

    return make_random_split(
        df,
        target_col=split_config.get("target_col", "life_expectancy"),
        test_size=split_config.get("test_size", 0.2),
        seed=split_config.get("seed", 42),
        dropna_target=split_config.get("dropna_target", True),
    )


def make_time_split(
    df: pd.DataFrame,
    *,
    target_col: str,
    year_col: str = "year",
    test_years: int = 3,
    dropna_target: bool = True,
) -> SplitResult:
    """Create a time-aware train/test split.

    The test set contains the last `test_years` years. All earlier rows are used
    for training.

    Args:
        df: Input modeling DataFrame.
        target_col: Target column name.
        year_col: Year column name.
        test_years: Number of latest years to reserve for testing.
        dropna_target: Whether to drop rows with missing target values.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test, and split metadata.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If years are non-numeric or the split is empty.
    """
    data = df.copy()
    require_columns(data, [target_col, year_col])

    if test_years <= 0:
        raise ValueError("test_years must be positive.")

    if dropna_target:
        data = data.dropna(subset=[target_col])

    data[year_col] = coerce_year_to_int(data[year_col], year_col=year_col)

    max_year = int(data[year_col].max())
    cutoff = max_year - test_years + 1

    train_df = data[data[year_col] < cutoff].copy()
    test_df = data[data[year_col] >= cutoff].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Time split produced an empty train or test set. "
            "Use fewer test years or check the year range."
        )

    x_train, y_train = split_xy(train_df, target_col=target_col)
    x_test, y_test = split_xy(test_df, target_col=target_col)

    info = SplitInfo(
        split_type="time",
        n_train=len(x_train),
        n_test=len(x_test),
        extra={
            "year_col": year_col,
            "test_years": test_years,
            "max_year": max_year,
            "cutoff": cutoff,
            "test_year_values": sorted(test_df[year_col].unique().tolist()),
            "dropna_target": dropna_target,
        },
    )

    return x_train, x_test, y_train, y_test, info


def make_time_split_from_config(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> SplitResult:
    """Create a time-aware split using modeling split configuration.

    Args:
        df: Input modeling DataFrame.
        config: Full project configuration dictionary containing `modeling.split`.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test, and split metadata.
    """
    split_config = config.get("modeling", {}).get("split", {})

    return make_time_split(
        df,
        target_col=split_config.get("target_col", "life_expectancy"),
        year_col=split_config.get("year_col", "year"),
        test_years=split_config.get("test_years", 3),
        dropna_target=split_config.get("dropna_target", True),
    )


def coerce_year_to_int(series: pd.Series, *, year_col: str) -> pd.Series:
    """Convert a year column to integer dtype.

    Args:
        series: Year-like Series.
        year_col: Name of the year column, used in error messages.

    Returns:
        Integer year Series.

    Raises:
        ValueError: If any year value is missing or non-numeric.
    """
    years = pd.to_numeric(series, errors="coerce")

    if years.isna().any():
        bad_count = int(years.isna().sum())
        raise ValueError(
            f"{bad_count} rows have non-numeric or missing {year_col!r} values."
        )

    return years.astype(int)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that required columns exist.

    Args:
        df: DataFrame to validate.
        columns: Required column names.

    Raises:
        KeyError: If any required columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
