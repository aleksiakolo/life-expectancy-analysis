from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitInfo:
    """Small container for split metadata."""
    split_type: str
    n_train: int
    n_test: int
    extra: Dict[str, object]


def filter_countries_min_years(df: pd.DataFrame, country_col: str = "country", year_col: str = "year", min_years: int = 5) -> pd.DataFrame:
    """Keep only countries that have at least `min_years` distinct years."""
    counts = df.groupby(country_col)[year_col].nunique()
    keep = counts[counts >= min_years].index
    return df[df[country_col].isin(keep)].copy()


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (all columns except target) and y (target)."""
    if target_col not in df.columns:
        raise KeyError(f"target_col='{target_col}' not found in df.columns")
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def make_random_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, seed: int = 42, dropna_target: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SplitInfo]:
    """Random train/test split."""
    data = df.copy()
    if dropna_target:
        data = data.dropna(subset=[target_col])

    X, y = split_xy(data, target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    info = SplitInfo(
        split_type="random",
        n_train=len(X_train),
        n_test=len(X_test),
        extra={"test_size": test_size, "seed": seed},
    )
    return X_train, X_test, y_train, y_test, info


def make_time_split(df: pd.DataFrame, target_col: str, year_col: str = "year", test_years: int = 3, dropna_target: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SplitInfo]:
    """
    Time-based holdout split:
      - Train = all years <= cutoff
      - Test  = last `test_years` years

    Example: if max year is 2015 and test_years=3,
             test years are 2013, 2014, 2015.
    """
    data = df.copy()
    if year_col not in data.columns:
        raise KeyError(f"year_col='{year_col}' not found in df.columns")

    if dropna_target:
        data = data.dropna(subset=[target_col])

    # ensure year is numeric
    years = pd.to_numeric(data[year_col], errors="coerce")
    if years.isna().any():
        bad_n = int(years.isna().sum())
        raise ValueError(f"{bad_n} rows have non-numeric '{year_col}'. Fix before time split.")
    data[year_col] = years.astype(int)

    max_year = int(data[year_col].max())
    cutoff = max_year - (test_years - 1)

    train_df = data[data[year_col] < cutoff].copy()
    test_df = data[data[year_col] >= cutoff].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError(
            "Time split produced empty train or test. "
            "Try smaller test_years or check your year range."
        )

    X_train, y_train = split_xy(train_df, target_col)
    X_test, y_test = split_xy(test_df, target_col)

    info = SplitInfo(
        split_type="time",
        n_train=len(X_train),
        n_test=len(X_test),
        extra={"year_col": year_col, "test_years": test_years, "max_year": max_year, "cutoff": cutoff},
    )
    return X_train, X_test, y_train, y_test, info
