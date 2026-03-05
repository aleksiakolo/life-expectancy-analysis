from __future__ import annotations
from typing import Sequence
import pandas as pd


def make_country_lag_features(
    df: pd.DataFrame,
    target_col: str = "life_expectancy_final",
    country_col: str = "country",
    year_col: str = "year",
    feature_cols: Sequence[str] | None = None,
    lags: Sequence[int] = (1, 2, 3),
    dropna_lagged: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    if country_col not in out.columns:
        raise KeyError(f"{country_col!r} not found in dataframe")
    if year_col not in out.columns:
        raise KeyError(f"{year_col!r} not found in dataframe")
    if target_col not in out.columns:
        raise KeyError(f"{target_col!r} not found in dataframe")

    out[year_col] = pd.to_numeric(out[year_col], errors="raise").astype(int)
    out = out.sort_values([country_col, year_col]).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [
            c for c in out.columns
            if c not in {target_col, country_col, "country_code", "region", "income_group", year_col}
        ]

    grouped = out.groupby(country_col, group_keys=False)

    created_cols: list[str] = []

    lag_sources = [target_col, *feature_cols]
    for col in lag_sources:
        for lag in sorted(set(lags)):
            new_col = f"{col}_lag{lag}"
            out[new_col] = grouped[col].shift(lag)
            created_cols.append(new_col)

    out[f"{target_col}_rollmean_3"] = grouped[target_col].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    created_cols.append(f"{target_col}_rollmean_3")

    if dropna_lagged:
        out = out.dropna(subset=created_cols).reset_index(drop=True)

    return out