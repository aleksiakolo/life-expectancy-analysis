import pandas as pd
import pytest

from life_expectancy.data.missingness import (
    drop_features_by_missingness,
    missingness_by_country,
    missingness_by_year,
    missingness_table,
)


def test_missingness_table() -> None:
    df = pd.DataFrame(
        {
            "a": [1, None, 3],
            "b": [None, None, 3],
        }
    )

    result = missingness_table(df)

    assert result.loc[0, "column"] == "b"
    assert result.loc[0, "missing_fraction"] == 0.6667
    assert result.loc[0, "missing_count"] == 2
    assert result.loc[0, "non_missing_count"] == 1


def test_missingness_by_year() -> None:
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2021],
            "gdp": [1.0, None, None],
        }
    )

    result = missingness_by_year(df, cols=["gdp"])

    assert result.loc[0, "year"] == 2020
    assert result.loc[0, "gdp_missing_fraction"] == 0.5
    assert result.loc[1, "gdp_missing_fraction"] == 1.0


def test_missingness_by_country() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "B"],
            "gdp": [1.0, None, None],
        }
    )

    result = missingness_by_country(df, cols=["gdp"], top_n=None)

    assert result.loc[0, "country"] == "B"
    assert result.loc[0, "gdp_missing_fraction"] == 1.0


def test_missingness_by_year_missing_column_raises() -> None:
    df = pd.DataFrame({"year": [2020]})

    with pytest.raises(KeyError):
        missingness_by_year(df, cols=["gdp"])


def test_drop_features_by_missingness_none() -> None:
    df = pd.DataFrame({"country": ["A"], "gdp": [None]})

    result, summary = drop_features_by_missingness(
        df,
        threshold=None,
        protected_cols={"country"},
    )

    assert result.equals(df)
    assert summary["n_dropped_feature_cols"] == 0


def test_drop_features_by_missingness_drops_unprotected() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "B", "C"],
            "gdp": [None, None, 3],
            "year": [2020, 2020, 2020],
        }
    )

    result, summary = drop_features_by_missingness(
        df,
        threshold=0.5,
        protected_cols={"country", "year"},
    )

    assert "gdp" not in result.columns
    assert summary["dropped_feature_cols"] == ["gdp"]


def test_drop_features_by_missingness_invalid_threshold() -> None:
    df = pd.DataFrame({"a": [1, None]})

    with pytest.raises(ValueError):
        drop_features_by_missingness(
            df,
            threshold=1.5,
            protected_cols=set(),
        )
