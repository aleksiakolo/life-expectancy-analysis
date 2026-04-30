import pandas as pd
import pytest

from life_expectancy.features.temporal import make_country_lag_features


def test_make_country_lag_features_basic() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "A", "A"],
            "year": [2010, 2011, 2012, 2013],
            "life_expectancy": [70.0, 71.0, 72.0, 73.0],
            "gdp": [100.0, 110.0, 120.0, 130.0],
        }
    )

    result = make_country_lag_features(
        df,
        target_col="life_expectancy",
        feature_cols=["gdp"],
        lags=[1],
        dropna_lagged=True,
    )

    assert "life_expectancy_lag1" in result.columns
    assert "gdp_lag1" in result.columns
    assert "life_expectancy_rollmean_3" in result.columns
    assert len(result) == 1
    assert result.iloc[0]["year"] == 2013


def test_make_country_lag_features_without_dropna() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A"],
            "year": [2010, 2011],
            "life_expectancy": [70.0, 71.0],
        }
    )

    result = make_country_lag_features(
        df,
        target_col="life_expectancy",
        lags=[1],
        dropna_lagged=False,
    )

    assert len(result) == 2
    assert pd.isna(result.loc[0, "life_expectancy_lag1"])
    assert result.loc[1, "life_expectancy_lag1"] == 70.0


def test_make_country_lag_features_separate_countries() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "year": [2010, 2011, 2010, 2011],
            "life_expectancy": [70.0, 71.0, 80.0, 81.0],
        }
    )

    result = make_country_lag_features(
        df,
        target_col="life_expectancy",
        lags=[1],
        dropna_lagged=False,
    )

    b_2011 = result[(result["country"] == "B") & (result["year"] == 2011)].iloc[0]

    assert b_2011["life_expectancy_lag1"] == 80.0


def test_make_country_lag_features_missing_column_raises() -> None:
    df = pd.DataFrame(
        {
            "country": ["A"],
            "year": [2010],
        }
    )

    with pytest.raises(KeyError):
        make_country_lag_features(df, target_col="life_expectancy")
