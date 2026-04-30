import pandas as pd

from life_expectancy.data.imputation import impute_panel


def test_impute_panel_global_median() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "B"],
            "year": [2010, 2011, 2010],
            "life_expectancy": [70.0, 71.0, 80.0],
            "gdp": [100.0, None, 300.0],
        }
    )

    config = {
        "imputation": {
            "key": ["country", "year"],
            "target_col": "life_expectancy",
            "numeric_only": True,
            "interpolate_cols": [],
            "group_cols": [],
            "group_median_cols": [],
            "median_impute_cols": ["gdp"],
        }
    }

    result, summary = impute_panel(df, config)

    assert result["gdp"].isna().sum() == 0
    assert result.loc[1, "gdp"] == 200.0
    assert summary["median_filled_counts"]["gdp"] == 1


def test_impute_panel_does_not_impute_target() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2010, 2010],
            "life_expectancy": [70.0, None],
            "gdp": [100.0, None],
        }
    )

    config = {
        "imputation": {
            "key": ["country", "year"],
            "target_col": "life_expectancy",
            "numeric_only": True,
            "interpolate_cols": [],
            "group_cols": [],
            "group_median_cols": [],
            "median_impute_cols": ["gdp"],
        }
    }

    result, summary = impute_panel(df, config)

    assert result["life_expectancy"].isna().sum() == 1
    assert summary["target_missing_count_after"] == 1
