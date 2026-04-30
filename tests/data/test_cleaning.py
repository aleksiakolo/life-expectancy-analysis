import pandas as pd

from life_expectancy.data.cleaning import clean_wb, clean_who


def test_clean_who_basic() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "B"],
            "year": [2010, 2011, 2010],
            "life_expectancy_who": [70.0, 71.0, None],
            "polio": [101, 90, 80],
        }
    )

    config = {
        "cleaning": {
            "who": {
                "key": ["country", "year"],
                "target_col": "life_expectancy_who",
                "drop_missing_target": True,
                "duplicate_policy": "keep_first",
                "min_years_per_country": 1,
                "life_expectancy_bounds": [0.0, 100.0],
                "clip_life_expectancy": False,
                "immunization_clip_0_100": True,
                "negative_to_na_cols": [],
                "feature_missingness_drop_threshold": None,
            }
        }
    }

    clean, summary = clean_who(df, config)

    assert len(clean) == 2
    assert clean["polio"].max() == 100
    assert summary["rows_dropped_missing_target"] == 1


def test_clean_who_invalid_life_expectancy_to_na_then_drop() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2010, 2010],
            "life_expectancy_who": [70.0, 130.0],
        }
    )

    config = {
        "cleaning": {
            "who": {
                "key": ["country", "year"],
                "target_col": "life_expectancy_who",
                "drop_missing_target": True,
                "duplicate_policy": "keep_first",
                "min_years_per_country": 1,
                "life_expectancy_bounds": [0.0, 100.0],
                "clip_life_expectancy": False,
                "immunization_clip_0_100": False,
                "negative_to_na_cols": [],
                "feature_missingness_drop_threshold": None,
            }
        }
    }

    clean, summary = clean_who(df, config)

    assert len(clean) == 1
    assert summary["life_expectancy_invalid_count"] == 1


def test_clean_wb_basic_numeric_conversion() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2010, 2010],
            "region": ["X", "Y"],
            "income_group": ["Low", "High"],
            "gdp": ["1000", "2000"],
            "unemployment": ["5.5", "6.0"],
        }
    )

    config = {
        "cleaning": {
            "wb": {
                "key": ["country", "year"],
                "indicator_subset": None,
                "winsorize_cols": [],
                "winsorize_limits": [0.01, 0.99],
                "log_transform_cols": [],
                "log_epsilon": 1e-9,
                "feature_missingness_drop_threshold": None,
            }
        }
    }

    clean, summary = clean_wb(df, config)

    assert pd.api.types.is_numeric_dtype(clean["gdp"])
    assert pd.api.types.is_numeric_dtype(clean["unemployment"])
    assert summary["duplicate_rows"] == 0
