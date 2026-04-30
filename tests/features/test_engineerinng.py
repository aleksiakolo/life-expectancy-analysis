import numpy as np
import pandas as pd
import pytest

from life_expectancy.features.feature_engineering import (
    add_interaction_features,
    add_log_features,
    add_missingness_flags,
    add_status_flag,
    prepare_numeric_model_frame,
)


def make_feature_config() -> dict:
    return {
        "features": {
            "target_col": "life_expectancy",
            "year_col": "year",
            "status_col": "status",
            "status_flag_col": "status_flag",
            "log_candidates": ["gdp", "population"],
            "missingness_flag_candidates": ["gdp", "schooling"],
            "interactions": [["schooling", "status_flag"]],
            "leakage_cols": ["life_expectancy_who", "life_expectancy_wb"],
            "drop_id_cols": ["country_code"],
        }
    }


def test_add_log_features() -> None:
    df = pd.DataFrame({"gdp": [0, 9, -5], "other": [1, 2, 3]})

    result = add_log_features(df, cols=["gdp", "missing_col"])

    assert "gdp_log1p" in result.columns
    assert result["gdp_log1p"].tolist() == [0.0, np.log1p(9), 0.0]


def test_add_status_flag() -> None:
    df = pd.DataFrame({"status": ["Developed", "Developing", "developed economy"]})

    result = add_status_flag(df, status_col="status", output_col="status_flag")

    assert result["status_flag"].tolist() == [1, 0, 1]


def test_add_status_flag_does_not_overwrite_existing() -> None:
    df = pd.DataFrame({"status": ["Developed"], "status_flag": [0]})

    result = add_status_flag(df, status_col="status", output_col="status_flag")

    assert result.loc[0, "status_flag"] == 0


def test_add_missingness_flags() -> None:
    df = pd.DataFrame({"gdp": [1.0, None], "schooling": [None, 12.0]})

    result = add_missingness_flags(df, cols=["gdp", "schooling"])

    assert result["gdp_missing_flag"].tolist() == [0, 1]
    assert result["schooling_missing_flag"].tolist() == [1, 0]


def test_add_interaction_features() -> None:
    df = pd.DataFrame({"schooling": [10, 12], "status_flag": [1, 0]})

    result = add_interaction_features(
        df,
        interaction_pairs=[["schooling", "status_flag"], ["bad_pair"]],
    )

    assert "schooling__x__status_flag" in result.columns
    assert result["schooling__x__status_flag"].tolist() == [10, 0]


def test_prepare_numeric_model_frame() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "B"],
            "country_code": ["AAA", "BBB"],
            "year": ["2010", "2011"],
            "status": ["Developed", "Developing"],
            "life_expectancy_who": [70.0, 80.0],
            "life_expectancy_wb": [72.0, 82.0],
            "life_expectancy": [71.0, 81.0],
            "gdp": [1000.0, None],
            "population": [10.0, 20.0],
            "schooling": [12.0, 10.0],
        }
    )

    result = prepare_numeric_model_frame(df, make_feature_config())

    assert "life_expectancy" in result.columns
    assert "country_code" not in result.columns
    assert "status" not in result.columns
    assert "life_expectancy_who" not in result.columns
    assert "life_expectancy_wb" not in result.columns
    assert "status_flag" in result.columns
    assert "gdp_log1p" in result.columns
    assert "gdp_missing_flag" in result.columns
    assert "schooling__x__status_flag" in result.columns
    assert pd.api.types.is_integer_dtype(result["year"])


def test_prepare_numeric_model_frame_missing_target_raises() -> None:
    df = pd.DataFrame({"year": [2010], "gdp": [100.0]})

    with pytest.raises(KeyError):
        prepare_numeric_model_frame(df, make_feature_config())
