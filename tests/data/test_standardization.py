import pandas as pd
import pytest

from life_expectancy.data.standardization import standardize


def test_standardize_who_panel() -> None:
    df = pd.DataFrame(
        {
            "Country": [" Albania "],
            "Year": ["2015"],
            "Life expectancy": [78.0],
        }
    )

    source_config = {
        "name": "who",
        "country_col": "Country",
        "year_col": "Year",
        "column_renames": {
            "life_expectancy": "life_expectancy_who",
        },
    }

    result = standardize(df, source_config)

    assert "country" in result.columns
    assert "year" in result.columns
    assert "life_expectancy_who" in result.columns
    assert result.loc[0, "country"] == "Albania"
    assert result.loc[0, "year"] == 2015


def test_standardize_wb_panel() -> None:
    df = pd.DataFrame(
        {
            "Country Name": ["Japan"],
            "Year": [2015],
            "Life Expectancy World Bank": [83.0],
        }
    )

    source_config = {
        "name": "wb",
        "country_col": "Country Name",
        "year_col": "Year",
        "column_renames": {
            "life_expectancy_world_bank": "life_expectancy_wb",
        },
    }

    result = standardize(df, source_config)

    assert result.loc[0, "country"] == "Japan"
    assert "life_expectancy_wb" in result.columns


def test_standardize_missing_key_column_raises() -> None:
    df = pd.DataFrame({"Country": ["Albania"]})

    source_config = {
        "name": "who",
        "country_col": "Country",
        "year_col": "Year",
        "column_renames": {},
    }

    with pytest.raises(KeyError):
        standardize(df, source_config)


def test_standardize_unsupported_source_raises() -> None:
    df = pd.DataFrame({"country": ["Albania"], "year": [2015]})

    with pytest.raises(ValueError):
        standardize(df, {"name": "unknown"})
