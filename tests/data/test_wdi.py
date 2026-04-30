import pandas as pd
import pytest

from life_expectancy.data.wdi import (
    drop_all_nan_indicator_rows,
    filter_year_range,
    find_year_columns,
    is_year_column,
    pivot_wdi,
    prefix_indicator_columns,
    wdi_long_to_panel,
    wdi_to_long,
)


def test_is_year_column() -> None:
    assert is_year_column("2000")
    assert is_year_column(2015)
    assert not is_year_column("1800")
    assert not is_year_column("year")


def test_find_year_columns() -> None:
    df = pd.DataFrame(columns=["country", "1960", "2000", "not_year"])

    assert find_year_columns(df) == ["1960", "2000"]


def test_wdi_to_long() -> None:
    df = pd.DataFrame(
        {
            "country": ["Albania"],
            "country_code": ["ALB"],
            "indicator_code": ["SP.DYN.LE00.IN"],
            "2000": ["75.0"],
            "2001": ["76.0"],
        }
    )

    result = wdi_to_long(
        df,
        year_cols=["2000", "2001"],
        country_col="country",
        country_code_col="country_code",
        indicator_code_col="indicator_code",
        value_name="value",
    )

    assert result.shape[0] == 2
    assert str(result["year"].dtype) == "Int64"
    assert result["value"].tolist() == [75.0, 76.0]


def test_filter_year_range() -> None:
    df = pd.DataFrame({"year": [1999, 2000, 2001], "value": [1, 2, 3]})

    result = filter_year_range(df, year_min=2000, year_max=2001)

    assert result["year"].tolist() == [2000, 2001]


def test_wdi_long_to_panel() -> None:
    long_df = pd.DataFrame(
        {
            "country": ["A", "A"],
            "year": [2000, 2000],
            "indicator_code": ["x", "y"],
            "value": [1.0, 2.0],
        }
    )

    result = wdi_long_to_panel(
        long_df,
        country_col="country",
        indicator_code_col="indicator_code",
        value_name="value",
    )

    assert result.loc[0, "x"] == 1.0
    assert result.loc[0, "y"] == 2.0


def test_prefix_indicator_columns() -> None:
    panel = pd.DataFrame({"country": ["A"], "year": [2000], "x": [1.0]})

    result = prefix_indicator_columns(
        panel,
        country_col="country",
        year_col="year",
        prefix="wdi_",
    )

    assert "wdi_x" in result.columns
    assert "country" in result.columns


def test_drop_all_nan_indicator_rows() -> None:
    panel = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2000, 2000],
            "x": [None, 1.0],
            "y": [None, None],
        }
    )

    result, summary = drop_all_nan_indicator_rows(
        panel,
        country_col="country",
        year_col="year",
    )

    assert len(result) == 1
    assert summary["rows_dropped_all_nan"] == 1


def test_pivot_wdi_success() -> None:
    wdi = pd.DataFrame(
        {
            "country": ["Albania", "Albania"],
            "country_code": ["ALB", "ALB"],
            "indicator_code": ["x", "y"],
            "2000": [1.0, 2.0],
            "2001": [3.0, None],
        }
    )

    config = {
        "wdi": {
            "indicator_codes": ["x"],
            "indicator_code_col": "indicator_code",
            "country_col": "country",
            "country_code_col": "country_code",
            "value_name": "value",
            "year_min": 2000,
            "year_max": 2001,
            "wide_prefix": "wdi_",
            "drop_all_nan_rows": True,
        }
    }

    result, summary = pivot_wdi(wdi, config)

    assert "wdi_x" in result.columns
    assert result.shape[0] == 2
    assert summary["rows_after_indicator_filter"] == 1
    assert summary["output_rows"] == 2


def test_pivot_wdi_missing_required_column_raises() -> None:
    wdi = pd.DataFrame({"country": ["A"], "2000": [1.0]})
    config = {"wdi": {}}

    with pytest.raises(KeyError):
        pivot_wdi(wdi, config)


def test_pivot_wdi_no_year_columns_raises() -> None:
    wdi = pd.DataFrame(
        {
            "country": ["A"],
            "country_code": ["AAA"],
            "indicator_code": ["x"],
        }
    )
    config = {"wdi": {}}

    with pytest.raises(ValueError):
        pivot_wdi(wdi, config)
