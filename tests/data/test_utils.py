from pathlib import Path

import pandas as pd
import pytest

from life_expectancy.data.utils import (
    clean_column_name,
    clean_string_series,
    coerce_year_to_int,
    require_columns,
    resolve_project_path,
)


def test_clean_column_name() -> None:
    assert (
        clean_column_name("Life Expectancy World Bank") == "life_expectancy_world_bank"
    )
    assert clean_column_name("Health Expenditure %") == "health_expenditure_percent"
    assert clean_column_name("A/B-C") == "a_b_c"


def test_clean_string_series() -> None:
    series = pd.Series([" Albania ", "Japan", None])
    result = clean_string_series(series)

    assert result.iloc[0] == "Albania"
    assert result.iloc[1] == "Japan"
    assert pd.isna(result.iloc[2])


def test_coerce_year_to_int() -> None:
    series = pd.Series(["2001", "2010", "bad"])
    result = coerce_year_to_int(series)

    assert str(result.dtype) == "Int64"
    assert result.iloc[0] == 2001
    assert pd.isna(result.iloc[2])


def test_require_columns_passes() -> None:
    df = pd.DataFrame({"country": ["Albania"], "year": [2010]})
    require_columns(df, ["country", "year"], name="df")


def test_require_columns_raises() -> None:
    df = pd.DataFrame({"country": ["Albania"]})

    with pytest.raises(KeyError):
        require_columns(df, ["country", "year"], name="df")


def test_resolve_project_path_relative(tmp_path: Path) -> None:
    config = {"project": {"root": str(tmp_path)}}

    resolved = resolve_project_path(config, "data/raw/file.csv")

    assert resolved == tmp_path / "data/raw/file.csv"
