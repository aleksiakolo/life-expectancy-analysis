import pandas as pd
import pytest

from life_expectancy.modeling.splits import (
    filter_countries_min_years,
    make_random_split,
    make_random_split_from_config,
    make_time_split,
    make_time_split_from_config,
    split_xy,
)


def make_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["A", "A", "A", "B", "B", "B"],
            "year": [2010, 2011, 2012, 2010, 2011, 2012],
            "feature": [1, 2, 3, 4, 5, 6],
            "life_expectancy": [70, 71, 72, 80, 81, 82],
        }
    )


def test_filter_countries_min_years() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "A", "B", "B"],
            "year": [2010, 2011, 2012, 2010, 2011],
        }
    )

    result = filter_countries_min_years(df, min_years=3)

    assert result["country"].unique().tolist() == ["A"]


def test_filter_countries_min_years_missing_column_raises() -> None:
    df = pd.DataFrame({"country": ["A"]})

    with pytest.raises(KeyError):
        filter_countries_min_years(df)


def test_split_xy() -> None:
    df = make_panel()

    x, y = split_xy(df, target_col="life_expectancy")

    assert "life_expectancy" not in x.columns
    assert y.tolist() == [70, 71, 72, 80, 81, 82]


def test_split_xy_missing_target_raises() -> None:
    df = pd.DataFrame({"feature": [1, 2]})

    with pytest.raises(KeyError):
        split_xy(df, target_col="target")


def test_make_random_split() -> None:
    df = make_panel()

    x_train, x_test, y_train, y_test, info = make_random_split(
        df,
        target_col="life_expectancy",
        test_size=0.33,
        seed=42,
    )

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert info.split_type == "random"
    assert info.n_train == len(x_train)
    assert info.n_test == len(x_test)


def test_make_random_split_drops_missing_target() -> None:
    df = make_panel()
    df.loc[0, "life_expectancy"] = None

    x_train, x_test, y_train, y_test, info = make_random_split(
        df,
        target_col="life_expectancy",
        test_size=0.2,
        seed=42,
    )

    assert len(x_train) + len(x_test) == 5
    assert y_train.notna().all()
    assert y_test.notna().all()
    assert info.extra["dropna_target"] is True


def test_make_random_split_from_config() -> None:
    df = make_panel()
    config = {
        "modeling": {
            "split": {
                "target_col": "life_expectancy",
                "test_size": 0.33,
                "seed": 7,
                "dropna_target": True,
            }
        }
    }

    _, _, _, _, info = make_random_split_from_config(df, config)

    assert info.extra["test_size"] == 0.33
    assert info.extra["seed"] == 7


def test_make_time_split() -> None:
    df = make_panel()

    x_train, x_test, y_train, y_test, info = make_time_split(
        df,
        target_col="life_expectancy",
        year_col="year",
        test_years=1,
    )

    assert x_train["year"].max() == 2011
    assert x_test["year"].min() == 2012
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert info.split_type == "time"
    assert info.extra["cutoff"] == 2012


def test_make_time_split_multiple_test_years() -> None:
    df = make_panel()

    x_train, x_test, _, _, info = make_time_split(
        df,
        target_col="life_expectancy",
        year_col="year",
        test_years=2,
    )

    assert x_train["year"].unique().tolist() == [2010]
    assert sorted(x_test["year"].unique().tolist()) == [2011, 2012]
    assert info.extra["test_year_values"] == [2011, 2012]


def test_make_time_split_from_config() -> None:
    df = make_panel()
    config = {
        "modeling": {
            "split": {
                "target_col": "life_expectancy",
                "year_col": "year",
                "test_years": 2,
                "dropna_target": True,
            }
        }
    }

    _, x_test, _, _, info = make_time_split_from_config(df, config)

    assert sorted(x_test["year"].unique().tolist()) == [2011, 2012]
    assert info.extra["test_years"] == 2


def test_make_time_split_non_numeric_year_raises() -> None:
    df = make_panel()
    df["year"] = df["year"].astype("object")
    df.loc[0, "year"] = "bad"

    with pytest.raises(ValueError):
        make_time_split(
            df,
            target_col="life_expectancy",
            year_col="year",
        )


def test_make_time_split_empty_split_raises() -> None:
    df = make_panel()

    with pytest.raises(ValueError):
        make_time_split(
            df,
            target_col="life_expectancy",
            year_col="year",
            test_years=10,
        )


def test_make_time_split_invalid_test_years_raises() -> None:
    df = make_panel()

    with pytest.raises(ValueError):
        make_time_split(
            df,
            target_col="life_expectancy",
            year_col="year",
            test_years=0,
        )
