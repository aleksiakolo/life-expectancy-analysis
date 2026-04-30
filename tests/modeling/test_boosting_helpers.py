import numpy as np
import pandas as pd
import pytest

from life_expectancy.modeling.experiments.boosting import (
    build_boosting_prediction_df,
    collect_experiment_columns,
    make_time_train_val_test,
)
from life_expectancy.modeling.model.boosting import get_xgb_trial_grid


def make_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["A"] * 5 + ["B"] * 5,
            "year": [2010, 2011, 2012, 2013, 2014] * 2,
            "gdp": [1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
            "schooling": [10, 11, 12, 13, 14, 9, 10, 11, 12, 13],
            "life_expectancy": [70, 71, 72, 73, 74, 80, 81, 82, 83, 84],
        }
    )


def test_make_time_train_val_test() -> None:
    df = make_panel()

    train_df, val_df, test_df = make_time_train_val_test(
        df,
        target_col="life_expectancy",
        year_col="year",
        test_years=1,
        val_years=1,
    )

    assert sorted(train_df["year"].unique().tolist()) == [2010, 2011, 2012]
    assert sorted(val_df["year"].unique().tolist()) == [2013]
    assert sorted(test_df["year"].unique().tolist()) == [2014]


def test_make_time_train_val_test_drops_missing_target() -> None:
    df = make_panel()
    df.loc[0, "life_expectancy"] = np.nan

    train_df, _, _ = make_time_train_val_test(
        df,
        target_col="life_expectancy",
        year_col="year",
        test_years=1,
        val_years=1,
    )

    assert train_df["life_expectancy"].notna().all()


def test_make_time_train_val_test_missing_column_raises() -> None:
    df = make_panel().drop(columns=["life_expectancy"])

    with pytest.raises(KeyError):
        make_time_train_val_test(
            df,
            target_col="life_expectancy",
            year_col="year",
        )


def test_make_time_train_val_test_not_enough_years_raises() -> None:
    df = make_panel()

    with pytest.raises(ValueError):
        make_time_train_val_test(
            df,
            target_col="life_expectancy",
            year_col="year",
            test_years=3,
            val_years=2,
        )


def test_collect_experiment_columns() -> None:
    df = make_panel()

    cols = collect_experiment_columns(
        df=df,
        feature_list=["gdp", "schooling"],
        target_col="life_expectancy",
        year_col="year",
        id_cols=["country", "region"],
    )

    assert cols == [
        "gdp",
        "schooling",
        "life_expectancy",
        "year",
        "country",
    ]


def test_collect_experiment_columns_missing_required_raises() -> None:
    df = make_panel()

    with pytest.raises(KeyError):
        collect_experiment_columns(
            df=df,
            feature_list=["missing_feature"],
            target_col="life_expectancy",
            year_col="year",
            id_cols=["country"],
        )


def test_build_boosting_prediction_df() -> None:
    test_df = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2014, 2014],
        },
        index=[10, 11],
    )

    pred_df = build_boosting_prediction_df(
        y_true=pd.Series([70.0, 80.0], index=[10, 11]),
        y_pred=np.array([72.0, 79.0]),
        test_df=test_df,
        id_cols=["country", "year"],
    )

    assert pred_df["y_true"].tolist() == [70.0, 80.0]
    assert pred_df["y_pred"].tolist() == [72.0, 79.0]
    assert pred_df["error"].tolist() == [2.0, -1.0]
    assert pred_df["abs_error"].tolist() == [2.0, 1.0]
    assert pred_df["country"].tolist() == ["A", "B"]
    assert pred_df["year"].tolist() == [2014, 2014]


def test_get_xgb_trial_grid() -> None:
    trials = get_xgb_trial_grid(random_state=7)

    assert len(trials) == 9
    assert all("trial_name" in trial for trial in trials)
    assert all("params" in trial for trial in trials)
    assert all(trial["params"]["random_state"] == 7 for trial in trials)
