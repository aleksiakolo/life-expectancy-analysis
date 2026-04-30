import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from life_expectancy.modeling.experiments.sequence import (
    build_country_sequences,
    build_lstm_prediction_df,
    build_lstm_training_targets,
    get_last_target_values,
    make_flat_lag_dataframe,
    scale_sequence_splits,
    split_sequences_timeaware,
)


def make_sequence_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["A"] * 5 + ["B"] * 5,
            "country_code": ["AAA"] * 5 + ["BBB"] * 5,
            "region": ["R1"] * 5 + ["R2"] * 5,
            "year": [2010, 2011, 2012, 2013, 2014] * 2,
            "gdp": [1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
            "schooling": [10, 11, 12, 13, 14, 9, 10, 11, 12, 13],
            "life_expectancy": [70, 71, 72, 73, 74, 80, 81, 82, 83, 84],
        }
    )


def test_make_flat_lag_dataframe() -> None:
    df = make_sequence_panel()

    result = make_flat_lag_dataframe(
        df,
        feature_cols=["gdp"],
        target_col="life_expectancy",
        lags=(1, 2),
    )

    assert "life_expectancy_lag1" in result.columns
    assert "life_expectancy_lag2" in result.columns
    assert "gdp_lag1" in result.columns
    assert "gdp_lag2" in result.columns
    assert result["life_expectancy_lag1"].notna().all()


def test_build_country_sequences() -> None:
    df = make_sequence_panel()

    x, y, meta_df, seq_cols = build_country_sequences(
        df,
        feature_cols=["gdp", "schooling"],
        target_col="life_expectancy",
        country_col="country",
        year_col="year",
        window=3,
        include_target_history=True,
    )

    assert x.shape == (4, 3, 3)
    assert y.shape == (4,)
    assert seq_cols == ["life_expectancy", "gdp", "schooling"]
    assert meta_df["year"].tolist() == [2013, 2014, 2013, 2014]


def test_build_country_sequences_without_target_history() -> None:
    df = make_sequence_panel()

    x, y, meta_df, seq_cols = build_country_sequences(
        df,
        feature_cols=["gdp", "schooling"],
        target_col="life_expectancy",
        window=3,
        include_target_history=False,
    )

    assert x.shape == (4, 3, 2)
    assert y.shape == (4,)
    assert seq_cols == ["gdp", "schooling"]
    assert len(meta_df) == 4


def test_build_country_sequences_no_valid_sequences_raises() -> None:
    df = make_sequence_panel().head(2)

    with pytest.raises(ValueError):
        build_country_sequences(
            df,
            feature_cols=["gdp"],
            target_col="life_expectancy",
            window=3,
        )


def test_split_sequences_timeaware() -> None:
    df = make_sequence_panel()

    x, y, meta_df, _ = build_country_sequences(
        df,
        feature_cols=["gdp"],
        target_col="life_expectancy",
        window=2,
    )

    split = split_sequences_timeaware(
        x,
        y,
        meta_df,
        year_col="year",
        test_years=1,
        val_years=1,
    )

    assert split["X_train"].shape[0] == 2
    assert split["X_val"].shape[0] == 2
    assert split["X_test"].shape[0] == 2
    assert split["meta_val"]["year"].unique().tolist() == [2013]
    assert split["meta_test"]["year"].unique().tolist() == [2014]


def test_split_sequences_timeaware_not_enough_years_raises() -> None:
    df = make_sequence_panel()

    x, y, meta_df, _ = build_country_sequences(
        df,
        feature_cols=["gdp"],
        target_col="life_expectancy",
        window=3,
    )

    with pytest.raises(ValueError):
        split_sequences_timeaware(
            x,
            y,
            meta_df,
            year_col="year",
            test_years=2,
            val_years=1,
        )


def test_scale_sequence_splits() -> None:
    df = make_sequence_panel()

    x, y, meta_df, _ = build_country_sequences(
        df,
        feature_cols=["gdp"],
        target_col="life_expectancy",
        window=2,
    )

    split = split_sequences_timeaware(
        x,
        y,
        meta_df,
        test_years=1,
        val_years=1,
    )

    scaled_split, scaler = scale_sequence_splits(split)

    assert isinstance(scaler, StandardScaler)
    assert scaled_split["X_train"].shape == split["X_train"].shape
    assert scaled_split["X_val"].shape == split["X_val"].shape
    assert scaled_split["X_test"].shape == split["X_test"].shape


def test_get_last_target_values() -> None:
    x = np.array(
        [
            [[70.0, 1.0], [71.0, 2.0]],
            [[80.0, 3.0], [81.0, 4.0]],
        ],
        dtype=np.float32,
    )

    result = get_last_target_values(x, target_history_channel=0)

    assert result.tolist() == [71.0, 81.0]


def test_get_last_target_values_missing_channel_raises() -> None:
    x = np.ones((2, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        get_last_target_values(x, target_history_channel=None)


def test_build_lstm_training_targets_delta() -> None:
    split = {
        "X_train": np.array(
            [
                [[70.0, 1.0], [71.0, 2.0]],
                [[80.0, 3.0], [81.0, 4.0]],
            ],
            dtype=np.float32,
        ),
        "y_train": np.array([72.0, 83.0], dtype=np.float32),
        "X_val": np.array(
            [
                [[60.0, 1.0], [62.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        "y_val": np.array([65.0], dtype=np.float32),
    }

    y_train, y_val = build_lstm_training_targets(
        split,
        predict_delta=True,
        target_history_channel=0,
    )

    assert y_train.tolist() == [1.0, 2.0]
    assert y_val.tolist() == [3.0]


def test_build_lstm_training_targets_level() -> None:
    split = {
        "y_train": np.array([72.0, 83.0], dtype=np.float32),
        "y_val": np.array([65.0], dtype=np.float32),
    }

    y_train, y_val = build_lstm_training_targets(
        split,
        predict_delta=False,
        target_history_channel=None,
    )

    assert y_train.tolist() == [72.0, 83.0]
    assert y_val.tolist() == [65.0]


def test_build_lstm_prediction_df() -> None:
    meta_df = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2014, 2014],
        }
    )

    result = build_lstm_prediction_df(
        meta_df=meta_df,
        y_true=np.array([70.0, 80.0]),
        y_pred=np.array([72.0, 79.0]),
    )

    assert result["error"].tolist() == [2.0, -1.0]
    assert result["abs_error"].tolist() == [2.0, 1.0]
