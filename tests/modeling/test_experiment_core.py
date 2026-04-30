from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from life_expectancy.modeling.experiments.core import (
    append_run_log,
    collect_required_columns,
    run_time_experiment,
    run_time_experiment_from_config,
    save_time_split_metadata,
)
from life_expectancy.modeling.splits import SplitInfo


def make_modeling_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["A", "A", "A", "B", "B", "B"],
            "year": [2010, 2011, 2012, 2010, 2011, 2012],
            "feature": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "life_expectancy": [70.0, 71.0, 72.0, 80.0, 81.0, 82.0],
        }
    )


def test_save_time_split_metadata(tmp_path: Path) -> None:
    split_info = SplitInfo(
        split_type="time",
        n_train=4,
        n_test=2,
        extra={"cutoff": 2012},
    )

    out_path = save_time_split_metadata(
        split_info,
        tmp_path / "split.json",
    )

    assert out_path.exists()
    assert "split.json" in str(out_path)


def test_append_run_log_creates_and_appends(tmp_path: Path) -> None:
    out_path = tmp_path / "runs.csv"

    append_run_log({"model_name": "m1", "rmse": 1.0}, out_path)
    append_run_log({"model_name": "m2", "rmse": 2.0}, out_path)

    result = pd.read_csv(out_path)

    assert result["model_name"].tolist() == ["m1", "m2"]
    assert result["rmse"].tolist() == [1.0, 2.0]


def test_collect_required_columns() -> None:
    df = make_modeling_df()

    cols = collect_required_columns(
        df=df,
        feature_list=["feature"],
        target_col="life_expectancy",
        year_col="year",
        id_cols=["country", "missing_optional"],
    )

    assert cols == ["feature", "life_expectancy", "year", "country"]


def test_run_time_experiment(tmp_path: Path) -> None:
    df = make_modeling_df()
    log_path = tmp_path / "run_log.csv"

    row, pred_df, split_info, pipeline = run_time_experiment(
        df=df,
        feature_list=["feature"],
        target_col="life_expectancy",
        year_col="year",
        model_name="linear",
        model=LinearRegression(),
        scale_numeric="none",
        test_years=1,
        run_log_path=log_path,
        split_label="time_test",
        id_cols=["country", "year"],
    )

    assert row["model_name"] == "linear"
    assert row["split_name"] == "time_test"
    assert row["n_train"] == 4
    assert row["n_test"] == 2
    assert "rmse" in row
    assert "mae" in row
    assert "r2" in row

    assert pred_df.shape[0] == 2
    assert "country" in pred_df.columns
    assert "year" in pred_df.columns
    assert "y_true" in pred_df.columns
    assert "y_pred" in pred_df.columns
    assert "abs_error" in pred_df.columns

    assert split_info.split_type == "time"
    assert isinstance(pipeline, Pipeline)
    assert log_path.exists()


def test_run_time_experiment_from_config(tmp_path: Path) -> None:
    df = make_modeling_df()

    config = {
        "modeling": {
            "split": {
                "target_col": "life_expectancy",
                "year_col": "year",
                "test_years": 1,
            },
            "pipeline": {
                "scale_numeric": "none",
                "add_numeric_missing_indicators": False,
            },
            "experiment": {
                "run_log_path": str(tmp_path / "configured_runs.csv"),
                "split_label": "configured_time",
                "id_cols": ["country", "year"],
            },
        }
    }

    row, pred_df, split_info, pipeline = run_time_experiment_from_config(
        df=df,
        feature_list=["feature"],
        model_name="linear_config",
        model=LinearRegression(),
        config=config,
    )

    assert row["model_name"] == "linear_config"
    assert row["split_name"] == "configured_time"
    assert pred_df.shape[0] == 2
    assert split_info.extra["test_years"] == 1
    assert isinstance(pipeline, Pipeline)
