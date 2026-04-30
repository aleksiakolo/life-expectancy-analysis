from pathlib import Path

import pandas as pd
import pytest

from life_expectancy.features.feature_selection import (
    build_feature_sets_abc,
    choose_manual_feature_set,
    compute_correlation_matrix,
    compute_vif_table,
    correlation_prune,
    high_correlation_pairs,
    iterative_vif_prune,
    save_feature_sets_json,
)


def make_model_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2010, 2011, 2012, 2013, 2014],
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 4, 3, 2, 1],
            "status_flag": [0, 1, 0, 1, 0],
            "life_expectancy": [60, 62, 64, 66, 68],
        }
    )


def make_feature_config() -> dict:
    return {
        "features": {
            "target_col": "life_expectancy",
            "year_col": "year",
            "status_col": "status",
            "status_flag_col": "status_flag",
            "log_candidates": [],
            "missingness_flag_candidates": [],
            "interactions": [],
            "leakage_cols": [],
            "drop_id_cols": [],
            "corr_threshold": 0.95,
            "vif_threshold": 10.0,
            "protected_features": ["year", "status_flag"],
            "manual_candidates": ["year", "a", "status_flag"],
            "manual_min_size": 2,
            "manual_max_size": 3,
        }
    }


def test_compute_correlation_matrix() -> None:
    df = make_model_df()

    result = compute_correlation_matrix(df, ["a", "b", "c"])

    assert result.loc["a", "b"] == 1.0
    assert result.loc["a", "c"] == -1.0


def test_high_correlation_pairs() -> None:
    df = make_model_df()

    result = high_correlation_pairs(df[["a", "b", "c"]], threshold=0.95)

    pairs = {
        tuple(sorted([row["feature_1"], row["feature_2"]]))
        for _, row in result.iterrows()
    }

    assert ("a", "b") in pairs
    assert ("a", "c") in pairs


def test_correlation_prune_drops_correlated_unprotected() -> None:
    df = make_model_df()

    kept, dropped, corr = correlation_prune(
        df,
        ["year", "a", "b", "status_flag"],
        threshold=0.95,
        protected=["year", "status_flag"],
    )

    assert "year" in kept
    assert "status_flag" in kept
    assert "b" in dropped
    assert "a" in corr.columns


def test_compute_vif_table() -> None:
    df = make_model_df()

    result = compute_vif_table(df, ["a", "status_flag"])

    assert set(result.columns) == {"feature", "vif"}
    assert set(result["feature"]) == {"a", "status_flag"}


def test_iterative_vif_prune_returns_lists_and_vif_table() -> None:
    df = make_model_df()

    with pytest.warns(RuntimeWarning):
        kept, dropped, vif_table = iterative_vif_prune(
            df,
            ["a", "b", "status_flag"],
            vif_threshold=10.0,
            protected=["status_flag"],
        )

    assert isinstance(kept, list)
    assert isinstance(dropped, list)
    assert "feature" in vif_table.columns
    assert "vif" in vif_table.columns


def test_choose_manual_feature_set() -> None:
    result = choose_manual_feature_set(
        ["year", "a", "b", "c"],
        manual_candidates=["year", "c", "missing"],
        min_size=3,
        max_size=3,
    )

    assert result == ["year", "c", "a"]


def test_build_feature_sets_abc() -> None:
    df = make_model_df()

    model_df, feature_sets, meta = build_feature_sets_abc(df, make_feature_config())

    assert "A" in feature_sets
    assert "B" in feature_sets
    assert "C" in feature_sets
    assert "life_expectancy" in model_df.columns
    assert meta["target_col"] == "life_expectancy"
    assert "corr_dropped" in meta
    assert "vif_dropped" in meta


def test_save_feature_sets_json(tmp_path: Path) -> None:
    out_path = tmp_path / "feature_sets.json"

    result_path = save_feature_sets_json(
        {"A": ["a", "b"], "B": ["a"], "C": ["a"]},
        {"note": "test"},
        out_path,
    )

    assert result_path == out_path
    assert out_path.exists()
