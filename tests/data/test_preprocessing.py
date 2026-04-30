from pathlib import Path

import pandas as pd

from life_expectancy.data.preprocessing import (
    build_panel,
    build_processed_dataset,
    clean_sources,
    load_and_standardize_sources,
)


def make_test_config(tmp_path: Path) -> dict:
    who_path = tmp_path / "who.csv"
    wb_path = tmp_path / "wb.csv"
    wdi_path = tmp_path / "wdi.csv"

    pd.DataFrame(
        {
            "Country": ["A", "B"],
            "Year": [2010, 2010],
            "Life expectancy": [70.0, 80.0],
            "Polio": [95, 90],
        }
    ).to_csv(who_path, index=False)

    pd.DataFrame(
        {
            "Country Name": ["A", "B"],
            "Year": [2010, 2010],
            "Life Expectancy World Bank": [72.0, 82.0],
            "Region": ["R1", "R2"],
            "IncomeGroup": ["Low", "High"],
            "GDP": [1000.0, 2000.0],
            "Unemployment": [5.0, 6.0],
        }
    ).to_csv(wb_path, index=False)

    pd.DataFrame(
        {
            "Country Name": ["A"],
            "Country Code": ["AAA"],
            "Indicator Name": ["Example"],
            "Indicator Code": ["x"],
            "2010": [1.0],
        }
    ).to_csv(wdi_path, index=False)

    return {
        "project": {"root": str(tmp_path)},
        "data": {
            "raw_sources": {
                "who": {
                    "name": "who",
                    "path": "who.csv",
                    "loader": "csv",
                    "country_col": "Country",
                    "year_col": "Year",
                    "column_renames": {
                        "life_expectancy": "life_expectancy_who",
                    },
                },
                "wb": {
                    "name": "wb",
                    "path": "wb.csv",
                    "loader": "csv",
                    "country_col": "Country Name",
                    "year_col": "Year",
                    "column_renames": {
                        "life_expectancy_world_bank": "life_expectancy_wb",
                        "incomegroup": "income_group",
                    },
                },
                "wdi": {
                    "name": "wdi",
                    "path": "wdi.csv",
                    "loader": "wdi",
                    "column_renames": {},
                },
            },
            "panel": {
                "on": ["country", "year"],
                "how": "inner",
                "validate": "one_to_one",
                "suffixes": ["_who", "_wb"],
                "target": {
                    "source_cols": ["life_expectancy_who", "life_expectancy_wb"],
                    "target_col": "life_expectancy",
                    "strategy": "mean",
                },
            },
        },
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
            },
            "wb": {
                "key": ["country", "year"],
                "indicator_subset": None,
                "winsorize_cols": [],
                "winsorize_limits": [0.01, 0.99],
                "log_transform_cols": [],
                "log_epsilon": 1e-9,
                "feature_missingness_drop_threshold": None,
            },
        },
        "imputation": {
            "key": ["country", "year"],
            "target_col": "life_expectancy",
            "numeric_only": True,
            "interpolate_cols": [],
            "group_cols": [],
            "group_median_cols": [],
            "median_impute_cols": ["gdp", "unemployment"],
        },
        "wdi": {
            "indicator_codes": ["x"],
            "indicator_code_col": "indicator_code",
            "country_col": "country",
            "country_code_col": "country_code",
            "value_name": "value",
            "year_min": 2010,
            "year_max": 2010,
            "wide_prefix": "wdi_",
            "drop_all_nan_rows": True,
        },
    }


def test_load_and_standardize_sources(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)

    sources, summary = load_and_standardize_sources(config)

    assert set(sources) == {"who", "wb", "wdi"}
    assert "country" in sources["who"].columns
    assert "life_expectancy_who" in sources["who"].columns
    assert "life_expectancy_wb" in sources["wb"].columns
    assert summary["who"]["raw_rows"] == 2
    assert summary["wb"]["standardized_rows"] == 2


def test_clean_sources(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    sources, _ = load_and_standardize_sources(config)

    cleaned, summary = clean_sources(sources, config)

    assert set(cleaned) == {"who", "wb"}
    assert len(cleaned["who"]) == 2
    assert len(cleaned["wb"]) == 2
    assert "who" in summary
    assert "wb" in summary


def test_build_panel(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    sources, _ = load_and_standardize_sources(config)
    cleaned, _ = clean_sources(sources, config)

    panel, summary = build_panel(cleaned, config)

    assert len(panel) == 2
    assert "life_expectancy" in panel.columns
    assert panel.loc[0, "life_expectancy"] == 71.0
    assert summary["target_col"] == "life_expectancy"


def test_build_processed_dataset(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)

    panel, summary = build_processed_dataset(config)

    assert len(panel) == 2
    assert "life_expectancy" in panel.columns
    assert summary["final_rows"] == 2
    assert "standardization" in summary
    assert "cleaning" in summary
    assert "panel" in summary
    assert "imputation" in summary
    assert "wdi" in summary
