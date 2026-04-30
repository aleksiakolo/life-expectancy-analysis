import pandas as pd

from life_expectancy.data.panel import add_life_expectancy_target, merge_panel_sources


def test_merge_panel_sources_inner() -> None:
    who = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2010, 2010],
            "life_expectancy_who": [70.0, 80.0],
        }
    )

    wb = pd.DataFrame(
        {
            "country": ["A", "C"],
            "year": [2010, 2010],
            "life_expectancy_wb": [72.0, 65.0],
        }
    )

    config = {
        "data": {
            "panel": {
                "on": ["country", "year"],
                "how": "inner",
                "validate": "one_to_one",
                "suffixes": ["_who", "_wb"],
            }
        }
    }

    panel, summary = merge_panel_sources(who, wb, config=config)

    assert len(panel) == 1
    assert panel.loc[0, "country"] == "A"
    assert summary["merged_rows"] == 1


def test_add_life_expectancy_target_mean() -> None:
    panel = pd.DataFrame(
        {
            "life_expectancy_who": [70.0, None],
            "life_expectancy_wb": [72.0, 80.0],
        }
    )

    config = {
        "data": {
            "panel": {
                "target": {
                    "source_cols": ["life_expectancy_who", "life_expectancy_wb"],
                    "target_col": "life_expectancy",
                    "strategy": "mean",
                }
            }
        }
    }

    result = add_life_expectancy_target(panel, config)

    assert result.loc[0, "life_expectancy"] == 71.0
    assert result.loc[1, "life_expectancy"] == 80.0


def test_add_life_expectancy_target_first_non_null() -> None:
    panel = pd.DataFrame(
        {
            "life_expectancy_who": [70.0, None],
            "life_expectancy_wb": [72.0, 80.0],
        }
    )

    config = {
        "data": {
            "panel": {
                "target": {
                    "source_cols": ["life_expectancy_who", "life_expectancy_wb"],
                    "target_col": "life_expectancy",
                    "strategy": "first_non_null",
                }
            }
        }
    }

    result = add_life_expectancy_target(panel, config)

    assert result.loc[0, "life_expectancy"] == 70.0
    assert result.loc[1, "life_expectancy"] == 80.0
