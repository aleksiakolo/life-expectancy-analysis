from __future__ import annotations

from typing import Any

import pandas as pd

from life_expectancy.data.cleaning import clean_wb, clean_who
from life_expectancy.data.imputation import impute_panel
from life_expectancy.data.loading import load_source
from life_expectancy.data.panel import add_life_expectancy_target, merge_panel_sources
from life_expectancy.data.standardization import standardize
from life_expectancy.data.wdi import pivot_wdi

Summary = dict[str, Any]


def load_and_standardize_sources(
    config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], Summary]:
    """Load and standardize all configured raw data sources.

    Args:
        config: Full project configuration dictionary.

    Returns:
        Tuple containing standardized source DataFrames and a summary dictionary.
    """

    def load_sources(config) -> dict[str, pd.DataFrame]:
        return {
            source_name: load_source(config, source_name)
            for source_name in config["data"]["raw_sources"]
        }

    raw_sources = load_sources(config)
    source_configs = config["data"]["raw_sources"]

    standardized_sources = {
        source_name: standardize(raw_df, source_configs[source_name])
        for source_name, raw_df in raw_sources.items()
    }

    summary: Summary = {
        source_name: {
            "raw_rows": len(raw_sources[source_name]),
            "raw_cols": raw_sources[source_name].shape[1],
            "standardized_rows": len(standardized_sources[source_name]),
            "standardized_cols": standardized_sources[source_name].shape[1],
        }
        for source_name in standardized_sources
    }

    return standardized_sources, summary


def clean_sources(
    sources: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], Summary]:
    """Clean standardized WHO and World Bank sources.

    Args:
        sources: Dictionary of standardized source DataFrames.
        config: Full project configuration dictionary.

    Returns:
        Tuple containing cleaned source DataFrames and cleaning summaries.
    """
    who_clean, who_summary = clean_who(sources["who"], config)
    wb_clean, wb_summary = clean_wb(sources["wb"], config)

    cleaned_sources = {
        "who": who_clean,
        "wb": wb_clean,
    }

    summary: Summary = {
        "who": who_summary,
        "wb": wb_summary,
    }

    return cleaned_sources, summary


def build_panel(
    cleaned_sources: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, Summary]:
    """Build the merged analytical panel from cleaned WHO and World Bank data.

    Args:
        cleaned_sources: Dictionary containing cleaned `who` and `wb` DataFrames.
        config: Full project configuration dictionary.

    Returns:
        Tuple containing merged panel and panel-building summary.
    """
    panel, merge_summary = merge_panel_sources(
        cleaned_sources["who"],
        cleaned_sources["wb"],
        config=config,
    )

    panel = add_life_expectancy_target(panel, config)

    summary: Summary = {
        "merge": merge_summary,
        "target_col": config["data"]["panel"].get("target_col", "life_expectancy"),
        "rows_after_target": len(panel),
        "cols_after_target": panel.shape[1],
    }

    return panel, summary


def build_processed_dataset(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Summary]:
    """Run the full preprocessing pipeline.

    Pipeline steps:
        1. Load raw configured sources.
        2. Standardize source schemas.
        3. Clean WHO and World Bank data.
        4. Optionally pivot WDI for diagnostics/reference.
        5. Merge cleaned WHO and World Bank data.
        6. Create final life expectancy target.
        7. Impute configured missing values.

    Args:
        config: Full project configuration dictionary.

    Returns:
        Tuple containing final processed dataset and full pipeline summary.
    """
    standardized_sources, standardization_summary = load_and_standardize_sources(config)
    cleaned_sources, cleaning_summary = clean_sources(standardized_sources, config)
    panel, panel_summary = build_panel(cleaned_sources, config)
    panel_imputed, imputation_summary = impute_panel(panel, config)

    summary: Summary = {
        "standardization": standardization_summary,
        "cleaning": cleaning_summary,
        "panel": panel_summary,
        "imputation": imputation_summary,
        "final_rows": len(panel_imputed),
        "final_cols": panel_imputed.shape[1],
    }

    if "wdi" in standardized_sources and "wdi" in config:
        wdi_panel, wdi_summary = pivot_wdi(standardized_sources["wdi"], config)
        summary["wdi"] = wdi_summary
        summary["wdi_rows"] = len(wdi_panel)
        summary["wdi_cols"] = wdi_panel.shape[1]

    return panel_imputed, summary
