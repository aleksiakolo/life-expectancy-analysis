from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from life_expectancy.data.utils import coerce_year_to_int, require_columns

JoinKey = tuple[str, str]
MergeHow = Literal["inner", "left", "right", "outer"]
MergeValidate = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
Summary = dict[str, Any]


def merge_panel_sources(
    who_df: pd.DataFrame,
    wb_df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Summary]:
    """Merge cleaned WHO and World Bank data into one country-year panel.

    Args:
        who_df: Cleaned WHO country-year DataFrame.
        wb_df: Cleaned World Bank country-year DataFrame.
        config: Full project configuration dictionary containing a `panel` section.

    Returns:
        Tuple containing the merged panel and a summary dictionary.

    Raises:
        KeyError: If required merge key columns are missing.
    """
    panel_config = config["data"]["panel"]

    on: JoinKey = tuple(panel_config.get("on", ["country", "year"]))
    how: MergeHow = panel_config.get("how", "inner")
    validate: MergeValidate = panel_config.get("validate", "one_to_one")
    suffixes: tuple[str, str] = tuple(panel_config.get("suffixes", ["_who", "_wb"]))

    country_key, year_key = on

    require_columns(who_df, [country_key, year_key], name="who_df")
    require_columns(wb_df, [country_key, year_key], name="wb_df")

    who = prepare_merge_keys(who_df, country_key=country_key, year_key=year_key)
    wb = prepare_merge_keys(wb_df, country_key=country_key, year_key=year_key)

    summary: Summary = {
        "who_rows": len(who),
        "wb_rows": len(wb),
        "how": how,
        "on": on,
        "validate": validate,
    }

    panel = who.merge(
        wb,
        how=how,
        on=list(on),
        validate=validate,
        suffixes=suffixes,
    )

    summary["merged_rows"] = len(panel)

    add_key_stats(
        summary,
        left=who,
        right=wb,
        merged=panel,
        country_key=country_key,
        year_key=year_key,
        left_name="who",
        right_name="wb",
    )

    summary.update(panel_output_summary(panel, year_key))

    return panel, summary


def add_life_expectancy_target(
    panel: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Create the final life expectancy target column.

    Args:
        panel: Merged country-year panel.
        config: Full project configuration dictionary containing a `target` section.

    Returns:
        Copy of the panel with the final target column added.

    Raises:
        KeyError: If none of the configured source target columns exist.
        ValueError: If the configured target strategy is unsupported.
    """
    target_config = config["data"]["panel"]["target"]

    source_cols: list[str] = target_config.get(
        "source_cols",
        ["life_expectancy_who", "life_expectancy_wb"],
    )
    target_col: str = target_config.get("target_col", "life_expectancy")
    strategy: str = target_config.get("strategy", "mean")

    available_cols = [col for col in source_cols if col in panel.columns]

    if not available_cols:
        raise KeyError(
            f"None of the configured target source columns exist: {source_cols}. "
            f"Found columns: {list(panel.columns)}"
        )

    out = panel.copy()

    if strategy == "mean":
        out[target_col] = out[available_cols].mean(axis=1)
        return out

    if strategy == "first_non_null":
        out[target_col] = out[available_cols].bfill(axis=1).iloc[:, 0]
        return out

    raise ValueError(f"Unsupported target strategy: {strategy}")


def prepare_merge_keys(
    df: pd.DataFrame,
    *,
    country_key: str,
    year_key: str,
) -> pd.DataFrame:
    """Clean merge key columns before joining.

    Args:
        df: Input DataFrame.
        country_key: Country key column.
        year_key: Year key column.

    Returns:
        Copy of the DataFrame with cleaned merge keys.
    """
    out = df.copy()
    out[country_key] = out[country_key].astype("string").str.strip()
    out[year_key] = coerce_year_to_int(out[year_key])

    return out


def add_key_stats(
    summary: Summary,
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    merged: pd.DataFrame,
    country_key: str,
    year_key: str,
    left_name: str,
    right_name: str,
) -> None:
    """Add merge key coverage statistics to a summary dictionary.

    Args:
        summary: Summary dictionary updated in place.
        left: Left input DataFrame.
        right: Right input DataFrame.
        merged: Merged output DataFrame.
        country_key: Country key column.
        year_key: Year key column.
        left_name: Label for left DataFrame in summary keys.
        right_name: Label for right DataFrame in summary keys.
    """
    left_keys = set(zip(left[country_key], left[year_key], strict=False))
    right_keys = set(zip(right[country_key], right[year_key], strict=False))
    merged_keys = set(zip(merged[country_key], merged[year_key], strict=False))

    summary[f"{left_name}_unique_keys"] = len(left_keys)
    summary[f"{right_name}_unique_keys"] = len(right_keys)
    summary["merged_unique_keys"] = len(merged_keys)
    summary[f"keys_lost_from_{left_name}"] = len(left_keys - merged_keys)
    summary[f"keys_lost_from_{right_name}"] = len(right_keys - merged_keys)


def panel_output_summary(df: pd.DataFrame, year_col: str) -> Summary:
    """Create standard output summary fields for panel data.

    Args:
        df: Output panel DataFrame.
        year_col: Year column name.

    Returns:
        Summary dictionary with output shape and year range.
    """
    year_values = pd.to_numeric(df[year_col], errors="coerce")

    return {
        "output_rows": len(df),
        "output_cols": df.shape[1],
        "year_min": int(year_values.min()) if year_values.notna().any() else None,
        "year_max": int(year_values.max()) if year_values.notna().any() else None,
    }
