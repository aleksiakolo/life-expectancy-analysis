from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def get_project_root(config: dict[str, Any]) -> Path:
    """Resolve the project root directory.

    The function first checks `config["project"]["root"]`. If that value is not
    set, it walks upward from the current working directory until it finds a project
    marker such as `configs/` or `pyproject.toml`.

    Args:
        config: Full project configuration dictionary.

    Returns:
        Absolute path to the project root.

    Raises:
        RuntimeError: If the project root cannot be found.
    """

    # 1. Config override (optional)
    root = config.get("project", {}).get("root")
    if root:
        return Path(root).expanduser().resolve()

    # 2. Auto-detect by looking for marker files
    current = Path().resolve()

    for parent in [current, *current.parents]:
        if (parent / "configs").exists() or (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Could not determine project root.")


def resolve_project_path(config: dict[str, Any], path: str | Path) -> Path:
    """Resolve a filesystem path against the project root.

    Args:
        config: Full project configuration dictionary.
        path: Absolute path or path relative to the project root.

    Returns:
        Absolute resolved path.
    """
    candidate_path = Path(path)

    if candidate_path.is_absolute():
        return candidate_path.expanduser().resolve()

    return (get_project_root(config) / candidate_path).resolve()


def clean_column_name(column: object) -> str:
    """Convert a raw column name to snake_case.

    Args:
        column: Raw column name.

    Returns:
        Cleaned snake_case column name.
    """
    cleaned = str(column).strip().lower()
    cleaned = cleaned.replace("%", "percent")
    cleaned = cleaned.replace("&", "and")
    cleaned = re.sub(r"[/\-()]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)

    return cleaned.strip("_")


def to_nullable_year(series: pd.Series) -> pd.Series:
    """Convert a year-like Series to pandas nullable integer dtype.

    Args:
        series: Series containing year-like values.

    Returns:
        Series converted to pandas nullable `Int64` dtype.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def clean_string_series(series: pd.Series) -> pd.Series:
    """Convert a Series to stripped pandas string dtype.

    Args:
        series: Series containing string-like values.

    Returns:
        Cleaned pandas string Series.
    """
    return series.astype("string").str.strip()


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that required columns exist in a DataFrame.

    Args:
        df: DataFrame to validate.
        required_columns: Column names that must be present.

    Raises:
        KeyError: If any required columns are missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            f"{missing_columns}. Found columns: {list(df.columns)}"
        )


def coerce_year_to_int(series: pd.Series) -> pd.Series:
    """Convert a year-like Series to pandas nullable integer dtype.

    Args:
        series: Series containing year-like values.

    Returns:
        Series converted to pandas nullable `Int64` dtype.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def require_columns(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    """Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        cols: Required column names.
        name: Human-readable DataFrame name used in errors.

    Raises:
        KeyError: If any required columns are missing.
    """
    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise KeyError(
            f"{name} is missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )


def prepare_panel_keys(
    df: pd.DataFrame, country_col: str, year_col: str
) -> pd.DataFrame:
    """Clean country and year key columns.

    Args:
        df: Input panel DataFrame.
        country_col: Country column name.
        year_col: Year column name.

    Returns:
        DataFrame with cleaned country and year columns.
    """
    out = df.copy()

    if country_col in out.columns:
        out[country_col] = out[country_col].astype("string").str.strip()

    if year_col in out.columns:
        out[year_col] = coerce_year_to_int(out[year_col])

    return out
