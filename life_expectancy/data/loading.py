from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from life_expectancy.data.utils import resolve_project_path


def load_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file from disk.

    Args:
        path: Path to the CSV file.
        **read_csv_kwargs: Additional keyword arguments passed to `pandas.read_csv`.

    Returns:
        Loaded CSV as a DataFrame.

    Raises:
        FileNotFoundError: If the file path does not exist.
    """
    csv_path = Path(path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return pd.read_csv(csv_path, **read_csv_kwargs)


def load_wdi_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a World Development Indicators CSV export.

    Some WDI exports contain metadata rows before the real header. This function
    first tries a direct CSV read. If expected WDI columns are missing, it falls
    back to reading with `skiprows=4`.

    Args:
        path: Path to the WDI CSV file.
        **read_csv_kwargs: Additional keyword arguments passed to `pandas.read_csv`.

    Returns:
        Loaded WDI data as a DataFrame.

    Raises:
        FileNotFoundError: If the file path does not exist.
    """
    csv_path = Path(path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"WDI file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, **read_csv_kwargs)
        if has_wdi_header(df):
            return df
    except (UnicodeDecodeError, pd.errors.ParserError):
        pass

    return pd.read_csv(csv_path, skiprows=4, **read_csv_kwargs)


def load_source(config: dict[str, Any], source_name: str) -> pd.DataFrame:
    """Load one raw data source using project configuration.

    Args:
        config: Full project configuration dictionary.
        source_name: Name of the source to load, such as `"who"`, `"wb"`, or `"wdi"`.

    Returns:
        Loaded data source as a DataFrame.

    Raises:
        KeyError: If the source is not defined in the configuration.
        ValueError: If the configured loader is unsupported.
        FileNotFoundError: If the resolved file path does not exist.
    """
    source_config = config["data"]["raw_sources"][source_name]

    path = resolve_project_path(config, source_config["path"])
    loader = source_config.get("loader", "csv")
    read_csv_kwargs = source_config.get("read_csv_kwargs", {})

    if loader == "csv":
        return load_csv(path, **read_csv_kwargs)

    if loader == "wdi":
        return load_wdi_csv(path, **read_csv_kwargs)

    raise ValueError(f"Unsupported loader: {loader}")


def has_wdi_header(df: pd.DataFrame) -> bool:
    """Check whether a DataFrame has the expected WDI columns.

    Args:
        df: Loaded candidate WDI DataFrame.

    Returns:
        True if the DataFrame contains the expected WDI header columns.
    """
    required_columns = {
        "Country Name",
        "Country Code",
        "Indicator Name",
        "Indicator Code",
    }

    return required_columns.issubset(df.columns)
