from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
import pandas as pd

DataSource = Literal["who", "wb", "wdi"]


def load_raw(
    source: DataSource,
    *,
    path: Optional[Union[str, Path]] = None,
    croissant_url: Optional[str] = None,
    record_set: Union[int, str, None] = 0,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load a raw dataset (WHO, WB panel, or WDI export).

    Prefer `path` for local CSVs. If `croissant_url` is provided, the data is
    loaded via Kaggle Croissant (mlcroissant).
    """
    if croissant_url:
        return _load_from_croissant(croissant_url, record_set)

    if path is None:
        raise ValueError("Please provide either `path` or `croissant_url`.")

    path = Path(path)

    if source == "wdi":
        return _load_wdi_export(path, **read_csv_kwargs)

    return pd.read_csv(path, **read_csv_kwargs)


def standardize(
    df: pd.DataFrame,
    source: DataSource,
    *,
    country_col: Optional[str] = None,
    year_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply simple standardization so all datasets can be processed consistently.

    Output always uses:
      - `country` (string)
      - `year` (Int64, pandas nullable int)
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if source == "who":
        return _standardize_who(out, country_col, year_col)
    if source == "wb":
        return _standardize_wb(out, country_col, year_col)
    if source == "wdi":
        return _standardize_wdi(out)

    raise ValueError(f"Unknown source: {source}")


def merge(
    who_df: pd.DataFrame,
    wb_df: pd.DataFrame,
    *,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    on: Tuple[str, str] = ("country", "year"),
    validate: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"] = "one_to_one",
    suffixes: Tuple[str, str] = ("_who", "_wb"),
    include_wdi: Optional[pd.DataFrame] = None,
    wdi_suffix: str = "_wdi",
    return_diagnostics: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Merge WHO and WB into a country-year panel. Optionally also merge a WDI panel.

    Returns the merged DataFrame, and (optionally) a small diagnostics dict.
    """
    country_key, year_key = on
    _require_cols(who_df, [country_key, year_key], "who_df")
    _require_cols(wb_df, [country_key, year_key], "wb_df")

    who = who_df.copy()
    wb = wb_df.copy()
    who[year_key] = _to_year_int(who[year_key])
    wb[year_key] = _to_year_int(wb[year_key])

    diag: Dict[str, Any] = {
        "who_rows": int(len(who)),
        "wb_rows": int(len(wb)),
        "how": how,
        "on": on,
    }

    merged = who.merge(
        wb,
        how=how,
        on=list(on),
        validate=validate,
        suffixes=suffixes,
    )
    diag["merged_rows"] = int(len(merged))

    _add_key_stats(diag, who, wb, merged, country_key, year_key)

    if include_wdi is not None:
        merged, wdi_diag = _merge_wdi(merged, include_wdi, how, on, year_key, wdi_suffix)
        diag.update(wdi_diag)

    if return_diagnostics:
        return merged, diag
    return merged


# ----------------------------
# Croissant + WDI loading
# ----------------------------

def _load_from_croissant(croissant_url: str, record_set: Union[int, str, None]) -> pd.DataFrame:
    """Load a dataset via Kaggle Croissant JSON-LD (requires mlcroissant)."""
    dataset, record_sets = _open_croissant(croissant_url)
    rs = _pick_record_set(record_sets, record_set)
    rows = list(dataset.records(record_set=rs.uuid))
    return pd.DataFrame(rows)


def _open_croissant(croissant_url: str):
    try:
        import mlcroissant as mlc  # type: ignore
    except Exception as e:
        raise ImportError(
            "To load from `croissant_url`, install mlcroissant (pip install mlcroissant). "
            "Or download the CSV and use `path=` instead."
        ) from e

    dataset = mlc.Dataset(croissant_url)
    record_sets = dataset.metadata.record_sets
    if not record_sets:
        raise ValueError("Croissant dataset has no record sets.")
    return dataset, record_sets


def _pick_record_set(record_sets: Sequence[Any], record_set: Union[int, str, None]) -> Any:
    if record_set is None:
        return record_sets[0]

    if isinstance(record_set, int):
        if record_set < 0 or record_set >= len(record_sets):
            raise IndexError(f"record_set index {record_set} out of range (n={len(record_sets)})")
        return record_sets[record_set]

    matches = [r for r in record_sets if getattr(r, "uuid", None) == record_set]
    if not matches:
        raise ValueError(f"No record_set with uuid={record_set} found.")
    return matches[0]


def _load_wdi_export(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Load a WDI export CSV. Some exports have a few metadata lines above the header.
    This tries to find the real header row.
    """
    df = _try_read_wdi_direct(path, **read_csv_kwargs)
    if df is not None:
        return df

    header_line = _find_wdi_header_line(path, encoding=read_csv_kwargs.get("encoding", "utf-8"))
    if header_line is None:
        return pd.read_csv(path, skiprows=4, **read_csv_kwargs)

    return pd.read_csv(path, skiprows=header_line, **read_csv_kwargs)


def _try_read_wdi_direct(path: Path, **read_csv_kwargs: Any) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, **read_csv_kwargs)
    except Exception:
        return None

    needed = {"Country Name", "Country Code", "Indicator Name", "Indicator Code"}
    if needed.issubset(set(df.columns)):
        return df
    return None


def _find_wdi_header_line(path: Path, *, encoding: str) -> Optional[int]:
    with path.open("r", encoding=encoding, errors="replace") as f:
        for i in range(20):
            line = f.readline()
            if not line:
                break
            if "Country Name" in line and "Indicator Name" in line and "Indicator Code" in line:
                return i
    return None


# ----------------------------
# Standardization helpers
# ----------------------------

def _standardize_who(df: pd.DataFrame, country_col: Optional[str], year_col: Optional[str]) -> pd.DataFrame:
    """Rename WHO keys to `country`/`year`, and standardize basic types."""
def _standardize_who(df: pd.DataFrame, country_col: Optional[str], year_col: Optional[str]) -> pd.DataFrame:
    """
    Standardize WHO dataset column names and key columns.

    - Renames country/year to `country`, `year`
    - Cleans all column names (strip, lowercase, underscores)
    - Converts year to integer
    """
    out = df.copy()

    # find key columns 
    country_col, year_col = _who_key_cols(out, country_col, year_col)

    # normalize ALL column names 
    cleaned_cols = {}

    for col in out.columns:
        new = str(col)

        # remove leading/trailing spaces
        new = new.strip()

        # lowercase
        new = new.lower()

        # replace "/" with space first (for HIV/AIDS)
        new = new.replace("/", " ")

        # replace hyphens with space
        new = new.replace("-", " ")

        # collapse multiple spaces
        new = " ".join(new.split())

        # replace spaces with underscores
        new = new.replace(" ", "_")

        cleaned_cols[col] = new

    out = out.rename(columns=cleaned_cols)

    # rename keys explicitly 
    out = out.rename(columns={
        cleaned_cols[country_col]: "country",
        cleaned_cols[year_col]: "year",
    })

    # final type fixes 
    out["country"] = out["country"].astype(str).str.strip()
    out["year"] = _to_year_int(out["year"])

    return out



def _who_key_cols(df: pd.DataFrame, country_col: Optional[str], year_col: Optional[str]) -> Tuple[str, str]:
    if country_col is None:
        country_col = "Country" if "Country" in df.columns else "country"
    if year_col is None:
        year_col = "Year" if "Year" in df.columns else "year"

    if country_col not in df.columns or year_col not in df.columns:
        raise KeyError(f"WHO data must include country/year columns. Found: {list(df.columns)}")
    return country_col, year_col


def _standardize_wb(df: pd.DataFrame, country_col: Optional[str], year_col: Optional[str]) -> pd.DataFrame:
    """
    Standardize World Bank panel dataset column names and keys.

    - Renames country/year to `country`, `year`
    - Cleans all column names (strip, lowercase, underscores)
    - Keeps identifiers (region, income_group, country_code)
    """
    out = df.copy()

    # find key columns 
    country_col, year_col = _wb_key_cols(out, country_col, year_col)

    # clean ALL column names 
    cleaned_cols = {}

    for col in out.columns:
        new = str(col)
        new = new.strip()
        new = new.lower()

        # remove % symbol
        new = new.replace("%", "percent")

        # remove parentheses if appear
        new = new.replace("(", "").replace(")", "")

        # fix slash if any
        new = new.replace("/", " ")

        # collapse spaces
        new = " ".join(new.split())

        # underscores
        new = new.replace(" ", "_")

        cleaned_cols[col] = new

    out = out.rename(columns=cleaned_cols)

    # fix important known names 
    rename_special = {
        cleaned_cols.get(country_col): "country",
        cleaned_cols.get(year_col): "year",
        "country_code": "country_code",
        "region": "region",
        "incomegroup": "income_group",
        "life_expectancy_world_bank": "life_expectancy_wb",
        "prevelance_of_undernourishment": "undernourishment",  # shorten
        "health_expenditure_percent": "health_expenditure_percent",
        "education_expenditure_percent": "education_expenditure_percent",
        "noncommunicable": "noncommunicable_disease",
    }

    # only rename those that exist
    rename_special = {k: v for k, v in rename_special.items() if k in out.columns}
    out = out.rename(columns=rename_special)

    # final type cleanup 
    out["country"] = out["country"].astype(str).str.strip()
    out["year"] = _to_year_int(out["year"])

    return out



def _wb_key_cols(df: pd.DataFrame, country_col: Optional[str], year_col: Optional[str]) -> Tuple[str, str]:
    if country_col is None:
        if "Country Name" in df.columns:
            country_col = "Country Name"
        elif "country" in df.columns:
            country_col = "country"
        else:
            country_col = "Country"

    if year_col is None:
        year_col = "Year" if "Year" in df.columns else "year"

    if country_col not in df.columns or year_col not in df.columns:
        raise KeyError(f"WB data must include country/year columns. Found: {list(df.columns)}")
    return country_col, year_col


def _standardize_wdi(df: pd.DataFrame) -> pd.DataFrame:
    """Rename WDI identifier columns to simpler names and trim strings."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    needed = {"Country Name", "Country Code", "Indicator Name", "Indicator Code"}
    if not needed.issubset(set(out.columns)):
        raise KeyError(f"WDI export missing expected columns. Found: {list(out.columns)}")

    # rename identifiers
    out = out.rename(
        columns={
            "Country Name": "country",
            "Country Code": "country_code",
            "Indicator Name": "indicator_name",
            "Indicator Code": "indicator_code",
        }
    )

    # remove empty column "Unnamed: 69" that appears because of last comma
    out = out.loc[:, ~out.columns.str.contains("^Unnamed")]

    # strip whitespace
    out["country"] = out["country"].astype(str).str.strip()
    out["country_code"] = out["country_code"].astype(str).str.strip()
    out["indicator_code"] = out["indicator_code"].astype(str).str.strip()

    return out



# ----------------------------
# Merge helpers
# ----------------------------

def _add_key_stats(
    diag: Dict[str, Any],
    who: pd.DataFrame,
    wb: pd.DataFrame,
    merged: pd.DataFrame,
    country_key: str,
    year_key: str,
) -> None:
    """Add simple key coverage stats to diagnostics."""
    who_keys = set(zip(who[country_key], who[year_key]))
    wb_keys = set(zip(wb[country_key], wb[year_key]))
    merged_keys = set(zip(merged[country_key], merged[year_key]))

    diag["who_unique_keys"] = int(len(who_keys))
    diag["wb_unique_keys"] = int(len(wb_keys))
    diag["merged_unique_keys"] = int(len(merged_keys))
    diag["keys_lost_from_who"] = int(len(who_keys - merged_keys))
    diag["keys_lost_from_wb"] = int(len(wb_keys - merged_keys))


def _merge_wdi(
    merged: pd.DataFrame,
    wdi: pd.DataFrame,
    how: Literal["inner", "left", "right", "outer"],
    on: Tuple[str, str],
    year_key: str,
    wdi_suffix: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Merge an already-pivoted WDI panel into the merged WHO/WB panel."""
    country_key, _ = on
    _require_cols(wdi, [country_key, year_key], "include_wdi")

    wdi_df = wdi.copy()
    wdi_df[year_key] = _to_year_int(wdi_df[year_key])

    before = len(merged)
    out = merged.merge(
        wdi_df,
        how=how,
        on=list(on),
        validate="one_to_one",
        suffixes=("", wdi_suffix),
    )

    return out, {
        "wdi_rows": int(len(wdi_df)),
        "merged_rows_after_wdi": int(len(out)),
        "rows_change_after_wdi": int(len(out) - before),
    }


# ----------------------------
# Small utilities
# ----------------------------

def _require_cols(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}. Present: {list(df.columns)}")


def _to_year_int(series: pd.Series) -> pd.Series:
    """Convert a year-like column to pandas nullable Int64. Non-numeric becomes NA."""
    s = pd.to_numeric(series, errors="coerce")
    return s.astype("Int64")
