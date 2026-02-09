from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

JoinKey = Tuple[str, str]  # ("country", "year")


# ---------------------------------------------------------------------
# WHO cleaning
# ---------------------------------------------------------------------

def clean_who(
    df: pd.DataFrame,
    *,
    key: JoinKey = ("country", "year"),
    target_col: str = "life_expectancy",
    drop_missing_target: bool = True,
    duplicate_policy: Literal["keep_first", "drop_all", "mean_aggregate"] = "keep_first",
    min_years_per_country: int = 5,
    life_expectancy_bounds: Tuple[float, float] = (0.0, 100.0),
    clip_life_expectancy: bool = False,
    immunization_clip_0_100: bool = True,
    negative_to_na_cols: Optional[Sequence[str]] = None,
    feature_missingness_drop_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean WHO panel data (one row per country-year).

    Assumes WHO data is already standardized so it has `country` and `year`.
    Returns (clean_df, summary_dict) for reporting.
    """
    country_col, year_col = key
    out = df.copy()

    out = _prep_panel_keys(out, country_col, year_col)

    summary: Dict[str, Any] = {
        "input_rows": int(len(out)),
        "input_cols": int(out.shape[1]),
        "duplicate_policy": duplicate_policy,
        "min_years_per_country": min_years_per_country,
        "drop_missing_target": drop_missing_target,
        "feature_missingness_drop_threshold": feature_missingness_drop_threshold,
    }

    _require_columns(out, [country_col, year_col, target_col], name="who_df")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    out, dup_summary = _handle_duplicates_who(out, country_col, year_col, duplicate_policy)
    summary.update(dup_summary)

    out, le_summary = _clean_life_expectancy(
        out,
        target_col=target_col,
        bounds=life_expectancy_bounds,
        clip=clip_life_expectancy,
    )
    summary.update(le_summary)

    if negative_to_na_cols:
        out = _negative_to_na(out, negative_to_na_cols)

    if immunization_clip_0_100:
        out, imm_summary = _clip_immunization(out)
        summary.update(imm_summary)
    else:
        summary["immunization_cols_clipped"] = []

    out, drop_summary = _drop_missing_target(out, target_col, drop_missing_target)
    summary.update(drop_summary)

    out, coverage_summary = _filter_country_coverage(out, country_col, target_col, min_years_per_country)
    summary.update(coverage_summary)

    out, miss_summary = _drop_features_by_missingness(
        out,
        threshold=feature_missingness_drop_threshold,
        protected={country_col, year_col, target_col},
    )
    summary.update(miss_summary)

    summary["output_rows"] = int(len(out))
    summary["output_cols"] = int(out.shape[1])
    summary["year_min"] = int(out[year_col].min()) if out[year_col].notna().any() else None
    summary["year_max"] = int(out[year_col].max()) if out[year_col].notna().any() else None

    return out, summary


def _handle_duplicates_who(
    df: pd.DataFrame,
    country_col: str,
    year_col: str,
    duplicate_policy: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dup_mask = df.duplicated(subset=[country_col, year_col], keep=False)
    duplicate_rows = int(dup_mask.sum())

    summary: Dict[str, Any] = {"duplicate_rows": duplicate_rows}

    if duplicate_rows == 0:
        summary["rows_after_dedup"] = int(len(df))
        return df, summary

    if duplicate_policy == "keep_first":
        out = df.sort_values([country_col, year_col]).drop_duplicates([country_col, year_col], keep="first")
    elif duplicate_policy == "drop_all":
        out = df.loc[~dup_mask].copy()
    elif duplicate_policy == "mean_aggregate":
        out = _mean_aggregate_duplicates(df, country_col, year_col)
    else:
        raise ValueError(f"Unknown duplicate_policy: {duplicate_policy}")

    summary["rows_after_dedup"] = int(len(out))
    return out, summary


def _mean_aggregate_duplicates(df: pd.DataFrame, country_col: str, year_col: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c not in [country_col, year_col]]

    agg: Dict[str, str] = {}
    for c in df.columns:
        if c in [country_col, year_col]:
            continue
        if c in numeric_cols:
            agg[c] = "mean"
        else:
            agg[c] = "first"

    return df.groupby([country_col, year_col], as_index=False).agg(agg)


def _clean_life_expectancy(
    df: pd.DataFrame,
    *,
    target_col: str,
    bounds: Tuple[float, float],
    clip: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lo, hi = bounds
    invalid = (df[target_col] <= lo) | (df[target_col] > hi)
    invalid = invalid.fillna(False)

    summary: Dict[str, Any] = {"life_expectancy_invalid_count": int(invalid.sum())}

    out = df.copy()
    if clip:
        out[target_col] = out[target_col].clip(lower=lo, upper=hi)
    else:
        out.loc[invalid, target_col] = np.nan

    return out, summary


def _negative_to_na(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] < 0, c] = np.nan
    return out


def _clip_immunization(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    immun_cols = [c for c in out.columns if _looks_like_immunization_col(c)]

    for c in immun_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").clip(0, 100)

    return out, {"immunization_cols_clipped": immun_cols}


def _drop_missing_target(df: pd.DataFrame, target_col: str, enabled: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not enabled:
        return df, {"rows_dropped_missing_target": 0}

    before = len(df)
    out = df.loc[df[target_col].notna()].copy()
    return out, {"rows_dropped_missing_target": int(before - len(out))}


def _filter_country_coverage(
    df: pd.DataFrame,
    country_col: str,
    target_col: str,
    min_years: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not min_years or min_years <= 0:
        return df, {"rows_dropped_low_coverage": 0, "countries_kept": int(df[country_col].nunique())}

    counts = df.groupby(country_col)[target_col].size()
    keep = counts[counts >= min_years].index

    before = len(df)
    out = df[df[country_col].isin(keep)].copy()

    return out, {
        "rows_dropped_low_coverage": int(before - len(out)),
        "countries_kept": int(len(keep)),
    }


# ---------------------------------------------------------------------
# WB panel cleaning (Kaggle panel)
# ---------------------------------------------------------------------

def clean_wb(
    df: pd.DataFrame,
    *,
    key: JoinKey = ("country", "year"),
    indicator_subset: Optional[Sequence[str]] = None,
    winsorize_cols: Optional[Sequence[str]] = None,
    winsorize_limits: Tuple[float, float] = (0.01, 0.99),
    log_transform_cols: Optional[Sequence[str]] = None,
    log_epsilon: float = 1e-9,
    feature_missingness_drop_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the World Bank Kaggle panel dataset.

    Keeps the panel structure (country, year) and optionally:
      - selects a subset of indicators
      - clips outliers
      - adds log_ columns for chosen variables
      - drops features above a missingness threshold
    """
    country_col, year_col = key
    out = df.copy()

    out = _prep_panel_keys(out, country_col, year_col)

    summary: Dict[str, Any] = {
        "input_rows": int(len(out)),
        "input_cols": int(out.shape[1]),
        "feature_missingness_drop_threshold": feature_missingness_drop_threshold,
        "winsorize_cols": list(winsorize_cols) if winsorize_cols else [],
        "log_transform_cols": list(log_transform_cols) if log_transform_cols else [],
    }

    if indicator_subset is not None:
        out, subset_summary = _apply_wb_subset(out, country_col, year_col, indicator_subset)
        summary.update(subset_summary)

    out = _convert_numeric_columns_wb(out, country_col, year_col)

    if winsorize_cols:
        out = _winsorize(out, winsorize_cols, winsorize_limits)

    if log_transform_cols:
        out = _add_log_columns(out, log_transform_cols, log_epsilon)

    out, miss_summary = _drop_features_by_missingness(
        out,
        threshold=feature_missingness_drop_threshold,
        protected={country_col, year_col},
    )
    summary.update(miss_summary)

    dup_mask = out.duplicated(subset=[country_col, year_col], keep=False)
    summary["duplicate_rows"] = int(dup_mask.sum())

    summary["output_rows"] = int(len(out))
    summary["output_cols"] = int(out.shape[1])
    summary["year_min"] = int(out[year_col].min()) if out[year_col].notna().any() else None
    summary["year_max"] = int(out[year_col].max()) if out[year_col].notna().any() else None

    return out, summary


def _apply_wb_subset(
    df: pd.DataFrame,
    country_col: str,
    year_col: str,
    indicator_subset: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    keep = {country_col, year_col}
    for c in ["Country Code", "country_code", "Region", "IncomeGroup", "income_group", "region"]:
        if c in df.columns:
            keep.add(c)

    keep.update(indicator_subset)
    missing_requested = [c for c in indicator_subset if c not in df.columns]

    out = df[[c for c in df.columns if c in keep]].copy()

    return out, {
        "cols_after_subset": int(out.shape[1]),
        "missing_requested_cols": missing_requested,
    }


def _convert_numeric_columns_wb(df: pd.DataFrame, country_col: str, year_col: str) -> pd.DataFrame:
    out = df.copy()
    categorical = {
        country_col, year_col,
        "Region", "IncomeGroup", "Country Code", "country_code", "region", "income_group"
    }

    for c in out.columns:
        if c in categorical:
            continue
        if out[c].dtype == "object":
            out[c] = pd.to_numeric(out[c], errors="ignore")

    return out


def _winsorize(df: pd.DataFrame, cols: Sequence[str], limits: Tuple[float, float]) -> pd.DataFrame:
    out = df.copy()
    lo_q, hi_q = limits

    for c in cols:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        lo = x.quantile(lo_q)
        hi = x.quantile(hi_q)
        out[c] = x.clip(lower=lo, upper=hi)

    return out


def _add_log_columns(df: pd.DataFrame, cols: Sequence[str], eps: float) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        x = x.where(x >= 0)  # negative values -> NaN for log
        out[f"log_{c}"] = np.log(x + eps)
    return out


# ---------------------------------------------------------------------
# WDI pivoting
# ---------------------------------------------------------------------

def pivot_wdi(
    wdi_df: pd.DataFrame,
    *,
    indicator_codes: Optional[Sequence[str]] = None,
    indicator_code_col: str = "indicator_code",
    country_col: str = "country",
    country_code_col: str = "country_code",
    value_name: str = "value",
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    wide_prefix: str = "",
    drop_all_nan_rows: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert a WDI export (wide years) into a country-year panel (wide indicators).

    Input (standardized WDI):
      - country, country_code, indicator_code
      - year columns like "1960", "1961", ...

    Output:
      - country, year, and one column per indicator_code
    """
    df = wdi_df.copy()
    required = {country_col, country_code_col, indicator_code_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"WDI pivot missing required columns: {sorted(missing)}")

    year_cols = [c for c in df.columns if _is_year_col(c)]
    if not year_cols:
        raise ValueError("No year columns detected in WDI data (expected columns like '1960', '1961', ...).")

    if indicator_codes is not None:
        df = df[df[indicator_code_col].isin(indicator_codes)].copy()

    summary: Dict[str, Any] = {
        "input_rows": int(len(wdi_df)),
        "rows_after_indicator_filter": int(len(df)),
        "n_year_columns": int(len(year_cols)),
        "indicator_codes": list(indicator_codes) if indicator_codes is not None else None,
    }

    long = _wdi_to_long(
        df,
        year_cols=year_cols,
        country_col=country_col,
        country_code_col=country_code_col,
        indicator_code_col=indicator_code_col,
        value_name=value_name,
    )
    long = _filter_year_range(long, year_min, year_max)

    panel = _wdi_long_to_panel(long, country_col, indicator_code_col, value_name)
    panel = _prefix_indicator_cols(panel, country_col, "year", wide_prefix)

    if drop_all_nan_rows:
        panel, drop_summary = _drop_all_nan_rows(panel, country_col, "year")
        summary.update(drop_summary)

    summary["output_rows"] = int(len(panel))
    summary["output_cols"] = int(panel.shape[1])

    return panel, summary


def _wdi_to_long(
    df: pd.DataFrame,
    *,
    year_cols: Sequence[str],
    country_col: str,
    country_code_col: str,
    indicator_code_col: str,
    value_name: str,
) -> pd.DataFrame:
    long = df.melt(
        id_vars=[country_col, country_code_col, indicator_code_col],
        value_vars=list(year_cols),
        var_name="year",
        value_name=value_name,
    )
    long["year"] = _coerce_year_to_int(long["year"])
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long


def _filter_year_range(long: pd.DataFrame, year_min: Optional[int], year_max: Optional[int]) -> pd.DataFrame:
    out = long
    if year_min is not None:
        out = out[out["year"] >= year_min]
    if year_max is not None:
        out = out[out["year"] <= year_max]
    return out


def _wdi_long_to_panel(
    long: pd.DataFrame,
    country_col: str,
    indicator_code_col: str,
    value_name: str,
) -> pd.DataFrame:
    panel = (
        long.pivot_table(
            index=[country_col, "year"],
            columns=indicator_code_col,
            values=value_name,
            aggfunc="first",
        )
        .reset_index()
    )
    panel.columns = [str(c) for c in panel.columns]
    return panel


def _prefix_indicator_cols(panel: pd.DataFrame, country_col: str, year_col: str, prefix: str) -> pd.DataFrame:
    if not prefix:
        return panel
    rename = {c: f"{prefix}{c}" for c in panel.columns if c not in {country_col, year_col}}
    return panel.rename(columns=rename)


def _drop_all_nan_rows(panel: pd.DataFrame, country_col: str, year_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    value_cols = [c for c in panel.columns if c not in {country_col, year_col}]
    before = len(panel)
    out = panel.dropna(subset=value_cols, how="all")
    return out, {"rows_dropped_all_nan": int(before - len(out))}


# ---------------------------------------------------------------------
# Missingness diagnostics
# ---------------------------------------------------------------------

def missingness_table(
    df: pd.DataFrame,
    *,
    sort_desc: bool = True,
    round_to: int = 4,
) -> pd.DataFrame:
    """Overall missingness per column (fraction + counts)."""
    n = len(df)
    miss_frac = df.isna().mean()
    miss_count = df.isna().sum()
    non_miss = n - miss_count

    out = pd.DataFrame(
        {
            "column": miss_frac.index,
            "missing_fraction": miss_frac.values,
            "missing_count": miss_count.values,
            "non_missing_count": non_miss.values,
        }
    )
    out["missing_fraction"] = out["missing_fraction"].round(round_to)

    if sort_desc:
        out = out.sort_values("missing_fraction", ascending=False).reset_index(drop=True)

    return out


def missingness_by_year(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
    cols: Sequence[str] = ("life_expectancy",),
    round_to: int = 4,
) -> pd.DataFrame:
    """Missingness fraction by year for selected columns."""
    _require_columns(df, [year_col], name="df")
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found for missingness_by_year.")

    rows = []
    for year, grp in df.groupby(year_col):
        row = {"year": year, "n_rows": int(len(grp))}
        for c in cols:
            row[f"{c}_missing_fraction"] = float(grp[c].isna().mean())
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    for c in cols:
        out[f"{c}_missing_fraction"] = out[f"{c}_missing_fraction"].round(round_to)

    return out


def missingness_by_country(
    df: pd.DataFrame,
    *,
    country_col: str = "country",
    cols: Sequence[str] = ("life_expectancy",),
    top_n: Optional[int] = 15,
    round_to: int = 4,
) -> pd.DataFrame:
    """Missingness fraction by country for selected columns."""
    _require_columns(df, [country_col], name="df")
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found for missingness_by_country.")

    rows = []
    for country, grp in df.groupby(country_col):
        row = {"country": country, "n_rows": int(len(grp))}
        for c in cols:
            row[f"{c}_missing_fraction"] = float(grp[c].isna().mean())
        rows.append(row)

    out = pd.DataFrame(rows).sort_values([f"{cols[0]}_missing_fraction"], ascending=False).reset_index(drop=True)
    for c in cols:
        out[f"{c}_missing_fraction"] = out[f"{c}_missing_fraction"].round(round_to)

    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Imputation policy for merged panel
# ---------------------------------------------------------------------

def impute_panel(
    df: pd.DataFrame,
    *,
    key: JoinKey = ("country", "year"),
    target_col: str = "life_expectancy",
    numeric_only: bool = True,
    interpolate_cols: Optional[Sequence[str]] = None,
    interpolate_limit: int = 2,
    interpolate_method: str = "linear",
    median_impute_cols: Optional[Sequence[str]] = None,
    group_cols: Optional[Sequence[str]] = None,
    group_median_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Impute missing values in a country-year panel.

    This does three optional steps (in order):
      1) interpolate within each country (short gaps)
      2) fill with group medians (e.g., region/income group)
      3) fill with global medians

    The target column is never imputed.
    """
    country_col, year_col = key
    out = df.copy()

    _require_columns(out, [country_col, year_col], name="df")
    if target_col not in out.columns:
        raise KeyError(f"Target column '{target_col}' is missing from panel data.")

    out[year_col] = _coerce_year_to_int(out[year_col])

    summary: Dict[str, Any] = {
        "input_rows": int(len(out)),
        "interpolate_cols": list(interpolate_cols) if interpolate_cols else [],
        "median_impute_cols": list(median_impute_cols) if median_impute_cols else [],
        "group_cols": list(group_cols) if group_cols else [],
        "group_median_cols": list(group_median_cols) if group_median_cols else [],
    }

    eligible_numeric = _eligible_numeric_cols(out, target_col, numeric_only)

    if interpolate_cols:
        filled = _interpolate_by_country(
            out,
            country_col=country_col,
            year_col=year_col,
            cols=interpolate_cols,
            eligible_numeric=eligible_numeric,
            method=interpolate_method,
            limit=interpolate_limit,
        )
        summary["interpolate_filled_counts"] = filled
    else:
        summary["interpolate_filled_counts"] = {}

    if group_cols and group_median_cols:
        filled = _group_median_impute(
            out,
            group_cols=group_cols,
            cols=group_median_cols,
            eligible_numeric=eligible_numeric,
        )
        summary["group_median_filled_counts"] = filled
    else:
        summary["group_median_filled_counts"] = {}

    if median_impute_cols:
        filled = _global_median_impute(
            out,
            cols=median_impute_cols,
            eligible_numeric=eligible_numeric,
        )
        summary["median_filled_counts"] = filled
    else:
        summary["median_filled_counts"] = {}

    summary["output_rows"] = int(len(out))
    summary["target_missing_count_after"] = int(out[target_col].isna().sum())

    return out, summary


def _eligible_numeric_cols(df: pd.DataFrame, target_col: str, numeric_only: bool) -> Sequence[str]:
    if not numeric_only:
        cols = [c for c in df.columns if c != target_col]
        return cols

    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in cols:
        cols.remove(target_col)
    return cols


def _interpolate_by_country(
    df: pd.DataFrame,
    *,
    country_col: str,
    year_col: str,
    cols: Sequence[str],
    eligible_numeric: Sequence[str],
    method: str,
    limit: int,
) -> Dict[str, int]:
    filled: Dict[str, int] = {}

    for c in cols:
        if c not in df.columns:
            continue
        if c not in eligible_numeric:
            continue

        before = int(df[c].isna().sum())

        df[c] = (
            df.sort_values([country_col, year_col])
            .groupby(country_col, group_keys=False)[c]
            .apply(lambda s: s.interpolate(method=method, limit=limit, limit_direction="both"))
        )

        after = int(df[c].isna().sum())
        filled[c] = before - after

    return filled


def _group_median_impute(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    cols: Sequence[str],
    eligible_numeric: Sequence[str],
) -> Dict[str, int]:
    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        raise KeyError(f"group_cols missing from DataFrame: {missing_group_cols}")

    filled: Dict[str, int] = {}

    for c in cols:
        if c not in df.columns:
            continue
        if c not in eligible_numeric:
            continue

        before = int(df[c].isna().sum())
        med = df.groupby(list(group_cols))[c].transform("median")
        df[c] = df[c].fillna(med)
        after = int(df[c].isna().sum())
        filled[c] = before - after

    return filled


def _global_median_impute(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    eligible_numeric: Sequence[str],
) -> Dict[str, int]:
    filled: Dict[str, int] = {}

    for c in cols:
        if c not in df.columns:
            continue
        if c not in eligible_numeric:
            continue

        before = int(df[c].isna().sum())
        median_val = df[c].median(skipna=True)
        df[c] = df[c].fillna(median_val)
        after = int(df[c].isna().sum())
        filled[c] = before - after

    return filled


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _prep_panel_keys(df: pd.DataFrame, country_col: str, year_col: str) -> pd.DataFrame:
    out = df.copy()
    if country_col in out.columns:
        out[country_col] = out[country_col].astype(str).str.strip()
    if year_col in out.columns:
        out[year_col] = _coerce_year_to_int(out[year_col])
    return out


def _drop_features_by_missingness(
    df: pd.DataFrame,
    *,
    threshold: Optional[float],
    protected: set,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if threshold is None:
        return df, {"dropped_feature_cols": [], "n_dropped_feature_cols": 0}

    if not (0.0 < threshold < 1.0):
        raise ValueError("feature_missingness_drop_threshold should be a fraction in (0,1), e.g. 0.6")

    miss = df.isna().mean()
    drop_cols = [c for c, m in miss.items() if (m > threshold and c not in protected)]
    out = df.drop(columns=drop_cols)

    return out, {"dropped_feature_cols": drop_cols, "n_dropped_feature_cols": int(len(drop_cols))}


# ---------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------

def _coerce_year_to_int(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.astype("Int64")


def _is_year_col(col: Any) -> bool:
    try:
        c = str(col).strip()
        if len(c) != 4:
            return False
        year = int(c)
        return 1950 <= year <= 2050
    except Exception:
        return False


def _looks_like_immunization_col(col: str) -> bool:
    c = col.lower().strip()
    return any(k in c for k in ["polio", "diphtheria", "hepatitis", "measles", "immun", "vacc"])


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}. Present: {list(df.columns)}")
