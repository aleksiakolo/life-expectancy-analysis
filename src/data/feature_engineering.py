from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

DEFAULT_TARGET = "life_expectancy_final"
DEFAULT_YEAR_COL = "year"
LEAKAGE_COLS = ("life_expectancy", "life_expectancy_wb")
DROP_ID_COLS = ("country_code")

LOG_CANDIDATES = ("gdp", "population", "co2")
MISSINGNESS_FLAG_CANDIDATES = ("gdp", "schooling", "sanitation", "adult_mortality")
DEFAULT_INTERACTIONS = (
    ("schooling", "status_flag"),
    ("gdp_log1p", "status_flag"),
)

DEFAULT_MANUAL_CANDIDATES = (
    "year",
    "adult_mortality",
    "schooling",
    "gdp_log1p",
    "sanitation",
    "hiv_aids",
    "income_composition_of_resources",
    "health_expenditure_percent",
    "under_five_deaths",
    "status_flag",
)


def add_log_features(
    df: pd.DataFrame,
    log_candidates: Sequence[str] = LOG_CANDIDATES,
) -> pd.DataFrame:
    out = df.copy()
    for col in log_candidates:
        if col in out.columns:
            clipped = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
            out[f"{col}_log1p"] = np.log1p(clipped)
    return out


def add_status_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "status" in out.columns and "status_flag" not in out.columns:
        out["status_flag"] = (
            out["status"].astype(str).str.lower().str.contains("developed")
        ).astype(int)
    return out


def add_missingness_flags(
    df: pd.DataFrame,
    cols: Sequence[str] = MISSINGNESS_FLAG_CANDIDATES,
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[f"{col}_missing_flag"] = out[col].isna().astype(int)
    return out


def add_interaction_features(
    df: pd.DataFrame,
    interaction_pairs: Sequence[tuple[str, str]] = DEFAULT_INTERACTIONS,
) -> pd.DataFrame:
    out = df.copy()
    for a, b in interaction_pairs:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = pd.to_numeric(out[a], errors="coerce") * pd.to_numeric(
                out[b], errors="coerce"
            )
    return out


def prepare_numeric_model_frame(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    year_col: str = DEFAULT_YEAR_COL,
) -> pd.DataFrame:
    """
    Build a numeric-only modeling frame:
    - keeps target
    - drops leakage cols and ID cols
    - creates status_flag if needed
    - adds log transforms
    - adds optional missingness flags
    - adds interaction features
    """
    out = df.copy()

    if target_col not in out.columns:
        raise KeyError(f"{target_col!r} not found in dataframe")

    out = add_status_flag(out)
    out = add_log_features(out)
    out = add_missingness_flags(out)
    out = add_interaction_features(out)

    # Drop leakage + IDs + original status string
    drop_cols = {target_col, *LEAKAGE_COLS, *DROP_ID_COLS, "status"}
    candidate_cols = [c for c in out.columns if c not in drop_cols]

    # Keep only numeric predictors + target
    numeric_predictors = [
        c for c in candidate_cols if pd.api.types.is_numeric_dtype(out[c])
    ]

    final_cols = list(dict.fromkeys([*numeric_predictors, target_col]))
    clean = out[final_cols].copy()
    clean = clean.dropna(subset=[target_col]).reset_index(drop=True)

    # Force year numeric if present
    if year_col in clean.columns:
        clean[year_col] = pd.to_numeric(clean[year_col], errors="raise").astype(int)

    return clean


def compute_correlation_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    return df[list(features)].corr(numeric_only=True)


def correlation_prune(
    df: pd.DataFrame,
    features: Sequence[str],
    threshold: float = 0.90,
    protected: Sequence[str] = ("year", "status_flag"),
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Greedy correlation pruning on absolute correlation matrix.
    Returns kept_features, dropped_features, corr_matrix.
    """
    corr = df[list(features)].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    protected = set(protected)
    to_drop: set[str] = set()

    for col in upper.columns:
        high_corr_rows = upper.index[upper[col] >= threshold].tolist()
        for row in high_corr_rows:
            if row in protected and col in protected:
                continue
            if row in protected and col not in protected:
                to_drop.add(col)
            elif col in protected and row not in protected:
                to_drop.add(row)
            else:
                to_drop.add(col)

    kept = [f for f in features if f not in to_drop]
    dropped = sorted(to_drop)
    return kept, dropped, corr


def compute_vif_table(
    df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """
    Compute VIF on numeric features only.
    Missing values are median-imputed for the VIF computation step.
    """
    X = df[list(features)].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    vif_rows = []
    for i, col in enumerate(X.columns):
        vif_val = variance_inflation_factor(X.values, i)
        vif_rows.append({"feature": col, "vif": float(vif_val)})

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


def iterative_vif_prune(
    df: pd.DataFrame,
    features: Sequence[str],
    vif_threshold: float = 10.0,
    protected: Sequence[str] = ("year", "status_flag"),
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Iteratively remove the highest-VIF non-protected feature
    until all VIFs are <= threshold or only protected features remain.
    """
    keep = list(features)
    protected = set(protected)
    dropped: list[str] = []

    while True:
        if len(keep) <= 1:
            break

        vif_df = compute_vif_table(df, keep)
        worst_row = vif_df.iloc[0]
        worst_feature = worst_row["feature"]
        worst_vif = float(worst_row["vif"])

        if worst_vif <= vif_threshold:
            return keep, dropped, vif_df

        droppable = vif_df.loc[~vif_df["feature"].isin(protected), "feature"].tolist()
        if not droppable:
            return keep, dropped, vif_df

        feature_to_remove = droppable[0]
        keep.remove(feature_to_remove)
        dropped.append(feature_to_remove)

    final_vif = compute_vif_table(df, keep)
    return keep, dropped, final_vif


def choose_manual_feature_set_c(
    available_features: Sequence[str],
    manual_candidates: Sequence[str] = DEFAULT_MANUAL_CANDIDATES,
    min_size: int = 5,
    max_size: int = 10,
) -> list[str]:
    chosen = [f for f in manual_candidates if f in available_features]

    if len(chosen) < min_size:
        extras = [f for f in available_features if f not in chosen]
        chosen.extend(extras[: max(0, min_size - len(chosen))])

    return chosen[:max_size]


def build_feature_sets_abc(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    year_col: str = DEFAULT_YEAR_COL,
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
) -> tuple[pd.DataFrame, Dict[str, list[str]], Dict[str, object]]:
    """
    Returns:
        model_df: numeric modeling dataframe
        feature_sets: {"A": [...], "B": [...], "C": [...]}
        meta: useful metadata for reporting/debugging
    """
    model_df = prepare_numeric_model_frame(df, target_col=target_col, year_col=year_col)

    all_features = [c for c in model_df.columns if c != target_col]

    # A = full numeric cleaned set
    feature_set_a = list(all_features)

    # B = correlation-pruned
    feature_set_b, corr_dropped, corr_matrix = correlation_prune(
        model_df,
        feature_set_a,
        threshold=corr_threshold,
        protected=(year_col, "status_flag"),
    )

    # C = VIF-pruned + manually interpretable subset
    vif_kept, vif_dropped, final_vif_df = iterative_vif_prune(
        model_df,
        feature_set_b,
        vif_threshold=vif_threshold,
        protected=(year_col, "status_flag"),
    )

    feature_set_c = choose_manual_feature_set_c(vif_kept)

    feature_sets = {
        "A": feature_set_a,
        "B": feature_set_b,
        "C": feature_set_c,
    }

    meta = {
        "corr_threshold": corr_threshold,
        "vif_threshold": vif_threshold,
        "corr_dropped": corr_dropped,
        "vif_dropped": vif_dropped,
        "final_vif_table": final_vif_df.to_dict(orient="records"),
    }

    return model_df, feature_sets, meta


def save_feature_sets_json(
    feature_sets: Dict[str, list[str]],
    meta: Dict[str, object],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "feature_sets": feature_sets,
        "meta": meta,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path