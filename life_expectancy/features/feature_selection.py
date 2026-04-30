from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from life_expectancy.features.feature_engineering import prepare_numeric_model_frame

Summary = dict[str, Any]
FeatureSets = dict[str, list[str]]


def compute_correlation_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute a numeric correlation matrix for selected features.

    Args:
        df: Input modeling DataFrame.
        features: Feature columns to include.

    Returns:
        Correlation matrix for the selected features.
    """
    return df[features].corr(numeric_only=True)


def high_correlation_pairs(
    df: pd.DataFrame, *, threshold: float = 0.85
) -> pd.DataFrame:
    """Find highly correlated numeric feature pairs.

    Args:
        df: Input DataFrame.
        threshold: Absolute correlation threshold.

    Returns:
        DataFrame with feature pairs and correlation values.
    """
    corr = df.corr(numeric_only=True)
    rows = []

    for i, left_col in enumerate(corr.columns):
        for j in range(i):
            right_col = corr.columns[j]
            value = corr.iloc[i, j]

            if abs(value) > threshold:
                rows.append(
                    {
                        "feature_1": left_col,
                        "feature_2": right_col,
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )


def correlation_prune(
    df: pd.DataFrame,
    features: list[str],
    *,
    threshold: float,
    protected: list[str],
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Remove highly correlated features using a greedy rule.

    Args:
        df: Input modeling DataFrame.
        features: Candidate feature columns.
        threshold: Absolute correlation threshold above which one feature is dropped.
        protected: Features that should be kept when possible.

    Returns:
        Tuple containing kept features, dropped features, and absolute correlation
        matrix.
    """
    corr = df[features].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    protected_set = set(protected)
    to_drop: set[str] = set()

    for col in upper.columns:
        high_corr_rows = upper.index[upper[col] >= threshold].tolist()

        for row in high_corr_rows:
            if row in protected_set and col in protected_set:
                continue
            if row in protected_set:
                to_drop.add(col)
            elif col in protected_set:
                to_drop.add(row)
            else:
                to_drop.add(col)

    kept = [feature for feature in features if feature not in to_drop]
    dropped = sorted(to_drop)

    return kept, dropped, corr


def compute_vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute variance inflation factors for selected numeric features.

    Missing and infinite values are replaced before VIF calculation so the table can
    be used as a diagnostic instead of failing on incomplete rows.

    Args:
        df: Input modeling DataFrame.
        features: Feature columns to include.

    Returns:
        DataFrame with one row per feature and its VIF value.
    """
    x = df[features].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))

    rows: list[dict[str, float | str]] = []

    for index, col in enumerate(x.columns):
        vif_value = variance_inflation_factor(x.values, index)
        rows.append({"feature": col, "vif": float(vif_value)})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def iterative_vif_prune(
    df: pd.DataFrame,
    features: list[str],
    *,
    vif_threshold: float,
    protected: list[str],
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Iteratively remove high-VIF features.

    Args:
        df: Input modeling DataFrame.
        features: Candidate feature columns.
        vif_threshold: Maximum allowed VIF.
        protected: Features that should not be removed.

    Returns:
        Tuple containing kept features, dropped features, and the final VIF table.
    """
    kept = list(features)
    protected_set = set(protected)
    dropped: list[str] = []

    while len(kept) > 1:
        vif_table = compute_vif_table(df, kept)
        worst_vif = float(vif_table.iloc[0]["vif"])

        if worst_vif <= vif_threshold:
            return kept, dropped, vif_table

        droppable = vif_table.loc[
            ~vif_table["feature"].isin(protected_set),
            "feature",
        ].tolist()

        if not droppable:
            return kept, dropped, vif_table

        feature_to_remove = droppable[0]
        kept.remove(feature_to_remove)
        dropped.append(feature_to_remove)

    final_vif_table = compute_vif_table(df, kept)
    return kept, dropped, final_vif_table


def choose_manual_feature_set(
    available_features: list[str],
    *,
    manual_candidates: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """Choose an interpretable manual feature subset.

    Args:
        available_features: Features available after prior selection steps.
        manual_candidates: Preferred feature names in priority order.
        min_size: Minimum desired feature set size.
        max_size: Maximum desired feature set size.

    Returns:
        Selected feature list.
    """
    chosen = [feature for feature in manual_candidates if feature in available_features]

    if len(chosen) < min_size:
        extras = [feature for feature in available_features if feature not in chosen]
        needed = max(0, min_size - len(chosen))
        chosen.extend(extras[:needed])

    return chosen[:max_size]


def build_feature_sets_abc(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, FeatureSets, Summary]:
    """Build feature sets A, B, and C using project configuration.

    Feature set A contains all numeric model features. Feature set B removes highly
    correlated features. Feature set C applies VIF pruning and then selects a smaller
    interpretable manual subset.

    Args:
        df: Processed panel DataFrame.
        config: Full project configuration dictionary containing a `features` section.

    Returns:
        Tuple containing:
            - Numeric modeling DataFrame.
            - Feature sets dictionary with keys `A`, `B`, and `C`.
            - Metadata dictionary for reporting/debugging.
    """
    feature_config = config["features"]

    target_col = feature_config.get("target_col", "life_expectancy")
    year_col = feature_config.get("year_col", "year")
    corr_threshold = feature_config.get("corr_threshold", 0.90)
    vif_threshold = feature_config.get("vif_threshold", 10.0)
    protected = feature_config.get("protected_features", [year_col, "status_flag"])
    manual_candidates = feature_config.get("manual_candidates", [])
    manual_min_size = feature_config.get("manual_min_size", 5)
    manual_max_size = feature_config.get("manual_max_size", 10)

    model_df = prepare_numeric_model_frame(df, config)
    all_features = [col for col in model_df.columns if col != target_col]

    feature_set_a = list(all_features)

    feature_set_b, corr_dropped, corr_matrix = correlation_prune(
        model_df,
        feature_set_a,
        threshold=corr_threshold,
        protected=protected,
    )

    vif_kept, vif_dropped, final_vif_table = iterative_vif_prune(
        model_df,
        feature_set_b,
        vif_threshold=vif_threshold,
        protected=protected,
    )

    feature_set_c = choose_manual_feature_set(
        vif_kept,
        manual_candidates=manual_candidates,
        min_size=manual_min_size,
        max_size=manual_max_size,
    )

    feature_sets: FeatureSets = {
        "A": feature_set_a,
        "B": feature_set_b,
        "C": feature_set_c,
    }

    meta: Summary = {
        "target_col": target_col,
        "corr_threshold": corr_threshold,
        "vif_threshold": vif_threshold,
        "corr_dropped": corr_dropped,
        "vif_dropped": vif_dropped,
        "corr_matrix": corr_matrix.to_dict(),
        "final_vif_table": final_vif_table.to_dict(orient="records"),
    }

    return model_df, feature_sets, meta


def save_feature_sets_json(
    feature_sets: FeatureSets,
    meta: Summary,
    out_path: str | Path,
) -> Path:
    """Save feature sets and metadata to a JSON file.

    Args:
        feature_sets: Feature sets dictionary.
        meta: Metadata dictionary.
        out_path: Output JSON path.

    Returns:
        Resolved output path.
    """
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "feature_sets": feature_sets,
        "meta": meta,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return output_path
