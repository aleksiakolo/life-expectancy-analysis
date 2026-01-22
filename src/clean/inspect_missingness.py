import pandas as pd

def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missingness statistics per column.
    """
    return pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": df.isna().mean() * 100
    }).sort_values("pct_missing", ascending=False)


def missingness_by_group(df: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Compute missingness by country or year.
    """
    return (
        df.groupby(group_col)
          .apply(lambda x: x.isna().mean())
          .mean(axis=1)
          .sort_values(ascending=False)
    )