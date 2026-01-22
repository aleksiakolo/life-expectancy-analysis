import pandas as pd

def interpolate_time_series(df, value_cols):
    """
    Interpolate within country for slow-moving indicators.
    """
    df = df.sort_values(["country", "year"])
    for col in value_cols:
        df[col] = (
            df.groupby("country")[col]
              .apply(lambda x: x.interpolate(limit=2))
        )
    return df


def grouped_median_impute(df, group_col, cols):
    """
    Impute using group-level medians (e.g. region or income group).
    """
    for col in cols:
        df[col] = df[col].fillna(
            df.groupby(group_col)[col].transform("median")
        )
    return df
