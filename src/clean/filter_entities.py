from src.clean.inspect_missingness import *

def drop_sparse_features(df, threshold=0.5):
    """
    Drop columns with missingness above threshold.
    """
    keep_cols = df.columns[df.isna().mean() < threshold]
    return df[keep_cols]


def drop_sparse_countries(df, min_years=5):
    """
    Drop countries with too few observed years.
    """
    counts = df.groupby("country")["year"].nunique()
    keep_countries = counts[counts >= min_years].index
    return df[df["country"].isin(keep_countries)]
