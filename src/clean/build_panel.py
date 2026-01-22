from src.clean.filter_entities import *
from src.clean.impute import *
from src.clean.merge import merge_who_wb

def build_clean_panel(who, wb):
    df = merge_who_wb(who, wb)

    df = drop_sparse_features(df, threshold=0.5)
    df = drop_sparse_countries(df, min_years=5)

    slow_moving = ["life_expectancy", "schooling"]
    df = interpolate_time_series(df, slow_moving)

    df = grouped_median_impute(
        df,
        group_col="income_group",
        cols=df.select_dtypes("number").columns
    )

    return df
