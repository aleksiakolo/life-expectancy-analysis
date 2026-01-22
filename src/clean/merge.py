import pandas as pd

def merge_who_wb(who, wb):
    """
    Merge WHO and World Bank datasets on (country, year).
    """
    merged = who.merge(
        wb,
        on=["country", "year"],
        how="inner",
        validate="one_to_one"
    )
    return merged