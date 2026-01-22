import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def to_snake_case(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[^\w\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    return col

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    return df

def log_schema(df: pd.DataFrame, name: str):
    print(f"\n--- {name} SCHEMA ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c}")
    print("-------------------")

def load_who():
    path = RAW_DIR / "who" / "WHO_life_expectancy_raw.csv"
    df = pd.read_csv(path)
    df = standardize_columns(df)
    log_schema(df, "WHO")
    df.to_parquet(INTERIM_DIR / "who_clean_base.parquet", index=False)

def load_world_bank():
    path = RAW_DIR / "world_bank" / "WB_indicators_raw.csv"
    df = pd.read_csv(path)
    df = standardize_columns(df)
    log_schema(df, "World Bank")
    df.to_parquet(INTERIM_DIR / "wb_clean_base.parquet", index=False)

if __name__ == "__main__":
    load_who()
    load_world_bank()
