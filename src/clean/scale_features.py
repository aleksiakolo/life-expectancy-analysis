import numpy as np
from sklearn.preprocessing import StandardScaler

def log_transform(df, cols):
    for col in cols:
        df[f"log_{col}"] = np.log1p(df[col])
    return df


def standardize(df, cols):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    return df_scaled, scaler