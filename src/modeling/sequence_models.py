from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def make_flat_lag_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "life_expectancy_final",
    country_col: str = "country",
    year_col: str = "year",
    lags: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="raise").astype(int)
    out = out.sort_values([country_col, year_col]).reset_index(drop=True)

    grouped = out.groupby(country_col, group_keys=False)
    created_cols = []

    seq_sources = [target_col] + feature_cols
    for col in seq_sources:
        for lag in lags:
            new_col = f"{col}_lag{lag}"
            out[new_col] = grouped[col].shift(lag)
            created_cols.append(new_col)

    out = out.dropna(subset=created_cols + [target_col]).reset_index(drop=True)
    return out


def build_country_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "life_expectancy_final",
    country_col: str = "country",
    year_col: str = "year",
    window: int = 3,
    include_target_history: bool = True,
):
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="raise").astype(int)
    out = out.sort_values([country_col, year_col]).reset_index(drop=True)

    seq_cols = list(feature_cols)
    if include_target_history and target_col not in seq_cols:
        seq_cols = [target_col] + seq_cols

    X_list = []
    y_list = []
    meta_rows = []

    meta_keep = [c for c in [country_col, "country_code", "region", "income_group", year_col] if c in out.columns]

    for _, g in out.groupby(country_col):
        g = g.sort_values(year_col).reset_index(drop=True)

        X_vals = g[seq_cols].to_numpy(dtype=np.float32)
        y_vals = g[target_col].to_numpy(dtype=np.float32)

        if len(g) <= window:
            continue

        for i in range(window, len(g)):
            x_seq = X_vals[i - window:i]
            y_now = y_vals[i]

            if np.isnan(x_seq).any() or np.isnan(y_now):
                continue

            X_list.append(x_seq)
            y_list.append(y_now)
            meta_rows.append(g.loc[i, meta_keep].to_dict())

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df, seq_cols


def split_sequences_timeaware(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    year_col: str = "year",
    test_years: int = 3,
    val_years: int = 1,
):
    years = pd.to_numeric(meta_df[year_col], errors="raise").astype(int)
    max_year = int(years.max())
    test_cutoff = max_year - (test_years - 1)

    test_mask = years >= test_cutoff
    trainval_mask = years < test_cutoff

    trainval_years = sorted(years[trainval_mask].unique())
    if len(trainval_years) <= val_years:
        raise ValueError("Not enough training years to create a validation sequence split.")

    val_year_values = trainval_years[-val_years:]
    val_mask = years.isin(val_year_values) & trainval_mask
    train_mask = trainval_mask & (~years.isin(val_year_values))

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "meta_train": meta_df.loc[train_mask].reset_index(drop=True),
        "meta_val": meta_df.loc[val_mask].reset_index(drop=True),
        "meta_test": meta_df.loc[test_mask].reset_index(drop=True),
        "cutoff": test_cutoff,
        "max_year": max_year,
        "val_year_values": list(val_year_values),
    }


def scale_sequence_splits(split_dict: dict):
    X_train = split_dict["X_train"]
    X_val = split_dict["X_val"]
    X_test = split_dict["X_test"]

    n_features = X_train.shape[2]
    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)

    def transform(arr):
        arr_flat = arr.reshape(-1, n_features)
        arr_scaled = scaler.transform(arr_flat)
        return arr_scaled.reshape(arr.shape).astype(np.float32)

    split_dict["X_train"] = transform(X_train)
    split_dict["X_val"] = transform(X_val)
    split_dict["X_test"] = transform(X_test)

    return split_dict, scaler


@dataclass
class LSTMEvalResult:
    model_name: str
    split_name: str
    n_train: int
    n_val: int
    n_test: int
    rmse: float
    mae: float
    r2: float
    best_epoch: int


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_tiny_lstm_timeaware(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    year_col: str = "year",
    test_years: int = 3,
    val_years: int = 1,
    hidden_size: int = 32,
    num_layers: int = 1,
    learning_rate: float = 5e-4,
    batch_size: int = 128,
    epochs: int = 60,
    patience: int = 8,
    random_state: int = 42,
    predict_delta: bool = True,
    delta_target_in_input: bool = True,  # assumes target history included in X (your include_target_history=True)
    clip_grad_norm: float = 1.0,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    torch_threads: int = 1,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)
    torch.set_num_threads(torch_threads)

    split_dict = split_sequences_timeaware(
        X, y, meta_df, year_col=year_col, test_years=test_years, val_years=val_years
    )
    split_dict, scaler = scale_sequence_splits(split_dict)

    # IMPORTANT: We assume the first channel is target history if include_target_history=True
    # because build_country_sequences does: seq_cols = [target_col] + feature_cols
    # If you change that, update this logic.
    Xtr_np = split_dict["X_train"]
    Xva_np = split_dict["X_val"]
    Xte_np = split_dict["X_test"]

    ytr_np = split_dict["y_train"]
    yva_np = split_dict["y_val"]
    yte_np = split_dict["y_test"]

    if predict_delta:
        if not delta_target_in_input:
            raise ValueError("predict_delta=True requires target history included in X.")
        last_y_train = Xtr_np[:, -1, 0].astype(np.float32)
        last_y_val = Xva_np[:, -1, 0].astype(np.float32)
        last_y_test = Xte_np[:, -1, 0].astype(np.float32)

        ytr_tgt = (ytr_np - last_y_train).astype(np.float32)
        yva_tgt = (yva_np - last_y_val).astype(np.float32)
        yte_tgt = (yte_np - last_y_test).astype(np.float32)
    else:
        ytr_tgt = ytr_np.astype(np.float32)
        yva_tgt = yva_np.astype(np.float32)
        yte_tgt = yte_np.astype(np.float32)

    X_train = torch.tensor(Xtr_np, dtype=torch.float32)
    y_train = torch.tensor(ytr_tgt, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(Xva_np, dtype=torch.float32)
    y_val = torch.tensor(yva_tgt, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(Xte_np, dtype=torch.float32)
    y_test = torch.tensor(yte_tgt, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    class TinyLSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            h_last = out[:, -1, :]
            return self.head(h_last)

    device = torch.device("cpu")
    model = TinyLSTMRegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Huber is more forgiving than MSE if you have outliers / scale issues
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            if clip_grad_norm is not None and clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(float(loss.item()))

        mean_val = float(np.mean(val_losses))

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_delta = model(X_test.to(device)).cpu().numpy().reshape(-1)

    if predict_delta:
        last_y_test = Xte_np[:, -1, 0].astype(np.float32)
        preds = pred_delta + last_y_test
        y_true = yte_np
    else:
        preds = pred_delta
        y_true = yte_np

    m = regression_metrics(y_true, preds)

    result = LSTMEvalResult(
        model_name="TinyLSTM_delta" if predict_delta else "TinyLSTM",
        split_name="time_sequence",
        n_train=len(split_dict["X_train"]),
        n_val=len(split_dict["X_val"]),
        n_test=len(split_dict["X_test"]),
        rmse=m["rmse"],
        mae=m["mae"],
        r2=m["r2"],
        best_epoch=best_epoch,
    )

    pred_df = split_dict["meta_test"].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = preds
    pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["error"])

    row = {
        "model_name": result.model_name,
        "split_name": result.split_name,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "n_test": result.n_test,
        "rmse": result.rmse,
        "mae": result.mae,
        "r2": result.r2,
        "best_epoch": result.best_epoch,
    }

    return row, pred_df, model, scaler, split_dict