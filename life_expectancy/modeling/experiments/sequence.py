from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from life_expectancy.modeling.model.lstm import build_lstm_regressor
from life_expectancy.modeling.train_eval import regression_metrics

Summary = dict[str, Any]
SequenceSplit = dict[str, Any]


@dataclass(frozen=True)
class LSTMEvalResult:
    """Evaluation result for an LSTM sequence model.

    Attributes:
        model_name: Model name.
        split_name: Split name.
        n_train: Number of training sequences.
        n_val: Number of validation sequences.
        n_test: Number of test sequences.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        r2: R-squared score.
        best_epoch: Epoch with best validation loss.
    """

    model_name: str
    split_name: str
    n_train: int
    n_val: int
    n_test: int
    rmse: float
    mae: float
    r2: float
    best_epoch: int


def make_flat_lag_dataframe(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str = "life_expectancy",
    country_col: str = "country",
    year_col: str = "year",
    lags: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Create a flat lag-feature DataFrame by country.

    Args:
        df: Input country-year panel.
        feature_cols: Feature columns to lag.
        target_col: Target column name.
        country_col: Country column name.
        year_col: Year column name.
        lags: Lag values to create.

    Returns:
        DataFrame with lagged target and lagged feature columns.

    Raises:
        KeyError: If required columns are missing.
    """
    require_columns(df, [country_col, year_col, target_col, *feature_cols])

    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="raise").astype(int)
    out = out.sort_values([country_col, year_col]).reset_index(drop=True)

    grouped = out.groupby(country_col, group_keys=False)
    created_cols: list[str] = []

    for col in [target_col, *feature_cols]:
        for lag in sorted(set(lags)):
            new_col = f"{col}_lag{lag}"
            out[new_col] = grouped[col].shift(lag)
            created_cols.append(new_col)

    out = out.dropna(subset=[*created_cols, target_col]).reset_index(drop=True)

    return out


def build_country_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str = "life_expectancy",
    country_col: str = "country",
    year_col: str = "year",
    window: int = 3,
    include_target_history: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Build fixed-window country sequences.

    For each country, this creates samples where the previous `window` rows are
    used to predict the current target value.

    Args:
        df: Input country-year panel.
        feature_cols: Feature columns included in each time step.
        target_col: Target column name.
        country_col: Country column name.
        year_col: Year column name.
        window: Number of previous years used as model input.
        include_target_history: Whether to include past target values as the
            first sequence channel.

    Returns:
        Tuple containing X array, y array, metadata DataFrame, and sequence
        column names.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If no valid sequences can be created.
    """
    require_columns(df, [country_col, year_col, target_col, *feature_cols])

    if window <= 0:
        raise ValueError("window must be positive.")

    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="raise").astype(int)
    out = out.sort_values([country_col, year_col]).reset_index(drop=True)

    seq_cols = list(feature_cols)
    if include_target_history and target_col not in seq_cols:
        seq_cols = [target_col, *seq_cols]

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    meta_cols = [
        col
        for col in [country_col, "country_code", "region", "income_group", year_col]
        if col in out.columns
    ]

    for _, group in out.groupby(country_col):
        group = group.sort_values(year_col).reset_index(drop=True)

        x_values = group[seq_cols].to_numpy(dtype=np.float32)
        y_values = group[target_col].to_numpy(dtype=np.float32)

        if len(group) <= window:
            continue

        for index in range(window, len(group)):
            x_seq = x_values[index - window : index]
            y_value = y_values[index]

            if np.isnan(x_seq).any() or np.isnan(y_value):
                continue

            x_rows.append(x_seq)
            y_rows.append(float(y_value))
            meta_rows.append(group.loc[index, meta_cols].to_dict())

    if not x_rows:
        raise ValueError("No valid sequences were created.")

    x = np.stack(x_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return x, y, meta_df, seq_cols


def split_sequences_timeaware(
    x: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    *,
    year_col: str = "year",
    test_years: int = 3,
    val_years: int = 1,
) -> SequenceSplit:
    """Split sequences into train, validation, and test blocks by year.

    Args:
        x: Sequence input array.
        y: Target array.
        meta_df: Metadata DataFrame with one row per sequence.
        year_col: Year column in metadata.
        test_years: Number of latest years used for testing.
        val_years: Number of latest pre-test years used for validation.

    Returns:
        Dictionary containing sequence arrays and metadata splits.

    Raises:
        KeyError: If the year column is missing.
        ValueError: If split settings create empty blocks.
    """
    require_columns(meta_df, [year_col])

    if test_years <= 0:
        raise ValueError("test_years must be positive.")

    if val_years <= 0:
        raise ValueError("val_years must be positive.")

    years = pd.to_numeric(meta_df[year_col], errors="raise").astype(int)
    max_year = int(years.max())
    test_cutoff = max_year - test_years + 1

    test_mask = years >= test_cutoff
    train_val_mask = years < test_cutoff

    train_val_years = sorted(years[train_val_mask].unique())

    if len(train_val_years) <= val_years:
        raise ValueError("Not enough training years for validation split.")

    val_year_values = train_val_years[-val_years:]
    val_mask = years.isin(val_year_values) & train_val_mask
    train_mask = train_val_mask & ~years.isin(val_year_values)

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Sequence split produced an empty block.")

    return {
        "X_train": x[train_mask],
        "y_train": y[train_mask],
        "X_val": x[val_mask],
        "y_val": y[val_mask],
        "X_test": x[test_mask],
        "y_test": y[test_mask],
        "meta_train": meta_df.loc[train_mask].reset_index(drop=True),
        "meta_val": meta_df.loc[val_mask].reset_index(drop=True),
        "meta_test": meta_df.loc[test_mask].reset_index(drop=True),
        "cutoff": test_cutoff,
        "max_year": max_year,
        "val_year_values": list(val_year_values),
    }


def scale_sequence_splits(
    split: SequenceSplit,
) -> tuple[SequenceSplit, StandardScaler]:
    """Scale sequence features using training data only.

    Args:
        split: Sequence split dictionary.

    Returns:
        Tuple containing updated split dictionary and fitted scaler.
    """
    x_train = split["X_train"]
    x_val = split["X_val"]
    x_test = split["X_test"]

    n_features = x_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, n_features))

    split = dict(split)
    split["X_train"] = transform_sequence_array(x_train, scaler)
    split["X_val"] = transform_sequence_array(x_val, scaler)
    split["X_test"] = transform_sequence_array(x_test, scaler)

    return split, scaler


def transform_sequence_array(
    x: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a fitted scaler to a 3D sequence array.

    Args:
        x: Sequence array with shape `(samples, window, features)`.
        scaler: Fitted StandardScaler.

    Returns:
        Scaled sequence array with the same shape.
    """
    n_features = x.shape[2]
    x_scaled = scaler.transform(x.reshape(-1, n_features))

    return x_scaled.reshape(x.shape).astype(np.float32)


def run_lstm_time_experiment(
    x: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    *,
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
    target_history_channel: int | None = 0,
    clip_grad_norm: float = 1.0,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    torch_threads: int = 1,
) -> tuple[Summary, pd.DataFrame, nn.Module, StandardScaler, SequenceSplit]:
    """Run a time-aware LSTM sequence experiment.

    Args:
        x: Sequence input array.
        y: Target array.
        meta_df: Metadata DataFrame.
        year_col: Year column name.
        test_years: Number of latest years used for testing.
        val_years: Number of latest pre-test years used for validation.
        hidden_size: LSTM hidden size.
        num_layers: Number of LSTM layers.
        learning_rate: AdamW learning rate.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        random_state: Random seed.
        predict_delta: Whether to predict change from previous target value.
        target_history_channel: Index of target-history channel in `x`.
        clip_grad_norm: Gradient clipping threshold.
        weight_decay: AdamW weight decay.
        dropout: Dropout probability.
        torch_threads: Number of CPU threads used by PyTorch.

    Returns:
        Tuple containing result row, prediction DataFrame, fitted model, scaler,
        and sequence split dictionary.

    Raises:
        ValueError: If delta prediction is requested without target history.
    """
    set_torch_seed(random_state=random_state, torch_threads=torch_threads)

    split = split_sequences_timeaware(
        x,
        y,
        meta_df,
        year_col=year_col,
        test_years=test_years,
        val_years=val_years,
    )
    split, scaler = scale_sequence_splits(split)

    train_target, val_target = build_lstm_training_targets(
        split,
        predict_delta=predict_delta,
        target_history_channel=target_history_channel,
    )

    train_loader = make_sequence_loader(
        split["X_train"],
        train_target,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = make_sequence_loader(
        split["X_val"],
        val_target,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cpu")
    model = build_lstm_regressor(
        input_size=split["X_train"].shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    best_epoch = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=patience,
        clip_grad_norm=clip_grad_norm,
        weight_decay=weight_decay,
        device=device,
    )

    predictions = predict_lstm(
        model=model,
        x_test=split["X_test"],
        device=device,
    )

    if predict_delta:
        last_target = get_last_target_values(
            split["X_test"],
            target_history_channel=target_history_channel,
        )
        predictions = predictions + last_target

    y_true = split["y_test"]
    metrics = regression_metrics(y_true, predictions)

    result = LSTMEvalResult(
        model_name="lstm_delta" if predict_delta else "lstm",
        split_name="time_sequence",
        n_train=len(split["X_train"]),
        n_val=len(split["X_val"]),
        n_test=len(split["X_test"]),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        best_epoch=best_epoch,
    )

    pred_df = build_lstm_prediction_df(
        meta_df=split["meta_test"],
        y_true=y_true,
        y_pred=predictions,
    )

    return asdict(result), pred_df, model, scaler, split


def build_lstm_training_targets(
    split: SequenceSplit,
    *,
    predict_delta: bool,
    target_history_channel: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train and validation targets for LSTM training.

    Args:
        split: Sequence split dictionary.
        predict_delta: Whether to predict deltas instead of target levels.
        target_history_channel: Channel containing target history.

    Returns:
        Train and validation target arrays.

    Raises:
        ValueError: If delta prediction lacks target-history channel.
    """
    y_train = split["y_train"].astype(np.float32)
    y_val = split["y_val"].astype(np.float32)

    if not predict_delta:
        return y_train, y_val

    train_last = get_last_target_values(
        split["X_train"],
        target_history_channel=target_history_channel,
    )
    val_last = get_last_target_values(
        split["X_val"],
        target_history_channel=target_history_channel,
    )

    return (
        (y_train - train_last).astype(np.float32),
        (y_val - val_last).astype(np.float32),
    )


def get_last_target_values(
    x: np.ndarray,
    *,
    target_history_channel: int | None,
) -> np.ndarray:
    """Return latest target-history values from sequence inputs.

    Args:
        x: Sequence input array.
        target_history_channel: Target-history channel index.

    Returns:
        Last target-history values.

    Raises:
        ValueError: If target-history channel is missing.
    """
    if target_history_channel is None:
        raise ValueError(
            "Delta prediction requires target history in the sequence input."
        )

    return x[:, -1, target_history_channel].astype(np.float32)


def make_sequence_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader for sequence arrays.

    Args:
        x: Sequence input array.
        y: Target array.
        batch_size: Batch size.
        shuffle: Whether to shuffle batches.

    Returns:
        DataLoader over TensorDataset.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_lstm_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    patience: int,
    clip_grad_norm: float,
    weight_decay: float,
    device: torch.device,
) -> int:
    """Train LSTM with early stopping.

    Args:
        model: LSTM model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        learning_rate: Optimizer learning rate.
        epochs: Maximum epochs.
        patience: Early stopping patience.
        clip_grad_norm: Gradient clipping threshold.
        weight_decay: Optimizer weight decay.
        device: Torch device.

    Returns:
        Best validation epoch.
    """
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        run_lstm_training_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            clip_grad_norm=clip_grad_norm,
            device=device,
        )

        val_loss = evaluate_lstm_loss(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_epoch


def run_lstm_training_epoch(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad_norm: float,
    device: torch.device,
) -> None:
    """Run one LSTM training epoch.

    Args:
        model: LSTM model.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        clip_grad_norm: Gradient clipping threshold.
        device: Torch device.
    """
    model.train()

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()


def evaluate_lstm_loss(
    *,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate validation loss.

    Args:
        model: LSTM model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Mean validation loss.
    """
    model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            losses.append(float(loss.item()))

    return float(np.mean(losses))


def predict_lstm(
    *,
    model: nn.Module,
    x_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Generate LSTM predictions.

    Args:
        model: Trained LSTM model.
        x_test: Test sequence array.
        device: Torch device.

    Returns:
        Prediction array.
    """
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(x_tensor).cpu().numpy().reshape(-1)

    return predictions.astype(np.float32)


def build_lstm_prediction_df(
    *,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Build prediction DataFrame for LSTM output.

    Args:
        meta_df: Metadata DataFrame for test sequences.
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Prediction DataFrame.
    """
    pred_df = meta_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["error"])

    return pred_df


def set_torch_seed(
    *,
    random_state: int,
    torch_threads: int,
) -> None:
    """Set PyTorch randomness and thread count.

    Args:
        random_state: Random seed.
        torch_threads: Number of torch CPU threads.
    """
    torch.manual_seed(random_state)
    torch.set_num_threads(torch_threads)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate required columns.

    Args:
        df: DataFrame to validate.
        columns: Required columns.

    Raises:
        KeyError: If required columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
