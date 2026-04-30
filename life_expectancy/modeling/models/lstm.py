from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """Small LSTM regressor for country-year sequence forecasting.

    Args:
        input_size: Number of input features per time step.
        hidden_size: LSTM hidden size.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Tensor of shape `(batch, window, input_size)`.

        Returns:
            Prediction tensor of shape `(batch, 1)`.
        """
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]

        return self.head(last_hidden)


def build_lstm_regressor(
    *,
    input_size: int,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
) -> LSTMRegressor:
    """Build an LSTM regressor.

    Args:
        input_size: Number of input features per time step.
        hidden_size: LSTM hidden size.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability.

    Returns:
        Configured LSTMRegressor.
    """
    return LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
