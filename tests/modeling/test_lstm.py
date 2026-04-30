import torch

from life_expectancy.modeling.model.lstm import LSTMRegressor, build_lstm_regressor


def test_build_lstm_regressor() -> None:
    model = build_lstm_regressor(
        input_size=3,
        hidden_size=8,
        num_layers=1,
        dropout=0.1,
    )

    assert isinstance(model, LSTMRegressor)


def test_lstm_forward_shape() -> None:
    model = build_lstm_regressor(
        input_size=3,
        hidden_size=8,
        num_layers=1,
        dropout=0.1,
    )

    x = torch.randn(5, 4, 3)
    y = model(x)

    assert y.shape == (5, 1)
