from __future__ import annotations

from typing import Sequence

from sklearn.neural_network import MLPRegressor


def mlp_regressor(
    *,
    hidden_layer_sizes: Sequence[int] = (128, 64),
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 1e-4,
    batch_size: str | int = "auto",
    learning_rate: str = "adaptive",
    learning_rate_init: float = 1e-3,
    max_iter: int = 500,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 10,
    random_state: int = 42,
) -> MLPRegressor:
    """Construct a sklearn MLPRegressor.

    This is a simple feedforward neural network baseline. It works best
    with scaled numeric inputs (use "standard" or "robust" scaling).

    Args:
        hidden_layer_sizes: Sizes of hidden layers.
        activation: Activation function ("relu", "tanh", etc.).
        solver: Optimization solver ("adam", "sgd", "lbfgs").
        alpha: L2 regularization strength.
        batch_size: Batch size or "auto".
        learning_rate: Learning rate schedule.
        learning_rate_init: Initial learning rate.
        max_iter: Maximum number of iterations.
        early_stopping: Whether to use early stopping.
        validation_fraction: Fraction of training data for validation.
        n_iter_no_change: Early stopping patience.
        random_state: Random seed.

    Returns:
        Configured MLPRegressor instance.
    """
    return MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )
