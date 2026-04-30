import matplotlib.pyplot as plt
import pandas as pd
import pytest

from life_expectancy.analysis.diagnostics import (
    attach_predictions,
    group_error_table,
    plot_predicted_vs_actual,
    plot_residual_hist,
    plot_residuals_vs_predicted,
    root_mean_square,
    time_slice_error_table,
    worst_errors_table,
)


def make_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["A", "B", "C"],
            "year": [2010, 2010, 2011],
            "region": ["R1", "R1", "R2"],
            "income_group": ["Low", "Low", "High"],
            "y_true": [10.0, 20.0, 30.0],
            "y_pred": [12.0, 18.0, 35.0],
            "error": [2.0, -2.0, 5.0],
            "abs_error": [2.0, 2.0, 5.0],
        }
    )


def test_attach_predictions() -> None:
    base = pd.DataFrame({"country": ["A", "B"]})

    result = attach_predictions(
        base,
        y_true=[10.0, 20.0],
        y_pred=[12.0, 18.0],
    )

    assert result["error"].tolist() == [2.0, -2.0]
    assert result["abs_error"].tolist() == [2.0, 2.0]


def test_attach_predictions_length_mismatch_raises() -> None:
    base = pd.DataFrame({"country": ["A", "B"]})

    with pytest.raises(ValueError):
        attach_predictions(
            base,
            y_true=[10.0],
            y_pred=[12.0, 18.0],
        )


def test_worst_errors_table() -> None:
    result = worst_errors_table(make_prediction_df(), n=2)

    assert result["country"].tolist() == ["C", "A"]
    assert result["abs_error"].tolist() == [5.0, 2.0]


def test_worst_errors_table_custom_columns() -> None:
    result = worst_errors_table(
        make_prediction_df(),
        n=1,
        cols_to_show=["country", "abs_error"],
    )

    assert result.columns.tolist() == ["country", "abs_error"]


def test_worst_errors_table_missing_abs_error_raises() -> None:
    df = pd.DataFrame({"y_true": [1.0]})

    with pytest.raises(KeyError):
        worst_errors_table(df)


def test_group_error_table() -> None:
    result = group_error_table(make_prediction_df(), group_col="region")

    assert result.loc[0, "region"] == "R2"
    assert result.loc[0, "mae"] == 5.0
    assert result.loc[1, "region"] == "R1"


def test_time_slice_error_table() -> None:
    result = time_slice_error_table(make_prediction_df(), year_col="year")

    assert result["year"].tolist() == [2010, 2011]
    assert result.loc[0, "n"] == 2
    assert result.loc[1, "mae"] == 5.0


def test_root_mean_square() -> None:
    result = root_mean_square(pd.Series([3.0, 4.0]))

    assert result == pytest.approx((25 / 2) ** 0.5)


def test_plot_predicted_vs_actual() -> None:
    ax = plot_predicted_vs_actual([1, 2, 3], [1, 2, 4])

    assert ax.get_xlabel() == "Actual"
    assert ax.get_ylabel() == "Predicted"
    plt.close(ax.figure)


def test_plot_residuals_vs_predicted() -> None:
    ax = plot_residuals_vs_predicted([1, 2, 3], [1, 2, 4])

    assert ax.get_xlabel() == "Predicted"
    assert ax.get_ylabel() == "Residual (pred - actual)"
    plt.close(ax.figure)


def test_plot_residual_hist() -> None:
    ax = plot_residual_hist([1, 2, 3], [1, 2, 4], bins=5)

    assert ax.get_xlabel() == "Residual (pred - actual)"
    assert ax.get_ylabel() == "Count"
    plt.close(ax.figure)
