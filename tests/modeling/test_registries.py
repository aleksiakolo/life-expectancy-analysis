import pytest
from sklearn.base import RegressorMixin

from life_expectancy.modeling.registries import (
    get_default_model_registry,
    get_model,
    get_model_registry_from_config,
    get_model_spec,
    get_scale_numeric,
    get_selected_model_registry,
)


def test_get_default_model_registry_contains_expected_models() -> None:
    registry = get_default_model_registry(random_state=7)

    assert set(registry) == {
        "mean",
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "hgb",
        "rf",
        "extra_trees",
        "mlp",
    }


def test_registry_specs_have_model_and_scale_numeric() -> None:
    registry = get_default_model_registry()

    for spec in registry.values():
        assert "model" in spec
        assert "scale_numeric" in spec
        assert spec["scale_numeric"] in {"none", "standard", "robust"}


def test_get_model_spec() -> None:
    spec = get_model_spec("ridge")

    assert "model" in spec
    assert spec["scale_numeric"] == "standard"


def test_get_model_spec_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_model_spec("unknown")


def test_get_model() -> None:
    model = get_model("rf")

    assert isinstance(model, RegressorMixin)


def test_get_scale_numeric() -> None:
    assert get_scale_numeric("ridge") == "standard"
    assert get_scale_numeric("rf") == "none"
    assert get_scale_numeric("mlp") == "robust"


def test_get_selected_model_registry() -> None:
    registry = get_selected_model_registry(
        ["ridge", "rf"],
        random_state=42,
    )

    assert list(registry) == ["ridge", "rf"]


def test_get_selected_model_registry_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_selected_model_registry(["ridge", "bad_model"])


def test_get_model_registry_from_config_all_models() -> None:
    config = {
        "modeling": {
            "registry": {
                "random_state": 42,
            }
        }
    }

    registry = get_model_registry_from_config(config)

    assert "ridge" in registry
    assert "rf" in registry


def test_get_model_registry_from_config_selected_models() -> None:
    config = {
        "modeling": {
            "registry": {
                "random_state": 42,
                "model_names": ["ridge", "extra_trees"],
            }
        }
    }

    registry = get_model_registry_from_config(config)

    assert list(registry) == ["ridge", "extra_trees"]
