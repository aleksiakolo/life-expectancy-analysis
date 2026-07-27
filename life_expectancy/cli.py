from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml

from life_expectancy.analysis.interpretability import (
    permutation_importance_df,
    plot_permutation_bar,
    plot_shap_bar,
    shap_importance,
)
from life_expectancy.data.loading import load_source
from life_expectancy.data.preprocessing import build_processed_dataset
from life_expectancy.data.standardization import standardize
from life_expectancy.data.wdi import attach_panel_metadata_to_wdi, pivot_wdi
from life_expectancy.features.feature_selection import (
    build_feature_sets_abc,
    save_feature_sets_json,
)
from life_expectancy.features.temporal import make_country_lag_features
from life_expectancy.modeling.experiments.boosting import run_boosting_time_experiment
from life_expectancy.modeling.experiments.core import run_time_experiment
from life_expectancy.modeling.experiments.wdi import (
    fit_evaluate_wdi_model,
    make_panel_overlap_split,
)
from life_expectancy.modeling.registries import get_default_model_registry
from life_expectancy.modeling.splits import make_time_split
from life_expectancy.modeling.train_eval import regression_metrics
from life_expectancy.modeling.tuning import (
    AVAILABLE_MODELS,
    bayes_search_results,
    build_tunable_pipeline,
    get_tunable_scale_numeric,
    run_bayes_search_from_config,
)

app = typer.Typer(help="Life expectancy analysis CLI.")

Config = dict[str, Any]
Summary = dict[str, Any]


def load_config(config_path: str | Path) -> Config:
    """Load project config and normalize project root."""
    path = Path(config_path).resolve()

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config.setdefault("project", {})
    config["project"]["root"] = str(path.parent.parent)

    return config


def project_path(config: Config, path: str | Path) -> Path:
    """Resolve a path relative to project root."""
    path = Path(path)

    if path.is_absolute():
        return path

    root = Path(config.get("project", {}).get("root", "."))
    return root / path


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save dictionary as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)

    return output_path


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save DataFrame as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def get_scale_mode(spec: dict[str, Any]) -> str | bool | None:
    """Get scale setting from a model spec."""
    return spec.get("scale_numeric", spec.get("scale_mode"))


def get_common_settings(config: Config) -> dict[str, Any]:
    """Extract common modeling settings."""
    split_config = config.get("modeling", {}).get("split", {})
    feature_config = config.get("features", {})

    return {
        "target_col": feature_config.get(
            "target_col",
            split_config.get("target_col", "life_expectancy"),
        ),
        "year_col": feature_config.get(
            "year_col",
            split_config.get("year_col", "year"),
        ),
        "test_years": split_config.get("test_years", 3),
        "val_years": split_config.get("val_years", 1),
        "random_state": config.get("random_seed", split_config.get("seed", 42)),
    }


def load_or_build_panel(
    config: Config,
    panel_path: str | Path,
) -> tuple[pd.DataFrame, Summary]:
    """Load processed panel if available, otherwise build it."""
    resolved_path = project_path(config, panel_path)

    if resolved_path.exists():
        panel = pd.read_csv(resolved_path)
        return panel, {"loaded_from": str(resolved_path)}

    panel, summary = build_processed_dataset(config)
    save_csv(panel, resolved_path)

    return panel, summary


def run_preprocess(
    config_path: Path,
    output_path: Path,
    summary_path: Path,
) -> None:
    """Run preprocessing workflow."""
    config = load_config(config_path)
    panel, summary = build_processed_dataset(config)

    save_csv(panel, project_path(config, output_path))
    save_json(summary, project_path(config, summary_path))

    typer.echo(f"Saved processed panel: {project_path(config, output_path)}")
    typer.echo(f"Saved summary: {project_path(config, summary_path)}")


def run_features(
    config_path: Path,
    panel_path: Path,
    model_frame_path: Path,
    feature_sets_path: Path,
) -> None:
    """Run feature-set workflow."""
    config = load_config(config_path)

    panel, _ = load_or_build_panel(config, panel_path)
    model_df, feature_sets, meta = build_feature_sets_abc(panel, config)

    save_csv(model_df, project_path(config, model_frame_path))
    save_feature_sets_json(
        feature_sets,
        meta,
        project_path(config, feature_sets_path),
    )

    typer.echo(f"Saved model frame: {project_path(config, model_frame_path)}")
    typer.echo(f"Saved feature sets: {project_path(config, feature_sets_path)}")


def run_train_baselines(
    config_path: Path,
    panel_path: Path,
    output_path: Path,
) -> None:
    """Run baseline model workflow."""
    config = load_config(config_path)
    settings = get_common_settings(config)

    panel, _ = load_or_build_panel(config, panel_path)
    model_df, feature_sets, _ = build_feature_sets_abc(panel, config)

    registry = get_default_model_registry(
        random_state=settings["random_state"],
    )

    model_specs = {
        "RidgeCV": registry["ridge"],
        "LassoCV": registry["lasso"],
        "HistGBR": registry["hgb"],
        "RandomForest": registry["rf"],
        "ExtraTrees": registry["extra_trees"],
    }

    rows = []

    for feature_set_name in ["A", "B", "C"]:
        for model_name, spec in model_specs.items():
            row, _, _, _ = run_time_experiment(
                df=model_df,
                feature_list=feature_sets[feature_set_name],
                target_col=settings["target_col"],
                year_col=settings["year_col"],
                model_name=model_name,
                model=spec["model"],
                scale_numeric=get_scale_mode(spec),
                test_years=settings["test_years"],
                split_label=f"time_{feature_set_name}",
            )
            row["feature_set"] = feature_set_name
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["rmse", "mae", "r2"],
        ascending=[True, True, False],
    )

    save_csv(results, project_path(config, output_path))
    typer.echo(f"Saved baseline results: {project_path(config, output_path)}")


def run_train_boosting(
    config_path: Path,
    panel_path: Path,
    output_path: Path,
    run_log_path: Path,
) -> None:
    """Run external boosting workflow."""
    config = load_config(config_path)
    settings = get_common_settings(config)

    panel, _ = load_or_build_panel(config, panel_path)
    model_df, feature_sets, _ = build_feature_sets_abc(panel, config)

    boosting_config = config.get("modeling", {}).get("boosting", {})
    rows = []

    for config_name, model_config in boosting_config.items():
        if not isinstance(model_config, dict):
            continue

        if not model_config.get("enabled", False):
            typer.echo(f"Skipping {config_name}: disabled.")
            continue

        feature_set_name = model_config.get("feature_set", "B")

        row, _, _, _ = run_boosting_time_experiment(
            df=model_df,
            feature_list=feature_sets[feature_set_name],
            target_col=settings["target_col"],
            year_col=settings["year_col"],
            model_family=model_config["model_family"],
            model_name=model_config["model_name"],
            test_years=settings["test_years"],
            val_years=settings["val_years"],
            scale_numeric=model_config.get("scale_numeric", "none"),
            model_params=model_config.get("model_params", {}).copy(),
            run_log_path=project_path(config, run_log_path),
            id_cols=[settings["year_col"]],
        )

        row["feature_set"] = feature_set_name
        rows.append(row)

    if not rows:
        typer.echo("No enabled boosting experiments found.")
        return

    results = pd.DataFrame(rows).sort_values(
        ["rmse", "mae", "r2"],
        ascending=[True, True, False],
    )

    save_csv(results, project_path(config, output_path))
    typer.echo(f"Saved boosting results: {project_path(config, output_path)}")


def run_train_advanced(
    config_path: Path,
    panel_path: Path,
    output_path: Path,
) -> None:
    """Run advanced sklearn model workflow."""
    config = load_config(config_path)
    settings = get_common_settings(config)

    panel, _ = load_or_build_panel(config, panel_path)
    model_df, feature_sets, _ = build_feature_sets_abc(panel, config)

    registry = get_default_model_registry(
        random_state=settings["random_state"],
    )

    model_specs = {
        "HistGBR": registry["hgb"],
        "RandomForest": registry["rf"],
        "ExtraTrees": registry["extra_trees"],
        "MLP": registry["mlp"],
    }

    rows = []

    for feature_set_name in ["A", "B", "C"]:
        for model_name, spec in model_specs.items():
            row, _, _, _ = run_time_experiment(
                df=model_df,
                feature_list=feature_sets[feature_set_name],
                target_col=settings["target_col"],
                year_col=settings["year_col"],
                model_name=model_name,
                model=spec["model"],
                scale_numeric=get_scale_mode(spec),
                test_years=settings["test_years"],
                split_label=f"time_{feature_set_name}",
            )
            row["feature_set"] = feature_set_name
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["rmse", "mae", "r2"],
        ascending=[True, True, False],
    )

    save_csv(results, project_path(config, output_path))
    typer.echo(f"Saved advanced results: {project_path(config, output_path)}")


def run_train_wdi(
    config_path: Path,
    panel_path: Path,
    output_path: Path,
    predictions_dir: Path,
) -> None:
    """Run WDI lag model workflow."""
    config = load_config(config_path)
    settings = get_common_settings(config)

    panel, _ = load_or_build_panel(config, panel_path)

    wdi_raw = load_source(config, "wdi")
    wdi_config = config["data"]["raw_sources"]["wdi"]
    wdi_standardized = standardize(wdi_raw, wdi_config)
    wdi_panel, _ = pivot_wdi(wdi_standardized, config)

    wdi_target = infer_wdi_target(config, wdi_panel)

    wdi_with_meta, _ = attach_panel_metadata_to_wdi(
        wdi_panel,
        panel,
        year_col=settings["year_col"],
        join_key="country",
        metadata_cols=["region", "income_group", "status"],
        restrict_to_panel_countries=True,
    )

    metadata_cols = [
        col
        for col in ["region", "income_group", "status"]
        if col in wdi_with_meta.columns
    ]

    id_cols = ["country", "country_code", settings["year_col"]]
    excluded = set(id_cols + metadata_cols + [wdi_target])

    wdi_indicator_features = [
        col
        for col in wdi_with_meta.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(wdi_with_meta[col])
    ]

    wdi_lagged = make_country_lag_features(
        wdi_with_meta,
        target_col=wdi_target,
        country_col="country",
        year_col=settings["year_col"],
        feature_cols=wdi_indicator_features,
        lags=(1, 2, 3),
        rolling_windows=(3,),
        dropna_lagged=True,
    )

    lag_feature_cols = [
        col
        for col in wdi_lagged.columns
        if (
            "_lag" in col
            or "_rollmean_" in col
            or col in [settings["year_col"], *metadata_cols]
        )
    ]

    wdi_train, wdi_val, wdi_test, _ = make_panel_overlap_split(
        wdi_lagged,
        panel,
        year_col=settings["year_col"],
        val_years=settings["val_years"],
        test_years=settings["test_years"],
    )

    x_train = wdi_train[lag_feature_cols].copy()
    y_train = wdi_train[wdi_target].copy()
    x_val = wdi_val[lag_feature_cols].copy()
    y_val = wdi_val[wdi_target].copy()
    x_test = wdi_test[lag_feature_cols].copy()
    y_test = wdi_test[wdi_target].copy()

    registry = get_default_model_registry(
        random_state=settings["random_state"],
    )

    wdi_models = {
        "WDI_RidgeLag": {
            "model": registry["ridge"]["model"],
            "scale_numeric": True,
        },
        "WDI_HistGBR_Lag": {
            "model": registry["hgb"]["model"],
            "scale_numeric": False,
        },
        "WDI_RandomForest_Lag": {
            "model": registry["rf"]["model"],
            "scale_numeric": False,
        },
        "WDI_ExtraTrees_Lag": {
            "model": registry["extra_trees"]["model"],
            "scale_numeric": False,
        },
    }

    rows = []
    prediction_output_dir = project_path(config, predictions_dir)
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, spec in wdi_models.items():
        row, pred_df, _ = fit_evaluate_wdi_model(
            model_name=model_name,
            model=spec["model"],
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            X_test=x_test,
            y_test=y_test,
            test_df=wdi_test,
            year_col=settings["year_col"],
            metadata_cols=metadata_cols,
            scale_numeric=spec["scale_numeric"],
        )

        rows.append(row)
        save_csv(pred_df, prediction_output_dir / f"{model_name}.csv")

    results = pd.DataFrame(rows).sort_values(
        ["rmse", "mae", "r2"],
        ascending=[True, True, False],
    )

    save_csv(results, project_path(config, output_path))
    typer.echo(f"Saved WDI results: {project_path(config, output_path)}")


def prepare_feature_split(
    config: Config,
    panel_path: str | Path,
    feature_set: str,
) -> dict[str, Any]:
    """Build the time-aware train/test split for one feature set.

    Reuses the same panel, feature sets, and split logic as the training
    commands so tuning and interpretability operate on identical inputs.

    Args:
        config: Loaded project configuration.
        panel_path: Path to the processed panel.
        feature_set: Feature-set name (`A`, `B`, or `C`).

    Returns:
        Dictionary with split frames, feature list, and common settings.

    Raises:
        KeyError: If the feature set is unknown.
    """
    settings = get_common_settings(config)
    panel, _ = load_or_build_panel(config, panel_path)
    model_df, feature_sets, _ = build_feature_sets_abc(panel, config)

    if feature_set not in feature_sets:
        available = sorted(feature_sets)
        raise KeyError(f"Unknown feature set {feature_set!r}. Available: {available}")

    features = feature_sets[feature_set]
    target_col = settings["target_col"]
    year_col = settings["year_col"]

    keep_cols = list(dict.fromkeys([*features, target_col, year_col]))
    work_df = model_df[[col for col in keep_cols if col in model_df.columns]].copy()

    x_train, x_test, y_train, y_test, split_info = make_time_split(
        work_df,
        target_col=target_col,
        year_col=year_col,
        test_years=settings["test_years"],
    )

    model_features = [col for col in features if col in x_train.columns]

    return {
        "x_train": x_train[model_features],
        "x_test": x_test[model_features],
        "y_train": y_train,
        "y_test": y_test,
        "split_info": split_info,
        "settings": settings,
        "features": model_features,
    }


def run_tune(
    config_path: Path,
    panel_path: Path,
    model_name: str,
    feature_set: str,
    results_path: Path,
    best_params_path: Path,
    compare_path: Path,
) -> None:
    """Run Bayesian hyperparameter search and compare against the default model."""
    if model_name not in AVAILABLE_MODELS:
        raise typer.BadParameter(
            f"Unknown model {model_name!r}. Available: {sorted(AVAILABLE_MODELS)}"
        )

    config = load_config(config_path)
    split = prepare_feature_split(config, panel_path, feature_set)
    settings = split["settings"]

    scale_numeric = get_tunable_scale_numeric(model_name)

    # Untuned baseline: same estimator with library default hyperparameters.
    default_pipeline = build_tunable_pipeline(
        model_name,
        split["x_train"],
        random_state=settings["random_state"],
        scale_numeric=scale_numeric,
    )
    default_pipeline.fit(split["x_train"], split["y_train"])
    default_metrics = regression_metrics(
        split["y_test"],
        default_pipeline.predict(split["x_test"]),
    )

    # Bayesian search using the project `tuning` config.
    searcher = run_bayes_search_from_config(
        model_name,
        split["x_train"],
        split["y_train"],
        config,
    )
    tuned_metrics = regression_metrics(
        split["y_test"],
        searcher.best_estimator_.predict(split["x_test"]),
    )

    results = bayes_search_results(searcher)
    result_cols = [
        col
        for col in ["rank_test_score", "mean_test_score", "std_test_score", "params"]
        if col in results.columns
    ]
    save_csv(results[result_cols], project_path(config, results_path))

    best_params = {
        key: _to_native(value) for key, value in searcher.best_params_.items()
    }
    summary = {
        "model_name": model_name,
        "feature_set": feature_set,
        "n_iter": config.get("tuning", {}).get("n_iter", 32),
        "scoring": config.get("tuning", {}).get(
            "scoring", "neg_root_mean_squared_error"
        ),
        "best_params": best_params,
        "best_cv_score": float(searcher.best_score_),
        "default_test_rmse": default_metrics["rmse"],
        "tuned_test_rmse": tuned_metrics["rmse"],
        "default_test_r2": default_metrics["r2"],
        "tuned_test_r2": tuned_metrics["r2"],
        "rmse_improvement": default_metrics["rmse"] - tuned_metrics["rmse"],
        "test_years": split["split_info"].extra.get("test_year_values"),
    }
    save_json(summary, project_path(config, best_params_path))

    compare_row = {
        "model_name": model_name,
        "feature_set": feature_set,
        "default_test_rmse": default_metrics["rmse"],
        "tuned_test_rmse": tuned_metrics["rmse"],
        "rmse_improvement": summary["rmse_improvement"],
        "default_test_r2": default_metrics["r2"],
        "tuned_test_r2": tuned_metrics["r2"],
        "best_cv_score": summary["best_cv_score"],
    }
    append_compare_row(compare_row, project_path(config, compare_path))

    typer.echo(
        f"Tuned {model_name} (set {feature_set}): "
        f"RMSE {default_metrics['rmse']:.4f} -> {tuned_metrics['rmse']:.4f} "
        f"(improvement {summary['rmse_improvement']:+.4f})"
    )
    typer.echo(f"Saved search results: {project_path(config, results_path)}")
    typer.echo(f"Saved best params: {project_path(config, best_params_path)}")
    typer.echo(f"Saved comparison: {project_path(config, compare_path)}")


def run_interpret(
    config_path: Path,
    panel_path: Path,
    model_name: str,
    feature_set: str,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    """Train a model and save SHAP + permutation importance tables and plots."""
    if model_name not in AVAILABLE_MODELS:
        raise typer.BadParameter(
            f"Unknown model {model_name!r}. Available: {sorted(AVAILABLE_MODELS)}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config(config_path)
    interp_config = config.get("interpretability", {})
    split = prepare_feature_split(config, panel_path, feature_set)
    settings = split["settings"]

    pipeline = build_tunable_pipeline(
        model_name,
        split["x_train"],
        random_state=settings["random_state"],
        scale_numeric=get_tunable_scale_numeric(model_name),
    )
    pipeline.fit(split["x_train"], split["y_train"])

    tables_out = project_path(config, tables_dir)
    figures_out = project_path(config, figures_dir)
    tables_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    n_features = interp_config.get("n_features", 20)

    permutation = permutation_importance_df(
        pipeline,
        split["x_test"],
        split["y_test"],
        n_repeats=interp_config.get("n_repeats", 10),
        scoring=interp_config.get("scoring", "neg_root_mean_squared_error"),
        random_state=interp_config.get("random_state", 42),
    )
    save_csv(permutation, tables_out / f"interpretability_{model_name}_permutation.csv")

    ax = plot_permutation_bar(permutation, n_features=n_features)
    ax.figure.tight_layout()
    ax.figure.savefig(
        figures_out / f"interpretability_{model_name}_permutation.png", dpi=150
    )
    plt.close(ax.figure)
    typer.echo(f"Saved permutation importance for {model_name}.")

    try:
        shap_df = shap_importance(
            pipeline,
            split["x_test"],
            max_background_samples=interp_config.get("max_background_samples", 200),
        )
    except Exception as exc:  # noqa: BLE001 - SHAP is optional/best-effort
        typer.echo(f"Skipped SHAP importance: {exc}")
        return

    save_csv(shap_df, tables_out / f"interpretability_{model_name}_shap.csv")
    ax = plot_shap_bar(shap_df, n_features=n_features)
    ax.figure.tight_layout()
    ax.figure.savefig(figures_out / f"interpretability_{model_name}_shap.png", dpi=150)
    plt.close(ax.figure)
    typer.echo(f"Saved SHAP importance for {model_name}.")


def append_compare_row(row: dict[str, Any], out_path: str | Path) -> Path:
    """Append one comparison row to a CSV log, creating it if needed."""
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row])
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        row_df = pd.concat([old_df, row_df], ignore_index=True)

    row_df.to_csv(output_path, index=False)
    return output_path


def _to_native(value: Any) -> Any:
    """Convert numpy scalar types to native Python for JSON serialization."""
    if hasattr(value, "item"):
        return value.item()
    return value


def infer_wdi_target(config: Config, wdi_panel: pd.DataFrame) -> str:
    """Infer WDI target column after pivoting."""
    wdi_config = config.get("wdi", {})
    prefix = wdi_config.get("wide_prefix", "")
    target_code = wdi_config.get("target_code", "SP.DYN.LE00.IN")

    candidates = [
        f"{prefix}{target_code}",
        target_code,
        "wdi_life_expectancy",
    ]

    for candidate in candidates:
        if candidate in wdi_panel.columns:
            return candidate

    raise KeyError(
        "Could not infer WDI target column. Tried: "
        f"{candidates}. Available columns include: {list(wdi_panel.columns)[:20]}"
    )


@app.command()
def info(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        help="Path to YAML config.",
    ),
) -> None:
    """Show basic project/config information."""
    config = load_config(config_path)
    settings = get_common_settings(config)

    typer.echo(f"Project root: {config['project']['root']}")
    typer.echo(f"Target column: {settings['target_col']}")
    typer.echo(f"Year column: {settings['year_col']}")
    typer.echo(f"Test years: {settings['test_years']}")
    typer.echo(f"Validation years: {settings['val_years']}")


@app.command()
def preprocess(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    output_path: Path = typer.Option(Path("data/processed/panel.csv")),
    summary_path: Path = typer.Option(
        Path("reports/tables/preprocessing_summary.json")
    ),
) -> None:
    """Build and save the processed analytical panel."""
    run_preprocess(config_path, output_path, summary_path)


@app.command()
def features(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    panel_path: Path = typer.Option(Path("data/processed/panel.csv")),
    model_frame_path: Path = typer.Option(Path("data/processed/model_frame.csv")),
    feature_sets_path: Path = typer.Option(Path("reports/tables/feature_sets.json")),
) -> None:
    """Build model frame and feature sets A/B/C."""
    run_features(
        config_path,
        panel_path,
        model_frame_path,
        feature_sets_path,
    )


@app.command("train-baselines")
def train_baselines(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    panel_path: Path = typer.Option(Path("data/processed/panel.csv")),
    output_path: Path = typer.Option(Path("reports/tables/baseline_results.csv")),
) -> None:
    """Train baseline models on time-aware split."""
    run_train_baselines(config_path, panel_path, output_path)


@app.command("train-boosting")
def train_boosting(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    panel_path: Path = typer.Option(Path("data/processed/panel.csv")),
    output_path: Path = typer.Option(
        Path("reports/tables/external_boosting_compare.csv")
    ),
    run_log_path: Path = typer.Option(
        Path("reports/tables/external_boosting_runs.csv")
    ),
) -> None:
    """Train configured XGBoost/LightGBM/CatBoost experiments."""
    run_train_boosting(
        config_path,
        panel_path,
        output_path,
        run_log_path,
    )


@app.command("train-advanced")
def train_advanced(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    panel_path: Path = typer.Option(Path("data/processed/panel.csv")),
    output_path: Path = typer.Option(
        Path("reports/tables/advanced_model_comparison.csv")
    ),
) -> None:
    """Train default advanced sklearn models over feature sets A/B/C."""
    run_train_advanced(config_path, panel_path, output_path)


@app.command("train-wdi")
def train_wdi(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    panel_path: Path = typer.Option(Path("data/processed/panel.csv")),
    output_path: Path = typer.Option(
        Path("reports/tables/wdi_panel_overlap_model_compare.csv")
    ),
    predictions_dir: Path = typer.Option(Path("reports/tables/wdi_predictions")),
) -> None:
    """Train WDI lag models on panel-overlap time split."""
    run_train_wdi(
        config_path,
        panel_path,
        output_path,
        predictions_dir,
    )


@app.command()
def tune(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    model_name: str = typer.Option("hgb", help="Model to tune (e.g. hgb, ridge, rf)."),
    feature_set: str = typer.Option("B", help="Feature set A, B, or C."),
    panel_path: Path = typer.Option(Path("data/processed/panel.csv")),
    results_path: Path = typer.Option(
        Path("reports/tables/bayes_tuning_results.csv"),
        help="CSV of all evaluated Bayesian search configurations.",
    ),
    best_params_path: Path = typer.Option(
        Path("reports/tables/bayes_best_params.json"),
        help="JSON of best params and tuned-vs-default metrics.",
    ),
    compare_path: Path = typer.Option(
        Path("reports/tables/bayes_tuning_compare.csv"),
        help="Appended default-vs-tuned comparison log.",
    ),
) -> None:
    """Run Bayesian hyperparameter search (scikit-optimize) for one model."""
    run_tune(
        config_path,
        panel_path,
        model_name,
        feature_set,
        results_path,
        best_params_path,
        compare_path,
    )


@app.command()
def interpret(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    model_name: str = typer.Option("hgb", help="Model to interpret (e.g. hgb, rf)."),
    feature_set: str = typer.Option("B", help="Feature set A, B, or C."),
    panel_path: Path = typer.Option(Path("data/processed/panel.csv")),
    tables_dir: Path = typer.Option(Path("reports/tables")),
    figures_dir: Path = typer.Option(Path("reports/figures")),
) -> None:
    """Compute SHAP and permutation feature importance for one model."""
    run_interpret(
        config_path,
        panel_path,
        model_name,
        feature_set,
        tables_dir,
        figures_dir,
    )


@app.command()
def all(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
) -> None:
    """Run core reproducible workflow."""
    run_preprocess(
        config_path,
        Path("data/processed/panel.csv"),
        Path("reports/tables/preprocessing_summary.json"),
    )
    run_features(
        config_path,
        Path("data/processed/panel.csv"),
        Path("data/processed/model_frame.csv"),
        Path("reports/tables/feature_sets.json"),
    )
    run_train_baselines(
        config_path,
        Path("data/processed/panel.csv"),
        Path("reports/tables/baseline_results.csv"),
    )
    run_train_advanced(
        config_path,
        Path("data/processed/panel.csv"),
        Path("reports/tables/advanced_model_comparison.csv"),
    )
    run_train_wdi(
        config_path,
        Path("data/processed/panel.csv"),
        Path("reports/tables/wdi_panel_overlap_model_compare.csv"),
        Path("reports/tables/wdi_predictions"),
    )


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
