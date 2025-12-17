"""Main entry point for the Fair Credit Score Prediction pipeline.

This module provides CLI commands for running the complete pipeline,
from data loading through model training, fairness analysis, and
explainability generation.

Usage:
    python -m src.main run --data-path data/credit.csv
    python -m src.main run --data-path data/credit.csv --config config/custom.yaml
    python -m src.main explain --model-path models/model.pkl --data-path data/credit.csv
"""

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def _parse_csv_option(value: str | None) -> list[str] | None:
    """Convert a comma-separated string into a list of tokens."""
    if not value:
        return None

    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def _parse_metrics_option(metrics_value: str | None) -> list[str] | None:
    """Convert comma-separated CLI option into a list of metrics."""
    result = _parse_csv_option(metrics_value)
    if result is None:
        return None
    return [metric.lower() for metric in result]


def _parse_models_option(models_value: str | None) -> list[str] | None:
    """Convert comma-separated CLI option into a list of model identifiers."""
    result = _parse_csv_option(models_value)
    if result is None:
        return None
    return [model.lower() for model in result]


def _parse_indices_option(indices_value: str | None, max_length: int) -> list[int]:
    """Convert comma-separated indices string into a list of ints."""
    if not indices_value:
        return list(range(min(5, max_length)))

    indices: list[int] = []
    for chunk in indices_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            indices.append(int(chunk))
        except ValueError:
            raise click.BadParameter(f"Invalid index value '{chunk}'") from None

    if not indices:
        return list(range(min(5, max_length)))

    return indices


def _write_json(data: Any, path: str | Path) -> Path:
    """Write a JSON file with pretty formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return path


def _save_calibration_plot(
    calibration_data: dict[str, Any],
    path: str | Path,
) -> Path:
    """Persist calibration curve plot to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    prob_pred = calibration_data.get("prob_pred", [])
    prob_true = calibration_data.get("prob_true", [])

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, load_config
from src.utils import (
    setup_logging,
    set_random_seed,
    save_metrics_to_json,
    get_logger,
)
from src.data_loader import DataLoader, load_data
from src.preprocessing import Preprocessor, preprocess_data
from src.model import CreditModel
from src.fairness import FairnessAnalyzer, FairnessMetrics
from src.explainability import Explainer
from src.protected_attribute import prepare_features_without_protected


DEFAULT_BENCHMARK_MODELS = ["xgboost", "rf", "lr"]

BENCHMARK_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "xgboost": {
        "label": "XGBoost",
        "aliases": {"xgb", "xg"},
        "factory": lambda cfg: XGBClassifier(**cfg.model.xgb_params),
    },
    "rf": {
        "label": "Random Forest",
        "aliases": {"random_forest", "randomforest", "forest"},
        "factory": lambda cfg: RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.model.random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    },
    "lr": {
        "label": "Logistic Regression",
        "aliases": {
            "logistic",
            "logistic_regression",
            "logreg",
            "logisticregression",
        },
        "factory": lambda cfg: LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        ),
    },
}

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "accuracy": {
        "requires_proba": False,
        "func": lambda y_true, y_pred, y_prob: accuracy_score(y_true, y_pred),
    },
    "precision": {
        "requires_proba": False,
        "func": lambda y_true, y_pred, y_prob: precision_score(
            y_true, y_pred, zero_division=0
        ),
    },
    "recall": {
        "requires_proba": False,
        "func": lambda y_true, y_pred, y_prob: recall_score(
            y_true, y_pred, zero_division=0
        ),
    },
    "f1": {
        "requires_proba": False,
        "func": lambda y_true, y_pred, y_prob: f1_score(
            y_true, y_pred, zero_division=0
        ),
    },
    "roc_auc": {
        "requires_proba": True,
        "func": lambda y_true, y_pred, y_prob: roc_auc_score(y_true, y_prob),
    },
}


def _resolve_benchmark_model(name: str) -> str:
    """Resolve a benchmark model alias to its canonical key."""
    normalized = name.strip().lower()
    for key, spec in BENCHMARK_MODEL_REGISTRY.items():
        if normalized == key or normalized in spec["aliases"]:
            return key
    raise click.BadParameter(
        f"Unknown model '{name}'. Available: {list(BENCHMARK_MODEL_REGISTRY)}"
    )


def _build_benchmark_estimator(model_key: str, config: Config):
    """Instantiate an estimator for benchmarking."""
    spec = BENCHMARK_MODEL_REGISTRY[model_key]
    estimator = spec["factory"](config)
    return spec["label"], estimator


def _fit_estimator(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> bool:
    """Fit estimator, returning whether sample weights were used."""
    if sample_weights is None:
        estimator.fit(X_train, y_train)
        return False

    try:
        estimator.fit(X_train, y_train, sample_weight=sample_weights)
        return True
    except TypeError:
        estimator.fit(X_train, y_train)
        return False


def _predict_proba_safe(estimator: Any, X: np.ndarray) -> np.ndarray | None:
    """Return probability predictions if supported."""
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        if probs.ndim == 1:
            return probs

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score > 0:
                return (scores - min_score) / (max_score - min_score)
    return None


def _evaluate_selected_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    metric_names: Sequence[str],
) -> dict[str, float]:
    """Compute requested metrics, skipping unsupported ones."""
    results: dict[str, float] = {}
    for metric in metric_names:
        metric_key = metric.lower()
        spec = METRIC_SPECS.get(metric_key)
        if spec is None:
            continue
        if spec["requires_proba"] and y_prob is None:
            continue
        try:
            value = spec["func"](y_true, y_pred, y_prob)
        except Exception:
            continue
        if value is not None:
            results[metric_key] = float(value)
    return results


def run_pipeline(
    data_path: str | Path,
    config_path: str | Path | None = None,
    save_model: bool = True,
    validate_data: bool = True,
    enable_cross_validation: bool = False,
    cross_validation_folds: int = 5,
    cross_validation_metrics: Sequence[str] | None = None,
    enable_tuning: bool = False,
    tuning_search_type: str = "grid",
    tuning_param_grid: dict[str, list[Any]] | None = None,
    tuning_cv_folds: int = 5,
    tuning_n_iter: int = 20,
    enable_calibration: bool = False,
    calibration_method: str = "isotonic",
    save_preprocessor: bool = False,
) -> dict[str, Any]:
    """Run the complete credit prediction pipeline.

    Args:
        data_path: Path to the input CSV data file.
        config_path: Path to configuration YAML file.
        save_model: Whether to save the trained model.
        validate_data: Run schema validation before preprocessing.
        enable_cross_validation: Run cross-validation prior to training.
        cross_validation_folds: Number of folds for CV.
        cross_validation_metrics: Metrics to compute during CV.
        enable_tuning: Run hyperparameter search before training.
        tuning_search_type: Grid or random search strategy.
        tuning_param_grid: Custom parameter grid (optional).
        tuning_cv_folds: CV folds to use during tuning.
        tuning_n_iter: Iterations for random search.
        enable_calibration: Calibrate probabilities after training.
        calibration_method: Calibration method ('isotonic' or 'sigmoid').
        save_preprocessor: Persist the fitted preprocessor artifact.

    Returns:
        Dictionary containing pipeline results and metrics.
    """
    # Load configuration
    config = load_config(config_path)
    config.paths.ensure_dirs()

    # Setup logging
    logger = setup_logging(config.logging)
    logger.info("="*60)
    logger.info("Fair Credit Score Prediction Pipeline")
    logger.info("="*60)

    # Set random seeds for reproducibility
    set_random_seed(config.model.random_state)

    results: dict[str, Any] = {
        "config_path": str(config_path) if config_path else "default",
        "data_path": str(data_path),
    }
    proxy_col = config.cdi.proxy_column

    # =========================================================================
    # STEP 1: Data Loading & Validation
    # =========================================================================
    logger.info("\n[STEP 1] Loading data...")

    data_loader = DataLoader(config)
    raw_data = data_loader.load_csv(data_path)
    validation_report: dict[str, Any] = {}

    if validate_data:
        logger.info("Running data validation...")
        validation_report = data_loader.validate(raw_data)
        results["validation_report"] = validation_report

        if validation_report.get("issues"):
            logger.warning(
                f"Data validation found {len(validation_report['issues'])} issues"
            )
    else:
        logger.info("Data validation skipped (disabled via CLI flag)")
        results["validation_report"] = {}

    # =========================================================================
    # STEP 2: Preprocessing & Feature Engineering
    # =========================================================================
    logger.info("\n[STEP 2] Preprocessing data...")

    preprocessor = Preprocessor(config)
    X_train, X_test, y_train, y_test, processed_df = preprocessor.fit_transform(raw_data)

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    results["data_shape"] = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": X_train.shape[1],
    }

    # =========================================================================
    # STEP 3: Fairness Pre-processing (Reweighing)
    # =========================================================================
    logger.info("\n[STEP 3] Applying fairness pre-processing...")

    fairness_analyzer = FairnessAnalyzer(config)

    sample_weights = None
    if config.fairness.apply_reweighting:
        sample_weights = fairness_analyzer.apply_reweighing(X_train, y_train)
        logger.info("Reweighing applied to training data")
    else:
        logger.info("Reweighing disabled in config")

    # =========================================================================
    # STEP 4: Model Training (with optional CV/tuning)
    # =========================================================================
    logger.info("\n[STEP 4] Training model...")

    model = CreditModel(config)
    if enable_cross_validation:
        cv_metrics = (
            list(cross_validation_metrics)
            if cross_validation_metrics
            else ["roc_auc"]
        )
        logger.info(
            f"Running {cross_validation_folds}-fold cross-validation "
            f"for metrics: {', '.join(cv_metrics)}"
        )
        cv_results = model.cross_validate(
            X_train,
            y_train,
            cv=cross_validation_folds,
            scoring=cv_metrics,
            protected_attribute=proxy_col,
        )
        results["cross_validation"] = cv_results

    if enable_tuning:
        logger.info(
            f"Running hyperparameter tuning via {tuning_search_type} search "
            f"with {tuning_cv_folds}-fold CV"
        )
        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            param_grid=tuning_param_grid,
            search_type=tuning_search_type,
            cv=tuning_cv_folds,
            n_iter=tuning_n_iter,
            protected_attribute=proxy_col,
            sample_weights=sample_weights,
        )
        results["tuning"] = tuning_results

        best_params = tuning_results.get("best_params")
        if best_params:
            config.model.xgb_params.update(best_params)
            logger.info("Model configuration updated with tuned parameters")

    model.train(X_train, y_train, sample_weights=sample_weights)

    if enable_calibration:
        logger.info(
            f"Calibrating model using {calibration_method} method on holdout set"
        )
        model.calibrate_model(
            X_test,
            y_test,
            method=calibration_method,
            protected_attribute=proxy_col,
        )

    # =========================================================================
    # STEP 5: Prediction & Evaluation
    # =========================================================================
    logger.info("\n[STEP 5] Generating predictions and evaluating...")

    # Standard predictions
    y_pred = model.predict(X_test, protected_attribute=proxy_col)
    y_prob = model.predict_proba(X_test, protected_attribute=proxy_col)

    # Model performance metrics
    model_metrics = model.evaluate(
        X_test,
        y_test,
        protected_attribute=proxy_col,
    )
    results["model_metrics"] = model_metrics

    if enable_calibration:
        calibration_curve = model.get_calibration_curve(
            X_test,
            y_test,
            protected_attribute=proxy_col,
        )
        results["calibration_curve"] = calibration_curve

    # =========================================================================
    # STEP 6: Fairness Post-processing (Threshold Adjustment)
    # =========================================================================
    logger.info("\n[STEP 6] Applying fairness post-processing...")

    protected_values = X_test[proxy_col].values

    y_pred_adjusted = fairness_analyzer.apply_threshold_adjustment(
        y_prob, protected_values
    )

    # =========================================================================
    # STEP 7: Fairness Analysis
    # =========================================================================
    logger.info("\n[STEP 7] Calculating fairness metrics...")

    # Before adjustment
    fairness_before = fairness_analyzer.calculate_metrics(
        X_test, y_test, y_pred
    )
    fairness_analyzer.log_metrics(fairness_before, "Fairness Metrics (Before Adjustment)")

    # After adjustment
    fairness_after = fairness_analyzer.calculate_metrics(
        X_test, y_test, y_pred_adjusted
    )
    fairness_analyzer.log_metrics(fairness_after, "Fairness Metrics (After Adjustment)")

    results["fairness_metrics"] = {
        "before_adjustment": fairness_before.to_dict(),
        "after_adjustment": fairness_after.to_dict(),
    }

    # Check if fairness targets are met
    fairness_results = fairness_after.passes_thresholds(
        di_threshold=config.fairness.target_disparate_impact_ratio,
        spd_threshold=config.fairness.target_statistical_parity_diff,
        eod_threshold=config.fairness.target_equalized_odds_diff,
    )
    results["fairness_targets_met"] = fairness_results

    all_targets_met = all(fairness_results.values())
    if all_targets_met:
        logger.info("✓ All fairness targets met!")
    else:
        failed = [k for k, v in fairness_results.items() if not v]
        logger.warning(f"✗ Some fairness targets not met: {failed}")

    # =========================================================================
    # STEP 8: Explainability
    # =========================================================================
    logger.info("\n[STEP 8] Generating explanations...")

    explainer = Explainer(config)
    explainer.setup_shap_explainer(model)

    # Generate global feature importance plot
    importance_plot_path = explainer.generate_global_importance_plot(X_test)
    detailed_plot_path = explainer.generate_detailed_shap_plot(X_test)

    # Generate individual explanations
    explanations = explainer.explain_samples(X_test, y_pred_adjusted)

    results["explanations"] = {
        "num_generated": len(explanations),
        "global_importance_plot": str(importance_plot_path),
        "detailed_plot": str(detailed_plot_path),
    }

    # =========================================================================
    # STEP 9: Save Artifacts
    # =========================================================================
    logger.info("\n[STEP 9] Saving artifacts...")

    # Save model
    if save_model:
        model_path = config.paths.models_dir / "credit_model.pkl"
        model.save(model_path)
        results["model_path"] = str(model_path)

    if save_model or save_preprocessor:
        preprocessor_path = config.paths.models_dir / "preprocessor.pkl"
        preprocessor.save(preprocessor_path)
        results["preprocessor_path"] = str(preprocessor_path)

    # Save metrics
    metrics_path = config.paths.output_dir / "metrics.json"
    save_metrics_to_json(results, metrics_path, logger)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Model Accuracy: {model_metrics['accuracy']:.4f}")
    logger.info(f"Disparate Impact: {fairness_after.disparate_impact:.4f}")
    logger.info(f"Statistical Parity Diff: {fairness_after.statistical_parity_difference:.4f}")
    logger.info(f"Equalized Odds Diff: {fairness_after.equalized_odds_difference:.4f}")
    logger.info(f"Explanations generated: {len(explanations)}")
    logger.info("="*60 + "\n")

    return results


@click.group()
def cli() -> None:
    """Fair Credit Score Prediction CLI."""
    pass


@cli.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to the input CSV data file.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--no-save-model",
    is_flag=True,
    default=False,
    help="Don't save the trained model.",
)
@click.option(
    "--tune/--no-tune",
    default=False,
    help="Enable hyperparameter tuning prior to training.",
)
@click.option(
    "--search-type",
    type=click.Choice(["grid", "random"]),
    default="grid",
    show_default=True,
    help="Search strategy used when --tune is enabled.",
)
@click.option(
    "--tuning-cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="CV folds to use during hyperparameter tuning.",
)
@click.option(
    "--tuning-n-iter",
    type=int,
    default=20,
    show_default=True,
    help="Iterations for random search (ignored for grid search).",
)
@click.option(
    "--cv/--no-cv",
    default=False,
    help="Run cross-validation before model training.",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of folds when --cv is enabled.",
)
@click.option(
    "--cv-metrics",
    default="roc_auc",
    show_default=True,
    help="Comma-separated metrics for cross-validation (e.g. 'roc_auc,f1').",
)
@click.option(
    "--calibrate/--no-calibrate",
    default=False,
    help="Calibrate probabilities using the holdout set.",
)
@click.option(
    "--calibration-method",
    type=click.Choice(["isotonic", "sigmoid"]),
    default="isotonic",
    show_default=True,
    help="Calibration method to use when --calibrate is enabled.",
)
@click.option(
    "--validate-data/--skip-validate-data",
    default=True,
    help="Run schema validation before preprocessing.",
)
@click.option(
    "--save-preprocessor/--no-save-preprocessor",
    default=False,
    help="Persist the fitted preprocessor artifact.",
)
def run(
    data_path: str,
    config: str | None,
    no_save_model: bool,
    tune: bool,
    search_type: str,
    tuning_cv_folds: int,
    tuning_n_iter: int,
    cv: bool,
    cv_folds: int,
    cv_metrics: str,
    calibrate: bool,
    calibration_method: str,
    validate_data: bool,
    save_preprocessor: bool,
) -> None:
    """Run the complete credit prediction pipeline."""
    metrics_list = _parse_metrics_option(cv_metrics)
    try:
        results = run_pipeline(
            data_path=data_path,
            config_path=config,
            save_model=not no_save_model,
            validate_data=validate_data,
            enable_cross_validation=cv,
            cross_validation_folds=cv_folds,
            cross_validation_metrics=metrics_list,
            enable_tuning=tune,
            tuning_search_type=search_type,
            tuning_param_grid=None,
            tuning_cv_folds=tuning_cv_folds,
            tuning_n_iter=tuning_n_iter,
            enable_calibration=calibrate,
            calibration_method=calibration_method,
            save_preprocessor=save_preprocessor,
        )

        logger = get_logger("cli")
        logger.info("Pipeline completed successfully!")
        logger.info(f"Trained model saved: {results.get('model_path')}")
        logger.info(f"Evaluation metrics: {results.get('model_metrics')}")

        click.echo("\nPipeline completed successfully!")
        model_metrics = results.get("model_metrics") or {}
        if "accuracy" in model_metrics:
            click.echo(f"Model accuracy: {model_metrics['accuracy']:.4f}")

        fairness_targets = results.get("fairness_targets_met")
        if fairness_targets:
            all_met = all(fairness_targets.values())
            sys.exit(0 if all_met else 1)
        sys.exit(0)

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to the training CSV data file.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--search-type",
    type=click.Choice(["grid", "random"]),
    default="random",
    show_default=True,
    help="Search strategy to use.",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of folds for tuning cross-validation.",
)
@click.option(
    "--n-iter",
    type=int,
    default=20,
    show_default=True,
    help="Iterations for random search.",
)
@click.option(
    "--param-grid",
    type=click.Path(exists=True),
    help="Optional JSON file with search space definition.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Where to write the tuning report JSON.",
)
def tune(
    data_path: str,
    config: str | None,
    search_type: str,
    cv_folds: int,
    n_iter: int,
    param_grid: str | None,
    output: str | None,
) -> None:
    """Run standalone hyperparameter tuning."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()

    logger = setup_logging(cfg.logging)
    logger.info("Starting hyperparameter tuning job...")

    df, _ = load_data(data_path, cfg, validate=True)
    preprocessor = Preprocessor(cfg)
    X_train, _, y_train, _, _ = preprocessor.fit_transform(df)

    sample_weights = None
    if cfg.fairness.apply_reweighting:
        fairness_analyzer = FairnessAnalyzer(cfg)
        sample_weights = fairness_analyzer.apply_reweighing(X_train, y_train)

    model = CreditModel(cfg)

    param_grid_data: dict[str, list[Any]] | None = None
    if param_grid:
        with open(param_grid) as f:
            param_grid_data = json.load(f)

    results = model.tune_hyperparameters(
        X_train,
        y_train,
        param_grid=param_grid_data,
        search_type=search_type,
        cv=cv_folds,
        n_iter=n_iter,
        protected_attribute=cfg.cdi.proxy_column,
        sample_weights=sample_weights,
    )

    output_path = (
        Path(output)
        if output
        else cfg.paths.output_dir / "tuning_results.json"
    )
    saved_path = _write_json(results, output_path)

    click.echo(
        f"Best score: {results.get('best_score', float('nan')):.4f} "
        f"with params: {results.get('best_params')}"
    )
    click.echo(f"Tuning report saved to {saved_path}")


@cli.command(name="cv")
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to the training CSV data file.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of CV folds.",
)
@click.option(
    "--metrics",
    default="roc_auc",
    show_default=True,
    help="Comma-separated list of metrics, e.g. 'roc_auc,f1,accuracy'.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Where to write the CV report JSON.",
)
def cross_validate(
    data_path: str,
    config: str | None,
    cv_folds: int,
    metrics: str,
    output: str | None,
) -> None:
    """Run cross-validation for the configured model."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()
    logger = setup_logging(cfg.logging)
    logger.info("Starting cross-validation job...")

    df, _ = load_data(data_path, cfg, validate=True)
    preprocessor = Preprocessor(cfg)
    X_train, _, y_train, _, _ = preprocessor.fit_transform(df)

    model = CreditModel(cfg)
    metrics_list = _parse_metrics_option(metrics) or ["roc_auc"]

    results = model.cross_validate(
        X_train,
        y_train,
        cv=cv_folds,
        scoring=metrics_list,
        protected_attribute=cfg.cdi.proxy_column,
    )

    output_path = (
        Path(output) if output else cfg.paths.output_dir / "cv_results.json"
    )
    saved_path = _write_json(results, output_path)

    for metric_name, values in results.items():
        click.echo(
            f"{metric_name}: mean={values['mean']:.4f}, std={values['std']:.4f}"
        )
    click.echo(f"Cross-validation results saved to {saved_path}")


@cli.command()
@click.option(
    "--model-path",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to the trained model file.",
)
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to calibration dataset.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--method",
    type=click.Choice(["isotonic", "sigmoid"]),
    default="isotonic",
    show_default=True,
    help="Calibration method.",
)
@click.option(
    "--output-model-path",
    type=click.Path(),
    help="Optional destination for the calibrated model.",
)
@click.option(
    "--metrics-output",
    type=click.Path(),
    help="Where to save calibration metrics JSON.",
)
@click.option(
    "--plot-path",
    type=click.Path(),
    help="Where to save the calibration curve plot.",
)
def calibrate(
    model_path: str,
    data_path: str,
    config: str | None,
    method: str,
    output_model_path: str | None,
    metrics_output: str | None,
    plot_path: str | None,
) -> None:
    """Calibrate an existing model and save updated artifacts."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()
    logger = setup_logging(cfg.logging)
    logger.info("Calibrating saved model...")

    model = CreditModel(cfg)
    model.load(model_path)

    df, _ = load_data(data_path, cfg, validate=True)
    _, X_test, _, _, _ = preprocess_data(df, cfg)

    model.calibrate_model(
        X_test,
        y_test,
        method=method,
        protected_attribute=cfg.cdi.proxy_column,
    )

    calibration_data = model.get_calibration_curve(
        X_test,
        y_test,
        protected_attribute=cfg.cdi.proxy_column,
    )

    metrics_path = (
        Path(metrics_output)
        if metrics_output
        else cfg.paths.output_dir / "calibration_metrics.json"
    )
    saved_metrics_path = _write_json(calibration_data, metrics_path)

    plot_output = (
        Path(plot_path)
        if plot_path
        else cfg.paths.reports_dir / "figures" / "calibration_curve.png"
    )
    saved_plot_path = _save_calibration_plot(calibration_data, plot_output)

    target_model_path = output_model_path or model_path
    model.save(target_model_path)

    click.echo(
        f"Calibration complete. Brier score: {calibration_data.get('brier_score'):.4f}"
    )
    click.echo(f"Metrics saved to {saved_metrics_path}")
    click.echo(f"Calibration curve saved to {saved_plot_path}")
    click.echo(f"Calibrated model persisted to {target_model_path}")


@cli.command(name="validate")
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to the CSV file to validate.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Destination for the validation report JSON.",
)
def validate_data_cli(
    data_path: str,
    config: str | None,
    output: str | None,
) -> None:
    """Validate dataset schema and log issues."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()
    logger = setup_logging(cfg.logging)
    logger.info("Running data validation...")

    loader = DataLoader(cfg)
    df = loader.load_csv(data_path)
    report = loader.validate(df)

    output_path = (
        Path(output)
        if output
        else cfg.paths.output_dir / "validation_report.json"
    )
    saved_path = _write_json(report, output_path)

    num_issues = len(report.get("issues", []))
    click.echo(
        f"Validation completed with {num_issues} issue(s). Report saved to {saved_path}"
    )


@cli.command(name="visualize-fairness")
@click.option(
    "--model-path",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to the trained model file.",
)
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Dataset to use for fairness evaluation.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--plot-path",
    type=click.Path(),
    help="Override path for the fairness plot image.",
)
@click.option(
    "--report-path",
    type=click.Path(),
    help="Override path for the fairness text report.",
)
@click.option(
    "--metrics-output",
    type=click.Path(),
    help="Where to save fairness metrics JSON.",
)
def visualize_fairness(
    model_path: str,
    data_path: str,
    config: str | None,
    plot_path: str | None,
    report_path: str | None,
    metrics_output: str | None,
) -> None:
    """Generate fairness plots and reports for a saved model."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()
    logger = setup_logging(cfg.logging)
    logger.info("Generating fairness visualizations...")

    model = CreditModel(cfg)
    model.load(model_path)

    df, _ = load_data(data_path, cfg, validate=True)
    _, X_test, _, y_test, _ = preprocess_data(df, cfg)

    fairness_analyzer = FairnessAnalyzer(cfg)
    y_pred = model.predict(X_test, protected_attribute=cfg.cdi.proxy_column)
    y_prob = model.predict_proba(X_test, protected_attribute=cfg.cdi.proxy_column)

    protected_values = X_test[cfg.cdi.proxy_column].values
    y_adjusted = fairness_analyzer.apply_threshold_adjustment(y_prob, protected_values)

    fairness_before = fairness_analyzer.calculate_metrics(X_test, y_test, y_pred)
    fairness_after = fairness_analyzer.calculate_metrics(
        X_test, y_test, y_adjusted
    )

    fairness_analyzer.log_metrics(fairness_before, "Fairness Metrics (Before Adjustment)")
    fairness_analyzer.log_metrics(fairness_after, "Fairness Metrics (After Adjustment)")

    plot_target = (
        Path(plot_path)
        if plot_path
        else cfg.paths.reports_dir / "figures" / "fairness_metrics.png"
    )
    saved_plot = fairness_analyzer.plot_fairness_metrics(fairness_after, plot_target)

    report_target = (
        Path(report_path)
        if report_path
        else cfg.paths.reports_dir / "fairness_report.txt"
    )
    saved_report = fairness_analyzer.generate_fairness_report(
        fairness_after, report_target
    )

    metrics_payload = {
        "before_adjustment": fairness_before.to_dict(),
        "after_adjustment": fairness_after.to_dict(),
    }
    metrics_target = (
        Path(metrics_output)
        if metrics_output
        else cfg.paths.output_dir / "fairness_metrics.json"
    )
    saved_metrics = _write_json(metrics_payload, metrics_target)

    click.echo(f"Fairness plot saved to {saved_plot}")
    click.echo(f"Fairness report saved to {saved_report}")
    click.echo(f"Fairness metrics JSON saved to {saved_metrics}")


@cli.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset used for benchmarking.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--models",
    default="xgboost,rf,lr",
    show_default=True,
    help="Comma-separated model identifiers (xgboost, rf, lr).",
)
@click.option(
    "--metrics",
    default="roc_auc,f1,accuracy",
    show_default=True,
    help="Comma-separated metrics (roc_auc,f1,accuracy,precision,recall).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Destination path for benchmark results JSON.",
)
def benchmark(
    data_path: str,
    config: str | None,
    models: str,
    metrics: str,
    output: str | None,
) -> None:
    """Train and compare multiple model types on the same dataset."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()
    logger = setup_logging(cfg.logging)
    logger.info("Starting benchmarking run...")

    requested_models = _parse_models_option(models) or DEFAULT_BENCHMARK_MODELS
    deduped_models: list[str] = []
    for name in requested_models:
        key = _resolve_benchmark_model(name)
        if key not in deduped_models:
            deduped_models.append(key)

    metric_list = _parse_metrics_option(metrics) or ["roc_auc", "f1", "accuracy"]

    df, _ = load_data(data_path, cfg, validate=True)
    preprocessor = Preprocessor(cfg)
    X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(df)

    sample_weights = None
    if cfg.fairness.apply_reweighting:
        fairness_analyzer = FairnessAnalyzer(cfg)
        sample_weights = fairness_analyzer.apply_reweighing(X_train, y_train)

    train_ctx = prepare_features_without_protected(X_train, config=cfg)
    test_ctx = prepare_features_without_protected(X_test, config=cfg)
    X_train_filtered = train_ctx.filtered_df
    X_test_filtered = test_ctx.filtered_df

    benchmark_rows: list[dict[str, Any]] = []

    for model_key in deduped_models:
        label, estimator = _build_benchmark_estimator(model_key, cfg)
        logger.info(f"Training benchmark model: {label}")

        start = perf_counter()
        used_weights = _fit_estimator(
            estimator,
            X_train_filtered,
            y_train,
            sample_weights=sample_weights,
        )
        train_time = perf_counter() - start

        y_pred = estimator.predict(X_test_filtered)
        y_prob = _predict_proba_safe(estimator, X_test_filtered)

        metrics_result = _evaluate_selected_metrics(
            y_test.values,
            y_pred,
            y_prob,
            metric_list,
        )

        benchmark_rows.append(
            {
                "model_key": model_key,
                "model_name": label,
                "metrics": metrics_result,
                "training_time_seconds": float(train_time),
                "used_sample_weights": used_weights,
                "probabilities_available": y_prob is not None,
            }
        )

    if not benchmark_rows:
        raise click.ClickException("No valid models were provided for benchmarking.")

    ranking_metric = None
    for metric_name in metric_list:
        if any(metric_name in row["metrics"] for row in benchmark_rows):
            ranking_metric = metric_name
            break

    if ranking_metric is None:
        ranking_metric = next(
            iter(benchmark_rows[0]["metrics"]), "accuracy"
        )

    best_row = max(
        benchmark_rows,
        key=lambda row: row["metrics"].get(ranking_metric, float("-inf")),
    )

    header = ["Model"] + [m.upper() for m in metric_list] + ["Train(s)"]
    row_format = "{:<22}" + "".join(" {:>12}" for _ in metric_list) + " {:>10}"
    click.echo("Benchmark Results")
    click.echo("-" * (12 * (len(metric_list) + 2)))
    click.echo(row_format.format(*header))

    for row in benchmark_rows:
        metric_values = []
        for metric_name in metric_list:
            value = row["metrics"].get(metric_name)
            metric_values.append(f"{value:.4f}" if value is not None else "--")
        click.echo(
            row_format.format(
                row["model_name"],
                *metric_values,
                f"{row['training_time_seconds']:.2f}",
            )
        )

    click.echo("-" * (12 * (len(metric_list) + 2)))
    click.echo(
        f"Best model ({ranking_metric}): {best_row['model_name']} "
        f"with {best_row['metrics'].get(ranking_metric, float('nan')):.4f}"
    )

    output_path = (
        Path(output)
        if output
        else cfg.paths.output_dir / "benchmark_results.json"
    )

    payload = {
        "requested_models": deduped_models,
        "metrics": metric_list,
        "ranking_metric": ranking_metric,
        "results": benchmark_rows,
        "best_model": {
            "model_key": best_row["model_key"],
            "model_name": best_row["model_name"],
            "metrics": best_row["metrics"],
        },
    }

    saved_path = _write_json(payload, output_path)
    click.echo(f"Benchmark report saved to {saved_path}")


@cli.command(name="compare-explanations")
@click.option(
    "--model-path",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to the trained model file.",
)
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Dataset to use for explanations.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--indices",
    help="Comma-separated sample indices to compare (default: first 5 rows).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Destination JSON file for the comparison report.",
)
def compare_explanations_cmd(
    model_path: str,
    data_path: str,
    config: str | None,
    indices: str | None,
    output: str | None,
) -> None:
    """Compare SHAP explanations for selected samples."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()
    logger = setup_logging(cfg.logging)
    logger.info("Comparing explanations for selected samples...")

    model = CreditModel(cfg)
    model.load(model_path)

    df, _ = load_data(data_path, cfg, validate=True)
    _, X_test, _, y_test, _ = preprocess_data(df, cfg)
    y_pred = model.predict(X_test, protected_attribute=cfg.cdi.proxy_column)

    indices_list = _parse_indices_option(indices, len(X_test))

    explainer = Explainer(cfg)
    explainer.setup_shap_explainer(model)

    output_path = (
        Path(output)
        if output
        else cfg.paths.reports_dir / "explanations" / "comparison.json"
    )

    comparison = explainer.compare_explanations(
        X_test,
        indices_list,
        y_pred,
        output_path=output_path,
        protected_attribute=cfg.cdi.proxy_column,
    )

    click.echo(
        f"Compared {comparison['num_samples']} samples. "
        f"Report saved to {output_path}"
    )


@cli.command()
@click.option(
    "--model-path",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to the saved model file.",
)
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to data file to explain.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--num-samples",
    "-n",
    type=int,
    default=5,
    help="Number of samples to explain.",
)
def explain(
    model_path: str,
    data_path: str,
    config: str | None,
    num_samples: int,
) -> None:
    """Generate explanations for predictions using a saved model."""
    # Load configuration
    cfg = load_config(config)
    cfg.paths.ensure_dirs()

    logger = setup_logging(cfg.logging)
    logger.info("Generating explanations for saved model...")

    # Load model
    model = CreditModel(cfg)
    model.load(model_path)

    # Load and preprocess data
    df, _ = load_data(data_path, cfg)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, cfg)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Generate explanations
    explainer = Explainer(cfg)
    explainer.setup_shap_explainer(model)

    explanations = explainer.explain_samples(
        X_test, y_pred, num_samples=num_samples
    )

    click.echo(f"Generated {len(explanations)} explanations")


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config/default_config.yaml",
    help="Output path for the default config file.",
)
def init_config(output: str) -> None:
    """Generate a default configuration file."""
    config = Config()
    config.to_yaml(output)
    click.echo(f"Default configuration written to {output}")


@cli.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to the input CSV data file.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for validation report.",
)
def validate(
    data_path: str,
    config: str | None,
    output: str | None,
) -> None:
    """Validate data quality and report issues."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()

    logger = setup_logging(cfg.logging)
    logger.info("="*60)
    logger.info("Data Validation")
    logger.info("="*60)

    # Load and validate data
    data_loader = DataLoader(cfg)
    raw_data = data_loader.load_csv(data_path)

    preprocessor = Preprocessor(cfg)
    validation_report = preprocessor.validate_data(raw_data)

    logger.info(f"\nData Shape: {validation_report['shape']}")
    logger.info(f"Missing Values: {validation_report['missing_values']}")
    logger.info(f"Duplicates: {validation_report['duplicates']}")

    if validation_report.get('issues'):
        logger.warning(f"\nFound {len(validation_report['issues'])} issues:")
        for issue in validation_report['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\n✓ No data quality issues found")

    # Save report
    if output:
        output_path = Path(output)
    else:
        output_path = cfg.paths.output_dir / "validation_report.json"

    save_metrics_to_json(validation_report, output_path, logger)
    click.echo(f"\nValidation complete! Report saved to {output_path}")


@cli.command()
@click.option(
    "--model-path",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to the saved model file.",
)
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation data file.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for plots.",
)
def visualize_fairness(
    model_path: str,
    data_path: str,
    config: str | None,
    output: str | None,
) -> None:
    """Generate fairness visualization plots and comprehensive report."""
    cfg = load_config(config)
    cfg.paths.ensure_dirs()

    logger = setup_logging(cfg.logging)
    logger.info("="*60)
    logger.info("Fairness Visualization")
    logger.info("="*60)

    # Load model
    model = CreditModel(cfg)
    model.load(model_path)

    # Load and preprocess data
    data_loader = DataLoader(cfg)
    raw_data = data_loader.load_csv(data_path)

    preprocessor = Preprocessor(cfg)
    X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(raw_data)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate fairness metrics
    fairness_analyzer = FairnessAnalyzer(cfg)
    fairness_metrics = fairness_analyzer.calculate_metrics(X_test, y_test, y_pred)

    # Generate plots
    if output:
        output_dir = Path(output)
    else:
        output_dir = cfg.paths.reports_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = fairness_analyzer.plot_fairness_metrics(
        fairness_metrics,
        save_path=output_dir / "fairness_metrics.png"
    )
    logger.info(f"Fairness plot saved to {plot_path}")

    # Generate comprehensive report
    report_path = fairness_analyzer.generate_fairness_report(
        fairness_metrics,
        X_test,
        y_test,
        y_pred,
        save_path=output_dir.parent / "fairness_report.json"
    )
    logger.info(f"Fairness report saved to {report_path}")

    click.echo(f"\nVisualization complete! Outputs saved to {output_dir}")


if __name__ == "__main__":
    cli()
