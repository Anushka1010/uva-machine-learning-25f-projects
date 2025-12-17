#!/usr/bin/env python3
"""Model benchmarking script.

Compare performance of multiple models (XGBoost, Random Forest, Logistic Regression)
on the credit prediction task and generate a comprehensive comparison report.

Usage:
    python scripts/benchmark_models.py --data-path data/credit.csv
    python scripts/benchmark_models.py --data-path data/credit.csv --models xgboost,rf,lr --cv-folds 10
"""

import sys
from pathlib import Path
import argparse
import json
import time
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from src.utils import setup_logging, set_random_seed
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import CreditModel
from src.fairness import FairnessAnalyzer


def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, config):
    """Train and evaluate XGBoost model."""
    start_time = time.time()

    model = CreditModel(config)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    training_time = time.time() - start_time

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "training_time": training_time,
        "model": model,
    }


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, config):
    """Train and evaluate Random Forest model."""
    start_time = time.time()

    # Drop protected attribute for fair comparison
    proxy_col = config.cdi.proxy_column
    X_train_clean = X_train.drop(columns=[proxy_col])
    X_test_clean = X_test.drop(columns=[proxy_col])

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=config.model.random_state,
        n_jobs=-1
    )
    model.fit(X_train_clean, y_train)

    y_pred = model.predict(X_test_clean)
    y_prob = model.predict_proba(X_test_clean)[:, 1]

    training_time = time.time() - start_time

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "training_time": training_time,
        "model": model,
    }


def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, config):
    """Train and evaluate Logistic Regression model."""
    start_time = time.time()

    # Drop protected attribute for fair comparison
    proxy_col = config.cdi.proxy_column
    X_train_clean = X_train.drop(columns=[proxy_col])
    X_test_clean = X_test.drop(columns=[proxy_col])

    model = LogisticRegression(
        max_iter=1000,
        random_state=config.model.random_state,
        n_jobs=-1
    )
    model.fit(X_train_clean, y_train)

    y_pred = model.predict(X_test_clean)
    y_prob = model.predict_proba(X_test_clean)[:, 1]

    training_time = time.time() - start_time

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "training_time": training_time,
        "model": model,
    }


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple models for credit prediction"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--models",
        default="xgboost,rf,lr",
        help="Comma-separated list of models: xgboost,rf,lr"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for benchmark results"
    )
    parser.add_argument(
        "--include-fairness",
        action="store_true",
        help="Include fairness metrics in comparison"
    )

    args = parser.parse_args()

    # Parse model list
    model_list = [m.strip().lower() for m in args.models.split(",")]
    valid_models = {"xgboost", "rf", "lr"}
    if not all(m in valid_models for m in model_list):
        print(f"Error: Invalid model specified. Valid models: {valid_models}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    config.paths.ensure_dirs()

    logger = setup_logging(config.logging)
    logger.info("="*70)
    logger.info("MODEL BENCHMARKING")
    logger.info("="*70)
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Models: {', '.join(model_list)}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info("="*70)

    set_random_seed(config.model.random_state)

    # Load and preprocess data
    logger.info("\nLoading and preprocessing data...")
    data_loader = DataLoader(config)
    raw_data = data_loader.load_csv(args.data_path)

    preprocessor = Preprocessor(config)
    X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(raw_data)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Benchmark each model
    results = {}
    model_map = {
        "xgboost": ("XGBoost", train_and_evaluate_xgboost),
        "rf": ("Random Forest", train_and_evaluate_random_forest),
        "lr": ("Logistic Regression", train_and_evaluate_logistic_regression),
    }

    for model_key in model_list:
        model_name, train_func = model_map[model_key]

        logger.info(f"\n{'='*70}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*70}")

        try:
            result = train_func(X_train, X_test, y_train, y_test, config)

            # Calculate metrics
            metrics = calculate_metrics(y_test, result["y_pred"], result["y_prob"])
            metrics["training_time"] = result["training_time"]

            # Calculate fairness metrics if requested
            if args.include_fairness and model_key == "xgboost":
                analyzer = FairnessAnalyzer(config)
                fairness_metrics = analyzer.calculate_metrics(
                    X_test, y_test, result["y_pred"]
                )
                metrics["fairness"] = fairness_metrics.to_dict()

            results[model_name] = metrics

            # Display metrics
            logger.info(f"\n{model_name} Results:")
            logger.info(f"  Accuracy:      {metrics['accuracy']:.4f}")
            logger.info(f"  Precision:     {metrics['precision']:.4f}")
            logger.info(f"  Recall:        {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:      {metrics['f1']:.4f}")
            logger.info(f"  ROC AUC:       {metrics['roc_auc']:.4f}")
            logger.info(f"  Training Time: {metrics['training_time']:.2f}s")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Generate comparison table
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*70)

    df_results = pd.DataFrame(results).T
    if "error" not in df_results.columns:
        # Sort by F1 score
        df_results = df_results.sort_values("f1", ascending=False)

        logger.info("\n" + df_results.to_string())

        # Determine best model
        best_model = df_results.index[0]
        logger.info(f"\n{'='*70}")
        logger.info(f"RECOMMENDATION: {best_model}")
        logger.info(f"{'='*70}")
        logger.info(f"  Best F1 Score: {df_results.loc[best_model, 'f1']:.4f}")
        logger.info(f"  ROC AUC:       {df_results.loc[best_model, 'roc_auc']:.4f}")

    # Save results
    output_path = args.output or config.paths.output_dir / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "models_compared": model_list,
        "cv_folds": args.cv_folds,
        "results": results,
        "best_model": best_model if "error" not in df_results.columns else None,
        "data_path": str(args.data_path),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")
    logger.info("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
