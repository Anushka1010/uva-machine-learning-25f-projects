#!/usr/bin/env python3
"""Model version comparison script.

Compare different versions of trained models to analyze performance differences
and support A/B testing scenarios.

Usage:
    python scripts/compare_model_versions.py --model-a models/model_v1.pkl --model-b models/model_v2.pkl --data-path data/credit.csv
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from src.utils import setup_logging
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import CreditModel
from src.fairness import FairnessAnalyzer


def load_and_evaluate_model(model_path: Path, X_test, y_test, config):
    """Load a model and evaluate it."""
    model = CreditModel(config)
    model.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = model.evaluate(X_test, y_test)

    # Get model metadata
    version_info = model.get_version()
    metadata = model.get_training_metadata()

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "version": version_info,
        "metadata": metadata,
        "is_calibrated": model._calibrated_model is not None,
    }


def compare_metrics(model_a_result, model_b_result):
    """Compare metrics between two models."""
    metrics_a = model_a_result["metrics"]
    metrics_b = model_b_result["metrics"]

    comparison = {}
    for metric in metrics_a.keys():
        diff = metrics_b[metric] - metrics_a[metric]
        pct_change = (diff / metrics_a[metric] * 100) if metrics_a[metric] != 0 else 0

        comparison[metric] = {
            "model_a": float(metrics_a[metric]),
            "model_b": float(metrics_b[metric]),
            "difference": float(diff),
            "percent_change": float(pct_change),
            "winner": "Model B" if diff > 0 else "Model A" if diff < 0 else "Tie"
        }

    return comparison


def compare_fairness(model_a_result, model_b_result, X_test, y_test, config):
    """Compare fairness metrics between two models."""
    analyzer = FairnessAnalyzer(config)

    fairness_a = analyzer.calculate_metrics(X_test, y_test, model_a_result["y_pred"])
    fairness_b = analyzer.calculate_metrics(X_test, y_test, model_b_result["y_pred"])

    return {
        "model_a": fairness_a.to_dict(),
        "model_b": fairness_b.to_dict(),
        "comparison": {
            "disparate_impact": {
                "model_a": fairness_a.disparate_impact,
                "model_b": fairness_b.disparate_impact,
                "difference": fairness_b.disparate_impact - fairness_a.disparate_impact
            },
            "statistical_parity_diff": {
                "model_a": fairness_a.statistical_parity_difference,
                "model_b": fairness_b.statistical_parity_difference,
                "difference": fairness_b.statistical_parity_difference - fairness_a.statistical_parity_difference
            }
        }
    }


def analyze_prediction_differences(model_a_result, model_b_result):
    """Analyze differences in predictions between models."""
    pred_a = model_a_result["y_pred"]
    pred_b = model_b_result["y_pred"]

    total = len(pred_a)
    agreements = (pred_a == pred_b).sum()
    disagreements = total - agreements

    return {
        "total_predictions": int(total),
        "agreements": int(agreements),
        "disagreements": int(disagreements),
        "agreement_rate": float(agreements / total),
        "disagreement_rate": float(disagreements / total)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two model versions"
    )
    parser.add_argument(
        "--model-a",
        required=True,
        type=Path,
        help="Path to first model (baseline)"
    )
    parser.add_argument(
        "--model-b",
        required=True,
        type=Path,
        help="Path to second model (challenger)"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for comparison report"
    )
    parser.add_argument(
        "--include-fairness",
        action="store_true",
        help="Include fairness metrics comparison"
    )

    args = parser.parse_args()

    # Verify model files exist
    if not args.model_a.exists():
        print(f"Error: Model A not found: {args.model_a}")
        sys.exit(1)
    if not args.model_b.exists():
        print(f"Error: Model B not found: {args.model_b}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    config.paths.ensure_dirs()

    logger = setup_logging(config.logging)
    logger.info("="*70)
    logger.info("MODEL VERSION COMPARISON")
    logger.info("="*70)
    logger.info(f"Model A (Baseline):  {args.model_a}")
    logger.info(f"Model B (Challenger): {args.model_b}")
    logger.info(f"Test Data:           {args.data_path}")
    logger.info("="*70)

    # Load and preprocess data
    logger.info("\nLoading test data...")
    data_loader = DataLoader(config)
    raw_data = data_loader.load_csv(args.data_path)

    preprocessor = Preprocessor(config)
    X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(raw_data)

    logger.info(f"Test samples: {len(X_test)}")

    # Evaluate both models
    logger.info("\nEvaluating Model A (Baseline)...")
    model_a_result = load_and_evaluate_model(args.model_a, X_test, y_test, config)

    logger.info("Evaluating Model B (Challenger)...")
    model_b_result = load_and_evaluate_model(args.model_b, X_test, y_test, config)

    # Compare metrics
    logger.info("\nComparing performance metrics...")
    metric_comparison = compare_metrics(model_a_result, model_b_result)

    # Compare predictions
    prediction_comparison = analyze_prediction_differences(model_a_result, model_b_result)

    # Build comparison report
    comparison_report = {
        "model_a": {
            "path": str(args.model_a),
            "version": model_a_result["version"],
            "is_calibrated": model_a_result["is_calibrated"],
            "metrics": model_a_result["metrics"]
        },
        "model_b": {
            "path": str(args.model_b),
            "version": model_b_result["version"],
            "is_calibrated": model_b_result["is_calibrated"],
            "metrics": model_b_result["metrics"]
        },
        "metric_comparison": metric_comparison,
        "prediction_comparison": prediction_comparison
    }

    # Fairness comparison
    if args.include_fairness:
        logger.info("Comparing fairness metrics...")
        fairness_comparison = compare_fairness(
            model_a_result, model_b_result, X_test, y_test, config
        )
        comparison_report["fairness_comparison"] = fairness_comparison

    # Display results
    logger.info("\n" + "="*70)
    logger.info("COMPARISON RESULTS")
    logger.info("="*70)

    logger.info("\nPerformance Metrics:")
    for metric, data in metric_comparison.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Model A:        {data['model_a']:.4f}")
        logger.info(f"  Model B:        {data['model_b']:.4f}")
        logger.info(f"  Difference:     {data['difference']:+.4f} ({data['percent_change']:+.2f}%)")
        logger.info(f"  Winner:         {data['winner']}")

    logger.info(f"\nPrediction Agreement:")
    logger.info(f"  Agreements:     {prediction_comparison['agreements']:,} ({prediction_comparison['agreement_rate']*100:.2f}%)")
    logger.info(f"  Disagreements:  {prediction_comparison['disagreements']:,} ({prediction_comparison['disagreement_rate']*100:.2f}%)")

    # Recommendation
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)

    # Count wins
    wins_a = sum(1 for m in metric_comparison.values() if m["winner"] == "Model A")
    wins_b = sum(1 for m in metric_comparison.values() if m["winner"] == "Model B")

    if wins_b > wins_a:
        logger.info("✓ Model B (Challenger) shows improvement")
        logger.info(f"  Better on {wins_b}/{len(metric_comparison)} metrics")
    elif wins_a > wins_b:
        logger.info("⚠ Model A (Baseline) performs better")
        logger.info(f"  Better on {wins_a}/{len(metric_comparison)} metrics")
    else:
        logger.info("- Models perform similarly")

    # Save report
    output_path = args.output or config.paths.reports_dir / "model_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(comparison_report, f, indent=2, default=str)

    logger.info(f"\nDetailed comparison saved to: {output_path}")
    logger.info("\nComparison complete!")


if __name__ == "__main__":
    main()
