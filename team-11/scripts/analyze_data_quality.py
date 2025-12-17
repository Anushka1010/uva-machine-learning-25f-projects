#!/usr/bin/env python3
"""Comprehensive data quality analysis script.

Analyzes data quality, generates detailed reports, and identifies potential issues
before training models.

Usage:
    python scripts/analyze_data_quality.py --data-path data/credit.csv
    python scripts/analyze_data_quality.py --data-path data/credit.csv --output reports/data_quality.json
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from src.utils import setup_logging
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing values in the dataset."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    return {
        "total_missing": int(missing.sum()),
        "columns_with_missing": int((missing > 0).sum()),
        "missing_by_column": {
            col: {
                "count": int(missing[col]),
                "percentage": float(missing_pct[col])
            }
            for col in df.columns if missing[col] > 0
        }
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze duplicate rows."""
    n_duplicates = df.duplicated().sum()

    return {
        "total_duplicates": int(n_duplicates),
        "percentage": float((n_duplicates / len(df)) * 100)
    }


def analyze_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data types and potential conversion issues."""
    type_counts = df.dtypes.value_counts().to_dict()

    return {
        "type_distribution": {str(k): int(v) for k, v in type_counts.items()},
        "columns_by_type": {
            dtype: list(df.select_dtypes(include=[dtype]).columns)
            for dtype in df.dtypes.unique()
        }
    }


def analyze_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Analyze numeric columns for outliers and distributions."""
    results = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors='coerce')

        if values.isnull().all():
            results[col] = {"status": "all_null"}
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = ((values < lower_bound) | (values > upper_bound)).sum()

        results[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "q25": float(q1),
            "median": float(values.median()),
            "q75": float(q3),
            "outliers": int(outliers),
            "outlier_percentage": float((outliers / len(values)) * 100),
            "zeros": int((values == 0).sum()),
            "negative": int((values < 0).sum())
        }

    return results


def analyze_categorical_columns(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Any]:
    """Analyze categorical columns for cardinality and distribution."""
    results = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        value_counts = df[col].value_counts()

        results[col] = {
            "cardinality": int(df[col].nunique()),
            "most_common": value_counts.head(5).to_dict(),
            "distribution": {
                "balanced": bool(value_counts.std() < value_counts.mean()),
                "min_count": int(value_counts.min()),
                "max_count": int(value_counts.max())
            }
        }

    return results


def analyze_target_balance(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Analyze target variable balance."""
    if target_col not in df.columns:
        return {"status": "target_not_found"}

    value_counts = df[target_col].value_counts()
    total = len(df)

    return {
        "distribution": value_counts.to_dict(),
        "percentages": (value_counts / total * 100).to_dict(),
        "is_balanced": bool(abs(value_counts.iloc[0] - value_counts.iloc[1]) < total * 0.1),
        "imbalance_ratio": float(value_counts.max() / value_counts.min())
    }


def identify_issues(analysis: Dict[str, Any]) -> List[str]:
    """Identify data quality issues based on analysis."""
    issues = []

    # Check missing values
    if analysis["missing_values"]["total_missing"] > 0:
        pct = (analysis["missing_values"]["total_missing"] /
               analysis["basic_stats"]["total_rows"]) * 100
        issues.append(f"Dataset has {analysis['missing_values']['total_missing']} missing values ({pct:.2f}%)")

    # Check duplicates
    if analysis["duplicates"]["total_duplicates"] > 0:
        issues.append(f"Found {analysis['duplicates']['total_duplicates']} duplicate rows ({analysis['duplicates']['percentage']:.2f}%)")

    # Check target balance
    if "target_analysis" in analysis:
        if not analysis["target_analysis"]["is_balanced"]:
            ratio = analysis["target_analysis"]["imbalance_ratio"]
            issues.append(f"Target variable is imbalanced (ratio: {ratio:.2f}:1)")

    # Check outliers
    if "numeric_analysis" in analysis:
        high_outlier_cols = [
            col for col, stats in analysis["numeric_analysis"].items()
            if isinstance(stats, dict) and stats.get("outlier_percentage", 0) > 5
        ]
        if high_outlier_cols:
            issues.append(f"High outlier percentage in columns: {', '.join(high_outlier_cols)}")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive data quality analysis"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to data CSV file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for quality report JSON"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.paths.ensure_dirs()

    logger = setup_logging(config.logging)
    logger.info("="*70)
    logger.info("DATA QUALITY ANALYSIS")
    logger.info("="*70)
    logger.info(f"Data: {args.data_path}")
    logger.info("="*70)

    # Load data
    logger.info("\nLoading data...")
    data_loader = DataLoader(config)
    df = data_loader.load_csv(args.data_path)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run comprehensive analysis
    logger.info("\nAnalyzing data quality...")

    analysis = {
        "data_path": str(args.data_path),
        "basic_stats": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        },
        "missing_values": analyze_missing_values(df),
        "duplicates": analyze_duplicates(df),
        "data_types": analyze_data_types(df),
        "numeric_analysis": analyze_numeric_columns(df, config.data.numeric_columns),
        "categorical_analysis": analyze_categorical_columns(df, config.data.categorical_columns),
        "target_analysis": analyze_target_balance(df, config.model.target_column)
    }

    # Identify issues
    analysis["issues"] = identify_issues(analysis)
    analysis["quality_score"] = max(0, 100 - len(analysis["issues"]) * 5)

    # Display summary
    logger.info("\n" + "="*70)
    logger.info("QUALITY SUMMARY")
    logger.info("="*70)
    logger.info(f"Quality Score: {analysis['quality_score']}/100")
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total Rows:    {analysis['basic_stats']['total_rows']:,}")
    logger.info(f"  Total Columns: {analysis['basic_stats']['total_columns']}")
    logger.info(f"  Memory Usage:  {analysis['basic_stats']['memory_usage_mb']:.2f} MB")

    logger.info(f"\nData Quality Checks:")
    logger.info(f"  Missing Values:  {analysis['missing_values']['total_missing']:,} ({analysis['missing_values']['columns_with_missing']} columns)")
    logger.info(f"  Duplicate Rows:  {analysis['duplicates']['total_duplicates']:,} ({analysis['duplicates']['percentage']:.2f}%)")

    if analysis["issues"]:
        logger.info(f"\nIssues Found ({len(analysis['issues'])}):")
        for i, issue in enumerate(analysis["issues"], 1):
            logger.info(f"  {i}. {issue}")
    else:
        logger.info("\n✓ No major data quality issues found")

    # Save report
    output_path = args.output or config.paths.reports_dir / "data_quality.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    logger.info(f"\nDetailed report saved to: {output_path}")
    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
