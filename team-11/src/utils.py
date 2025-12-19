"""Utility functions and shared helpers.

This module contains logging setup, validation utilities, and other
shared functionality used across the credit prediction pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from config.config import Config, LoggingConfig, get_config


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Configure and return the root logger.

    Args:
        config: Logging configuration. If None, uses default config.

    Returns:
        Configured root logger.
    """
    if config is None:
        config = get_config().logging

    # Create logger
    logger = logging.getLogger("credit_prediction")
    logger.setLevel(getattr(logging, config.level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if config.log_to_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Name of the module requesting the logger.

    Returns:
        Logger instance with the given name.
    """
    return logging.getLogger(f"credit_prediction.{name}")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    logger: logging.Logger | None = None,
) -> tuple[bool, list[str]]:
    """Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        logger: Optional logger for reporting issues.

    Returns:
        Tuple of (is_valid, missing_columns).
    """
    if logger is None:
        logger = get_logger("utils")

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        logger.warning(f"DataFrame missing required columns: {missing}")
        return False, missing

    return True, []


def report_coercion_failures(
    df: pd.DataFrame,
    column: str,
    original_count: int,
    logger: logging.Logger | None = None,
) -> int:
    """Report how many values failed numeric coercion.

    Args:
        df: DataFrame after coercion.
        column: Column that was coerced.
        original_count: Original row count.
        logger: Optional logger for reporting.

    Returns:
        Number of NaN values created by coercion.
    """
    if logger is None:
        logger = get_logger("utils")

    nan_count = df[column].isna().sum()
    if nan_count > 0:
        pct = (nan_count / original_count) * 100
        logger.warning(
            f"Column '{column}': {nan_count} values ({pct:.2f}%) "
            f"became NaN after numeric coercion"
        )

    return int(nan_count)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    # Try to set XGBoost seed if available
    try:
        import xgboost as xgb

        # XGBoost uses random_state in model params, not a global seed
    except ImportError:
        pass

    logger = get_logger("utils")
    logger.info(f"Random seed set to {seed}")


def calculate_metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate common classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional).

    Returns:
        Dictionary of metric names to values.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Can happen if only one class present
            pass

    return metrics


def save_metrics_to_json(
    metrics: dict[str, Any],
    path: str | Path,
    logger: logging.Logger | None = None,
) -> None:
    """Save metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary of metrics.
        path: Output file path.
        logger: Optional logger.
    """
    import json

    if logger is None:
        logger = get_logger("utils")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    with open(path, "w") as f:
        json.dump(_convert(metrics), f, indent=2)

    logger.info(f"Metrics saved to {path}")


def format_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
) -> str:
    """Format a confusion matrix for pretty printing.

    Args:
        cm: Confusion matrix (2x2 for binary classification).
        labels: Optional class labels.

    Returns:
        Formatted string representation.
    """
    if labels is None:
        labels = ["Negative (0)", "Positive (1)"]

    lines = [
        "Confusion Matrix:",
        f"                    Predicted",
        f"                    {labels[0]:^15} {labels[1]:^15}",
        f"Actual {labels[0]:^12}  {cm[0, 0]:^15} {cm[0, 1]:^15}",
        f"       {labels[1]:^12}  {cm[1, 0]:^15} {cm[1, 1]:^15}",
        "",
        f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}",
    ]

    return "\n".join(lines)
