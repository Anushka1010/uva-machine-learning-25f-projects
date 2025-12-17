"""Input validation utilities for the credit prediction pipeline.

This module provides validation functions and decorators to ensure
data quality and catch errors early. Because garbage in = garbage out! 🐶
"""

from functools import wraps
from typing import Any, Callable, TypeVar
import logging

import pandas as pd
import numpy as np

from src.utils import get_logger

logger = get_logger("validators")

F = TypeVar("F", bound=Callable[..., Any])


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate that a DataFrame is not empty.
    
    Args:
        df: DataFrame to validate.
        name: Name for error messages.
        
    Raises:
        ValidationError: If DataFrame is empty.
    """
    if df.empty:
        raise ValidationError(f"{name} is empty - cannot proceed with empty data!")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    name: str = "DataFrame",
) -> None:
    """Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        name: Name for error messages.
        
    Raises:
        ValidationError: If any required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"{name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_no_nan_in_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    name: str = "DataFrame",
) -> None:
    """Validate that specified columns have no NaN values.
    
    Args:
        df: DataFrame to validate.
        columns: Columns to check. If None, checks all columns.
        name: Name for error messages.
        
    Raises:
        ValidationError: If NaN values found.
    """
    if columns is None:
        columns = list(df.columns)
    
    nan_counts = {}
    for col in columns:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_counts[col] = nan_count
    
    if nan_counts:
        raise ValidationError(
            f"{name} contains NaN values in columns: {nan_counts}"
        )


def validate_predictions_shape(
    y_pred: np.ndarray,
    y_true: pd.Series | np.ndarray,
    name: str = "predictions",
) -> None:
    """Validate that predictions match expected shape.
    
    Args:
        y_pred: Predicted values.
        y_true: True labels (to compare length).
        name: Name for error messages.
        
    Raises:
        ValidationError: If shapes don't match.
    """
    expected_len = len(y_true)
    actual_len = len(y_pred)
    
    if expected_len != actual_len:
        raise ValidationError(
            f"{name} length mismatch: expected {expected_len}, got {actual_len}"
        )


def validate_probability_range(
    y_prob: np.ndarray,
    name: str = "probabilities",
) -> None:
    """Validate that probabilities are in [0, 1] range.
    
    Args:
        y_prob: Probability values.
        name: Name for error messages.
        
    Raises:
        ValidationError: If values outside valid range.
    """
    min_val = y_prob.min()
    max_val = y_prob.max()
    
    if min_val < 0 or max_val > 1:
        raise ValidationError(
            f"{name} contains values outside [0, 1]: min={min_val:.4f}, max={max_val:.4f}"
        )


def validate_binary_labels(
    y: np.ndarray | pd.Series,
    name: str = "labels",
) -> None:
    """Validate that labels are binary (0 or 1).
    
    Args:
        y: Label values.
        name: Name for error messages.
        
    Raises:
        ValidationError: If non-binary values found.
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    unique = np.unique(y[~np.isnan(y)])
    valid_values = {0, 1, 0.0, 1.0}
    
    invalid = [v for v in unique if v not in valid_values]
    if invalid:
        raise ValidationError(
            f"{name} contains non-binary values: {invalid}. Expected only 0 and 1."
        )


def validate_model_trained(model: Any, name: str = "Model") -> None:
    """Validate that a model has been trained.
    
    Args:
        model: Model object (should have _model attribute or similar).
        name: Name for error messages.
        
    Raises:
        ValidationError: If model not trained.
    """
    # Check common patterns for "model is trained"
    if hasattr(model, "_model") and model._model is None:
        raise ValidationError(f"{name} has not been trained. Call train() first.")
    if hasattr(model, "is_fitted") and not model.is_fitted:
        raise ValidationError(f"{name} has not been fitted. Call fit() first.")


# =============================================================================
# Validation Decorators
# =============================================================================

def require_non_empty_dataframe(arg_name: str = "df") -> Callable[[F], F]:
    """Decorator to require a non-empty DataFrame argument.
    
    Args:
        arg_name: Name of the DataFrame argument to validate.
        
    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get the DataFrame from args or kwargs
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            df = None
            if arg_name in kwargs:
                df = kwargs[arg_name]
            elif arg_name in params:
                idx = params.index(arg_name)
                if idx < len(args):
                    df = args[idx]
            
            if df is not None and isinstance(df, pd.DataFrame):
                validate_dataframe_not_empty(df, arg_name)
            
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def require_trained_model(model_attr: str = "_model") -> Callable[[F], F]:
    """Decorator to require that self has a trained model.
    
    Args:
        model_attr: Name of the model attribute to check.
        
    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            model = getattr(self, model_attr, None)
            if model is None:
                raise ValidationError(
                    f"Model not trained. Call train() before calling {func.__name__}()"
                )
            return func(self, *args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def log_validation_errors(func: F) -> F:
    """Decorator to log validation errors before re-raising.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.error(f"Validation failed in {func.__name__}: {e}")
            raise
    return wrapper  # type: ignore
