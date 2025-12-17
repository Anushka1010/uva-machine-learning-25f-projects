"""Protected attribute handling utilities.

This module provides DRY utilities for handling protected attributes
across the credit prediction pipeline. Instead of repeating the same
pattern of "check if None, default to config, drop from dataframe"
everywhere, we centralize it here.

Zen of Python: DRY, DRY, DRY! 🐶
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, Any, runtime_checkable

import pandas as pd
import numpy as np

from config.config import Config, get_config


@runtime_checkable
class HasConfig(Protocol):
    """Protocol for objects that have a config attribute."""
    config: Config


@dataclass
class ProtectedAttributeContext:
    """Context for working with protected attributes.
    
    Holds both the original data and the filtered data for easy access.
    """
    original_df: pd.DataFrame
    filtered_df: pd.DataFrame
    protected_column: str
    protected_values: np.ndarray
    
    @property
    def privileged_mask(self) -> np.ndarray:
        """Boolean mask for privileged group (protected_value == 0)."""
        return self.protected_values == 0
    
    @property
    def unprivileged_mask(self) -> np.ndarray:
        """Boolean mask for unprivileged group (protected_value == 1)."""
        return self.protected_values == 1


def get_protected_column(config: Config | None = None) -> str:
    """Get the protected attribute column name from config.
    
    Args:
        config: Optional config object. Uses global config if None.
        
    Returns:
        The protected attribute column name.
    """
    if config is None:
        config = get_config()
    return config.cdi.proxy_column


def resolve_protected_attribute(
    protected_attribute: str | None,
    config: Config | None = None,
) -> str:
    """Resolve protected attribute to a concrete column name.
    
    Args:
        protected_attribute: Optional column name. If None, uses config default.
        config: Optional config object. Uses global config if None.
        
    Returns:
        The resolved protected attribute column name.
    """
    if protected_attribute is not None:
        return protected_attribute
    return get_protected_column(config)


def drop_protected_attribute(
    df: pd.DataFrame,
    protected_attribute: str | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Drop protected attribute from DataFrame if present.
    
    Args:
        df: Input DataFrame.
        protected_attribute: Column to drop. Uses config default if None.
        config: Optional config object.
        
    Returns:
        DataFrame with protected attribute removed.
    """
    column = resolve_protected_attribute(protected_attribute, config)
    
    if column in df.columns:
        return df.drop(columns=[column])
    return df


def extract_protected_values(
    df: pd.DataFrame,
    protected_attribute: str | None = None,
    config: Config | None = None,
) -> np.ndarray:
    """Extract protected attribute values from DataFrame.
    
    Args:
        df: Input DataFrame.
        protected_attribute: Column to extract. Uses config default if None.
        config: Optional config object.
        
    Returns:
        Array of protected attribute values.
        
    Raises:
        KeyError: If protected attribute column not found.
    """
    column = resolve_protected_attribute(protected_attribute, config)
    
    if column not in df.columns:
        raise KeyError(
            f"Protected attribute column '{column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    return df[column].values


def prepare_features_without_protected(
    df: pd.DataFrame,
    protected_attribute: str | None = None,
    config: Config | None = None,
) -> ProtectedAttributeContext:
    """Prepare features by removing protected attribute while preserving access.
    
    This is the main DRY function that replaces the repeated pattern of:
    1. Copy the dataframe
    2. Resolve protected attribute
    3. Drop protected attribute if present
    4. Sometimes extract the values for later use
    
    Args:
        df: Input DataFrame.
        protected_attribute: Column to handle. Uses config default if None.
        config: Optional config object.
        
    Returns:
        ProtectedAttributeContext with original, filtered, and metadata.
    """
    column = resolve_protected_attribute(protected_attribute, config)
    original_df = df.copy()
    
    # Extract protected values if available
    if column in df.columns:
        protected_values = df[column].values
        filtered_df = df.drop(columns=[column])
    else:
        protected_values = np.array([])
        filtered_df = df.copy()
    
    return ProtectedAttributeContext(
        original_df=original_df,
        filtered_df=filtered_df,
        protected_column=column,
        protected_values=protected_values,
    )


@contextmanager
def without_protected_attribute(
    df: pd.DataFrame,
    protected_attribute: str | None = None,
    config: Config | None = None,
):
    """Context manager for temporarily working without protected attribute.
    
    Usage:
        with without_protected_attribute(df) as ctx:
            # ctx.filtered_df has protected attribute removed
            model.predict(ctx.filtered_df)
            # ctx.protected_values still available for fairness analysis
            
    Args:
        df: Input DataFrame.
        protected_attribute: Column to exclude. Uses config default if None.
        config: Optional config object.
        
    Yields:
        ProtectedAttributeContext with filtered data and original access.
    """
    ctx = prepare_features_without_protected(df, protected_attribute, config)
    yield ctx


def for_protected_attribute(obj: HasConfig) -> str:
    """Get protected attribute column for an object with config.
    
    Convenience function for class methods.
    
    Args:
        obj: Object with a `config` attribute.
        
    Returns:
        Protected attribute column name.
    """
    return get_protected_column(obj.config)
