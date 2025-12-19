"""Tests for protected attribute utilities.

These tests verify the DRY utilities for protected attribute handling
work correctly across the pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from src.protected_attribute import (
    get_protected_column,
    resolve_protected_attribute,
    drop_protected_attribute,
    extract_protected_values,
    prepare_features_without_protected,
    without_protected_attribute,
    ProtectedAttributeContext,
)
from config.config import Config


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with protected attribute."""
    return pd.DataFrame({
        "Income": [50000, 75000, 100000],
        "Debt": [10000, 20000, 5000],
        "Proxy_Disadvantaged": [0, 1, 0],
        "CDI": [1, 3, 1],
    })


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


class TestResolveProtectedAttribute:
    """Tests for resolve_protected_attribute function."""

    def test_returns_provided_value_when_given(self):
        """Should return the provided value, not config default."""
        result = resolve_protected_attribute("custom_column")
        assert result == "custom_column"

    def test_returns_config_default_when_none(self, config):
        """Should return config default when None is provided."""
        result = resolve_protected_attribute(None, config)
        assert result == config.cdi.proxy_column
        assert result == "Proxy_Disadvantaged"


class TestDropProtectedAttribute:
    """Tests for drop_protected_attribute function."""

    def test_drops_column_when_present(self, sample_df):
        """Should drop the protected attribute column."""
        result = drop_protected_attribute(sample_df)
        assert "Proxy_Disadvantaged" not in result.columns
        assert "Income" in result.columns

    def test_returns_unchanged_when_column_missing(self, sample_df):
        """Should return unchanged df if column not present."""
        df_no_proxy = sample_df.drop(columns=["Proxy_Disadvantaged"])
        result = drop_protected_attribute(df_no_proxy)
        assert result.equals(df_no_proxy)

    def test_preserves_other_columns(self, sample_df):
        """Should preserve all other columns."""
        result = drop_protected_attribute(sample_df)
        assert list(result.columns) == ["Income", "Debt", "CDI"]


class TestExtractProtectedValues:
    """Tests for extract_protected_values function."""

    def test_extracts_values_correctly(self, sample_df):
        """Should extract the protected attribute values."""
        values = extract_protected_values(sample_df)
        np.testing.assert_array_equal(values, [0, 1, 0])

    def test_raises_on_missing_column(self, sample_df):
        """Should raise KeyError when column not found."""
        df_no_proxy = sample_df.drop(columns=["Proxy_Disadvantaged"])
        with pytest.raises(KeyError, match="not found in DataFrame"):
            extract_protected_values(df_no_proxy)


class TestPrepareFeatures:
    """Tests for prepare_features_without_protected function."""

    def test_returns_context_with_filtered_df(self, sample_df):
        """Should return context with filtered dataframe."""
        ctx = prepare_features_without_protected(sample_df)
        assert isinstance(ctx, ProtectedAttributeContext)
        assert "Proxy_Disadvantaged" not in ctx.filtered_df.columns

    def test_preserves_original_df(self, sample_df):
        """Should preserve the original dataframe unchanged."""
        ctx = prepare_features_without_protected(sample_df)
        assert "Proxy_Disadvantaged" in ctx.original_df.columns

    def test_extracts_protected_values(self, sample_df):
        """Should extract protected values for later use."""
        ctx = prepare_features_without_protected(sample_df)
        np.testing.assert_array_equal(ctx.protected_values, [0, 1, 0])

    def test_stores_column_name(self, sample_df):
        """Should store the protected column name."""
        ctx = prepare_features_without_protected(sample_df)
        assert ctx.protected_column == "Proxy_Disadvantaged"


class TestProtectedAttributeContext:
    """Tests for ProtectedAttributeContext helper properties."""

    def test_privileged_mask(self, sample_df):
        """Should correctly identify privileged group."""
        ctx = prepare_features_without_protected(sample_df)
        np.testing.assert_array_equal(ctx.privileged_mask, [True, False, True])

    def test_unprivileged_mask(self, sample_df):
        """Should correctly identify unprivileged group."""
        ctx = prepare_features_without_protected(sample_df)
        np.testing.assert_array_equal(ctx.unprivileged_mask, [False, True, False])


class TestContextManager:
    """Tests for without_protected_attribute context manager."""

    def test_provides_filtered_df_in_context(self, sample_df):
        """Should provide filtered df within context."""
        with without_protected_attribute(sample_df) as ctx:
            assert "Proxy_Disadvantaged" not in ctx.filtered_df.columns

    def test_provides_protected_values_in_context(self, sample_df):
        """Should provide protected values within context."""
        with without_protected_attribute(sample_df) as ctx:
            np.testing.assert_array_equal(ctx.protected_values, [0, 1, 0])
