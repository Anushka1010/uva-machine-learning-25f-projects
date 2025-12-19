"""Tests for validators module.

These tests ensure our validation utilities correctly catch errors
and provide helpful error messages.
"""

import pytest
import numpy as np
import pandas as pd

from src.validators import (
    ValidationError,
    validate_dataframe_not_empty,
    validate_required_columns,
    validate_no_nan_in_columns,
    validate_predictions_shape,
    validate_probability_range,
    validate_binary_labels,
    require_non_empty_dataframe,
)


class TestValidateDataframeNotEmpty:
    """Tests for validate_dataframe_not_empty."""

    def test_passes_for_non_empty_df(self):
        """Should pass for DataFrame with data."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Should not raise
        validate_dataframe_not_empty(df)

    def test_raises_for_empty_df(self):
        """Should raise ValidationError for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="is empty"):
            validate_dataframe_not_empty(df)

    def test_includes_name_in_error(self):
        """Should include the name in error message."""
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="MyData is empty"):
            validate_dataframe_not_empty(df, "MyData")


class TestValidateRequiredColumns:
    """Tests for validate_required_columns."""

    def test_passes_when_all_columns_present(self):
        """Should pass when all required columns exist."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        # Should not raise
        validate_required_columns(df, ["a", "b"])

    def test_raises_when_columns_missing(self):
        """Should raise when required columns are missing."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValidationError, match="missing required columns"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_lists_missing_columns(self):
        """Should list the specific missing columns."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValidationError) as exc_info:
            validate_required_columns(df, ["a", "missing1", "missing2"])
        assert "missing1" in str(exc_info.value)
        assert "missing2" in str(exc_info.value)


class TestValidateNoNanInColumns:
    """Tests for validate_no_nan_in_columns."""

    def test_passes_for_clean_data(self):
        """Should pass when no NaN values present."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Should not raise
        validate_no_nan_in_columns(df)

    def test_raises_when_nan_present(self):
        """Should raise when NaN values found."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        with pytest.raises(ValidationError, match="contains NaN"):
            validate_no_nan_in_columns(df)

    def test_only_checks_specified_columns(self):
        """Should only check the specified columns."""
        df = pd.DataFrame({"a": [1, np.nan], "b": [4, 5]})
        # Should not raise because we only check column 'b'
        validate_no_nan_in_columns(df, columns=["b"])


class TestValidatePredictionsShape:
    """Tests for validate_predictions_shape."""

    def test_passes_for_matching_shapes(self):
        """Should pass when shapes match."""
        y_pred = np.array([0, 1, 0])
        y_true = np.array([0, 0, 1])
        # Should not raise
        validate_predictions_shape(y_pred, y_true)

    def test_raises_for_mismatched_shapes(self):
        """Should raise when shapes don't match."""
        y_pred = np.array([0, 1])
        y_true = np.array([0, 0, 1])
        with pytest.raises(ValidationError, match="length mismatch"):
            validate_predictions_shape(y_pred, y_true)


class TestValidateProbabilityRange:
    """Tests for validate_probability_range."""

    def test_passes_for_valid_range(self):
        """Should pass for values in [0, 1]."""
        y_prob = np.array([0.0, 0.5, 1.0])
        # Should not raise
        validate_probability_range(y_prob)

    def test_raises_for_values_below_zero(self):
        """Should raise for negative values."""
        y_prob = np.array([-0.1, 0.5, 0.9])
        with pytest.raises(ValidationError, match="outside \\[0, 1\\]"):
            validate_probability_range(y_prob)

    def test_raises_for_values_above_one(self):
        """Should raise for values > 1."""
        y_prob = np.array([0.1, 0.5, 1.1])
        with pytest.raises(ValidationError, match="outside \\[0, 1\\]"):
            validate_probability_range(y_prob)


class TestValidateBinaryLabels:
    """Tests for validate_binary_labels."""

    def test_passes_for_binary_values(self):
        """Should pass for 0 and 1 values."""
        y = np.array([0, 1, 0, 1, 1])
        # Should not raise
        validate_binary_labels(y)

    def test_passes_for_float_binary_values(self):
        """Should pass for 0.0 and 1.0 values."""
        y = np.array([0.0, 1.0, 0.0])
        # Should not raise
        validate_binary_labels(y)

    def test_raises_for_non_binary_values(self):
        """Should raise for values other than 0 and 1."""
        y = np.array([0, 1, 2])
        with pytest.raises(ValidationError, match="non-binary values"):
            validate_binary_labels(y)

    def test_works_with_pandas_series(self):
        """Should work with pandas Series."""
        y = pd.Series([0, 1, 0])
        # Should not raise
        validate_binary_labels(y)


class TestRequireNonEmptyDecorator:
    """Tests for require_non_empty_dataframe decorator."""

    def test_passes_for_non_empty_df(self):
        """Should allow function to execute with non-empty df."""
        @require_non_empty_dataframe("df")
        def process_data(df):
            return len(df)

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = process_data(df)
        assert result == 3

    def test_raises_for_empty_df(self):
        """Should raise before function executes with empty df."""
        @require_non_empty_dataframe("df")
        def process_data(df):
            return len(df)

        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="is empty"):
            process_data(df)
