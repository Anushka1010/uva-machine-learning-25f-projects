"""Tests for preprocessing module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import Preprocessor, preprocess_data
from config.config import Config, CDIConfig


class TestPreprocessor:
    """Tests for the Preprocessor class."""

    def test_convert_numeric_columns(self, sample_credit_data, test_config):
        """Test numeric column conversion."""
        preprocessor = Preprocessor(test_config)

        result = preprocessor.convert_numeric_columns(
            sample_credit_data, columns=["Income", "Debt"]
        )

        assert result["Income"].dtype in [np.float64, np.int64]
        assert result["Debt"].dtype in [np.float64, np.int64]

    def test_create_derived_features(self, sample_credit_data, test_config):
        """Test derived feature creation."""
        preprocessor = Preprocessor(test_config)

        # First convert to numeric
        df = preprocessor.convert_numeric_columns(sample_credit_data)
        result = preprocessor.create_derived_features(df)

        assert "Debt_to_Income" in result.columns
        assert "Loan_to_Income" in result.columns

        # Check calculation is correct
        expected_dti = df["Debt"] / df["Income"]
        np.testing.assert_array_almost_equal(
            result["Debt_to_Income"].values,
            expected_dti.values,
            decimal=5,
        )

    def test_calculate_cdi(self, sample_credit_data, test_config):
        """Test CDI calculation."""
        preprocessor = Preprocessor(test_config)

        result = preprocessor.calculate_cdi(sample_credit_data)

        assert "CDI" in result.columns
        assert "Proxy_Disadvantaged" in result.columns

        # CDI should be 0-4 based on default factors
        assert result["CDI"].min() >= 0
        assert result["CDI"].max() <= 4

        # Proxy_Disadvantaged should be binary
        assert set(result["Proxy_Disadvantaged"].unique()).issubset({0, 1})

    def test_calculate_cdi_custom_threshold(self, sample_credit_data, test_config):
        """Test CDI calculation with custom threshold."""
        preprocessor = Preprocessor(test_config)

        # Use threshold of 3 (stricter definition of disadvantaged)
        custom_cdi = CDIConfig(threshold=3)
        result = preprocessor.calculate_cdi(sample_credit_data, cdi_config=custom_cdi)

        # With higher threshold, fewer should be classified as disadvantaged
        high_threshold_count = result["Proxy_Disadvantaged"].sum()

        # Compare to default threshold of 2
        default_cdi = CDIConfig(threshold=2)
        result2 = preprocessor.calculate_cdi(sample_credit_data, cdi_config=default_cdi)
        low_threshold_count = result2["Proxy_Disadvantaged"].sum()

        assert high_threshold_count <= low_threshold_count

    def test_encode_categorical(self, sample_credit_data, test_config):
        """Test categorical encoding."""
        preprocessor = Preprocessor(test_config)

        result = preprocessor.encode_categorical(
            sample_credit_data, columns=["Gender", "Education"]
        )

        # Original columns should be gone
        assert "Gender" not in result.columns
        assert "Education" not in result.columns

        # Dummy columns should exist
        dummy_cols = [c for c in result.columns if c.startswith(("Gender_", "Education_"))]
        assert len(dummy_cols) > 0

    def test_prepare_features_and_target(self, sample_credit_data, test_config):
        """Test feature/target separation."""
        preprocessor = Preprocessor(test_config)

        # Preprocess first
        df = preprocessor.convert_numeric_columns(sample_credit_data)
        df = preprocessor.calculate_cdi(df)
        df = preprocessor.encode_categorical(df)

        X, y = preprocessor.prepare_features_and_target(df)

        assert "Creditworthiness" not in X.columns
        assert len(y) == len(X)
        assert y.dtype == float

    def test_split_data(self, sample_credit_data, test_config):
        """Test data splitting."""
        preprocessor = Preprocessor(test_config)

        # Preprocess
        df = preprocessor.convert_numeric_columns(sample_credit_data)
        df = preprocessor.calculate_cdi(df)
        df = preprocessor.encode_categorical(df)
        X, y = preprocessor.prepare_features_and_target(df)

        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        # Check sizes
        expected_test_size = int(len(X) * 0.2)
        assert len(X_test) == expected_test_size or len(X_test) == expected_test_size + 1
        assert len(X_train) == len(X) - len(X_test)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_split_data_is_reproducible(self, sample_credit_data, test_config):
        """Test that data splitting is reproducible with same seed."""
        preprocessor = Preprocessor(test_config)

        # Preprocess
        df = preprocessor.convert_numeric_columns(sample_credit_data)
        df = preprocessor.calculate_cdi(df)
        df = preprocessor.encode_categorical(df)
        X, y = preprocessor.prepare_features_and_target(df)

        # Split twice
        X_train1, X_test1, _, _ = preprocessor.split_data(X, y, random_state=42)
        X_train2, X_test2, _, _ = preprocessor.split_data(X, y, random_state=42)

        # Should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)

    def test_align_to_feature_space_restores_missing_columns(
        self, sample_credit_data, test_config
    ):
        """Test that align_to_feature_space reintroduces dropped columns."""
        preprocessor = Preprocessor(test_config)
        X_train, _, _, _, _ = preprocessor.fit_transform(sample_credit_data)

        proxy_col = test_config.cdi.proxy_column
        feature_only = X_train.drop(columns=[proxy_col])
        missing_feature = preprocessor.get_feature_names()[0]

        truncated = feature_only.drop(columns=[missing_feature])
        aligned = preprocessor.align_to_feature_space(truncated)

        assert missing_feature in aligned.columns
        assert aligned.shape[1] == len(preprocessor.get_feature_names())

    def test_prepare_inference_features_handles_unseen_categories(
        self, sample_credit_data, test_config
    ):
        """Test inference preprocessing aligns columns and retains proxy attribute."""
        preprocessor = Preprocessor(test_config)
        preprocessor.fit_transform(sample_credit_data)

        new_data = sample_credit_data.copy()
        new_data.loc[:, "Education"] = "Trade School"  # unseen category

        features = preprocessor.prepare_inference_features(new_data)

        assert features.shape[1] == len(preprocessor.get_feature_names()) + 1
        assert test_config.cdi.proxy_column in features.columns


class TestPreprocessDataConvenience:
    """Tests for the preprocess_data convenience function."""

    def test_preprocess_data_returns_all_outputs(self, sample_credit_data, test_config):
        """Test preprocess_data returns expected outputs."""
        X_train, X_test, y_train, y_test, processed_df = preprocess_data(
            sample_credit_data, test_config
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(processed_df, pd.DataFrame)

        # Proxy column should be in X for fairness analysis
        assert "Proxy_Disadvantaged" in X_train.columns
        assert "Proxy_Disadvantaged" in X_test.columns

    def test_preprocess_data_handles_missing_values(self, test_config):
        """Test preprocessing handles missing values gracefully."""
        # Create data with some missing values
        data = pd.DataFrame(
            {
                "Income": ["50000", "60000", None, "70000"],
                "Debt": ["10000", None, "15000", "20000"],
                "Loan_Amount": ["25000", "30000", "35000", "40000"],
                "Loan_Term": ["24", "36", "48", "60"],
                "Num_Credit_Cards": ["2", "3", "4", "5"],
                "Credit_Score": ["700", "720", "680", "750"],
                "Gender": ["Male", "Female", "Male", "Female"],
                "Education": ["High School", "Bachelor's", "Master's", "PhD"],
                "Payment_History": ["Good", "Excellent", "Fair", "Good"],
                "Employment_Status": ["Employed", "Employed", "Self-Employed", "Employed"],
                "Residence_Type": ["Owned", "Rented", "Owned", "Mortgage"],
                "Marital_Status": ["Single", "Married", "Single", "Married"],
                "Creditworthiness": ["1", "1", "0", "1"],
            }
        )

        # Should not raise an error
        X_train, X_test, y_train, y_test, processed_df = preprocess_data(
            data, test_config
        )

        # Results should be valid DataFrames (may have NaN values)
        assert isinstance(X_train, pd.DataFrame)
        assert len(X_train) + len(X_test) == len(data)


class TestNewPreprocessingFeatures:
    """Tests for new preprocessing features."""

    def test_validate_data(self, sample_credit_data, test_config):
        """Test data validation."""
        preprocessor = Preprocessor(test_config)
        validation_report = preprocessor.validate_data(sample_credit_data)

        assert "shape" in validation_report
        assert "missing_values" in validation_report
        assert "duplicates" in validation_report
        assert isinstance(validation_report.get("issues", []), list)

    def test_validate_data_detects_issues(self, test_config):
        """Test that validation detects data quality issues."""
        # Create data with issues
        data = pd.DataFrame({
            "Income": ["50000", "60000", None, "70000"],  # Missing value
            "Debt": ["10000", "15000", "15000", "20000"],
            "Loan_Amount": ["25000", "30000", "35000", "40000"],
            "Loan_Term": ["24", "36", "48", "60"],
            "Num_Credit_Cards": ["2", "3", "4", "5"],
            "Credit_Score": ["700", "720", "680", "750"],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Education": ["High School", "Bachelor's", "Master's", "PhD"],
            "Payment_History": ["Good", "Excellent", "Fair", "Good"],
            "Employment_Status": ["Employed", "Employed", "Self-Employed", "Employed"],
            "Residence_Type": ["Owned", "Rented", "Owned", "Mortgage"],
            "Marital_Status": ["Single", "Married", "Single", "Married"],
            "Creditworthiness": ["1", "1", "0", "1"],
        })

        preprocessor = Preprocessor(test_config)
        validation_report = preprocessor.validate_data(data)

        # Should detect missing values
        assert validation_report["missing_values"] > 0

    def test_apply_scaling(self, sample_credit_data, test_config):
        """Test feature scaling."""
        preprocessor = Preprocessor(test_config)

        # Convert to numeric first
        df = preprocessor.convert_numeric_columns(sample_credit_data)

        # Apply scaling
        scaled_df = preprocessor.apply_scaling(df, columns=["Income", "Debt"])

        # Scaled values should have mean ≈ 0 and std ≈ 1
        assert abs(scaled_df["Income"].mean()) < 1.0
        assert abs(scaled_df["Income"].std() - 1.0) < 0.5

    def test_preprocessor_save_and_load(self, sample_credit_data, test_config):
        """Test preprocessor persistence."""
        import tempfile

        preprocessor = Preprocessor(test_config)
        preprocessor.fit_transform(sample_credit_data)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            preprocessor.save(tmp_path)

            # Load
            preprocessor2 = Preprocessor(test_config)
            preprocessor2.load(tmp_path)

            # Check feature names match
            assert preprocessor.get_feature_names() == preprocessor2.get_feature_names()

            # Cleanup
            tmp_path.unlink()

    def test_handle_unseen_categories(self, sample_credit_data, test_config):
        """Test handling of unseen categories during inference."""
        preprocessor = Preprocessor(test_config)
        preprocessor.fit_transform(sample_credit_data)

        # Create data with unseen category
        new_data = sample_credit_data.copy()
        new_data.loc[0, "Education"] = "Trade School"  # Unseen category

        # Should handle gracefully
        features = preprocessor.handle_unseen_categories(new_data)
        assert features is not None
        assert len(features) == len(new_data)

    def test_prepare_inference_features(self, sample_credit_data, test_config):
        """Test full inference pipeline."""
        preprocessor = Preprocessor(test_config)
        preprocessor.fit_transform(sample_credit_data)

        # Prepare new data for inference
        new_data = sample_credit_data.iloc[:5].copy()
        features = preprocessor.prepare_inference_features(new_data)

        # Should have correct shape
        assert features.shape[0] == 5
        # Should have all features plus proxy column
