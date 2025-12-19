"""Tests for explainability module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explainability import Explainer
from src.model import CreditModel
from src.preprocessing import preprocess_data


class TestExplainer:
    """Tests for the Explainer class."""

    def test_setup_shap_explainer(self, preprocessed_data, test_config):
        """Test SHAP explainer initialization."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        assert explainer._shap_explainer is not None

    def test_calculate_shap_values(self, preprocessed_data, test_config):
        """Test SHAP value calculation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)
        shap_values = explainer.calculate_shap_values(X_test)

        # SHAP values should have same number of samples as X_test
        # (minus protected attribute column)
        assert len(shap_values) == len(X_test)

    def test_explain_individual(self, preprocessed_data, test_config):
        """Test individual explanation generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)
        explainer.calculate_shap_values(X_test)

        # Explain first sample
        first_index = X_test.index[0]
        explanation = explainer.explain_individual(
            X_test, first_index, y_pred[0]
        )

        assert "index" in explanation
        assert "prediction" in explanation
        assert "prediction_text" in explanation
        assert "feature_impacts" in explanation
        assert "top_positive_factors" in explanation
        assert "top_negative_factors" in explanation

    def test_generate_natural_language_explanation(self, preprocessed_data, test_config):
        """Test natural language explanation generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)
        explainer.calculate_shap_values(X_test)

        first_index = X_test.index[0]
        explanation = explainer.explain_individual(
            X_test, first_index, y_pred[0]
        )

        nl_explanation = explainer.generate_natural_language_explanation(explanation)

        assert isinstance(nl_explanation, str)
        assert "Explanation for Person" in nl_explanation
        assert "Model Decision" in nl_explanation
        assert "HELPED" in nl_explanation
        assert "HURT" in nl_explanation

    def test_explain_samples(self, preprocessed_data, test_config, test_data_dir):
        """Test batch explanation generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        # Create temp directory for explanations
        output_dir = test_data_dir / "test_explanations"
        output_dir.mkdir(exist_ok=True)

        try:
            explainer = Explainer(test_config)
            explainer.setup_shap_explainer(model)

            explanations = explainer.explain_samples(
                X_test, y_pred, num_samples=2, output_dir=output_dir
            )

            assert len(explanations) == 2
            assert all("natural_language" in e for e in explanations)
            assert all("output_path" in e for e in explanations)

            # Check files were created
            txt_files = list(output_dir.glob("*.txt"))
            assert len(txt_files) == 2

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_explain_samples_reproducible(self, preprocessed_data, test_config):
        """Test that explanations are reproducible with same seed."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        # Generate explanations twice with same config
        explainer1 = Explainer(test_config)
        explainer1.setup_shap_explainer(model)
        shap1 = explainer1.calculate_shap_values(X_test)

        explainer2 = Explainer(test_config)
        explainer2.setup_shap_explainer(model)
        shap2 = explainer2.calculate_shap_values(X_test)

        # SHAP values should be identical
        np.testing.assert_array_almost_equal(shap1, shap2)


class TestExplainerLIME:
    """Tests for LIME functionality."""

    def test_setup_lime_explainer(self, preprocessed_data, test_config):
        """Test LIME explainer initialization."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Drop protected attribute for LIME
        X_train_clean = X_train.drop(columns=["Proxy_Disadvantaged"])

        explainer = Explainer(test_config)
        explainer.setup_lime_explainer(X_train_clean)

        assert explainer._lime_explainer is not None

    def test_generate_lime_explanation(self, preprocessed_data, test_config):
        """Test LIME explanation generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        # Setup LIME
        X_train_clean = X_train.drop(columns=["Proxy_Disadvantaged"])
        explainer = Explainer(test_config)
        explainer.setup_lime_explainer(X_train_clean)

        # Generate LIME explanation
        first_index = X_test.index[0]
        lime_exp = explainer.generate_lime_explanation(
            model, X_test, first_index, num_features=5
        )

        assert "index" in lime_exp
        assert "feature_weights" in lime_exp
        assert len(lime_exp["feature_weights"]) <= 5


class TestExplainerPlots:
    """Tests for visualization generation."""

    def test_generate_global_importance_plot(self, preprocessed_data, test_config, test_data_dir):
        """Test global feature importance plot generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        output_path = test_data_dir / "test_importance.png"

        try:
            explainer = Explainer(test_config)
            explainer.setup_shap_explainer(model)
            result_path = explainer.generate_global_importance_plot(
                X_test, output_path=output_path
            )

            assert Path(result_path).exists()
            assert Path(result_path).suffix == ".png"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_generate_detailed_shap_plot(self, preprocessed_data, test_config, test_data_dir):
        """Test detailed SHAP summary plot generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        output_path = test_data_dir / "test_detailed.png"

        try:
            explainer = Explainer(test_config)
            explainer.setup_shap_explainer(model)
            result_path = explainer.generate_detailed_shap_plot(
                X_test, output_path=output_path
            )

            assert Path(result_path).exists()

        finally:
            if output_path.exists():
                output_path.unlink()


class TestNewExplainabilityFeatures:
    """Tests for new explainability features."""

    def test_generate_waterfall_plot(self, preprocessed_data, test_config, test_data_dir):
        """Test waterfall plot generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        output_path = test_data_dir / "test_waterfall.png"

        try:
            result_path = explainer.generate_waterfall_plot(
                X_test, index=0, output_path=output_path
            )

            assert Path(result_path).exists()
            assert Path(result_path).suffix == ".png"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_generate_force_plot(self, preprocessed_data, test_config, test_data_dir):
        """Test force plot generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        output_path = test_data_dir / "test_force.html"

        try:
            result_path = explainer.generate_force_plot(
                X_test, index=0, output_path=output_path
            )

            assert Path(result_path).exists()
            assert Path(result_path).suffix == ".html"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_compare_explanations(self, preprocessed_data, test_config):
        """Test comparison of multiple explanations."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        # Compare first 5 samples
        comparison = explainer.compare_explanations(
            X_test.iloc[:5], y_pred[:5], indices=list(range(5))
        )

        assert "individual_explanations" in comparison
        assert "common_patterns" in comparison
        assert len(comparison["individual_explanations"]) == 5

    def test_generate_interactive_report(self, preprocessed_data, test_config, test_data_dir):
        """Test interactive HTML report generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        output_path = test_data_dir / "test_interactive_report.html"

        try:
            result_path = explainer.generate_interactive_report(
                X_test.iloc[:5], y_pred[:5], output_path=output_path
            )

            assert Path(result_path).exists()
            assert Path(result_path).suffix == ".html"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_get_top_features_summary(self, preprocessed_data, test_config):
        """Test top features summary generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        summary = explainer.get_top_features_summary(X_test, n_features=5)

        assert "top_features" in summary
        assert len(summary["top_features"]) <= 5
        assert all("feature" in item for item in summary["top_features"])
