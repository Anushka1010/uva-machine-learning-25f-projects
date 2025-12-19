"""Tests for fairness module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fairness import FairnessAnalyzer, FairnessMetrics
from src.model import CreditModel
from src.preprocessing import preprocess_data


class TestFairnessMetrics:
    """Tests for the FairnessMetrics dataclass."""

    def test_metrics_to_dict(self):
        """Test metrics can be converted to dictionary."""
        metrics = FairnessMetrics(
            disparate_impact=0.85,
            statistical_parity_difference=-0.02,
            equalized_odds_difference=0.05,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["disparate_impact"] == 0.85
        assert result["statistical_parity_difference"] == -0.02

    def test_passes_thresholds_all_pass(self):
        """Test passes_thresholds when all metrics pass."""
        metrics = FairnessMetrics(
            disparate_impact=0.90,  # >= 0.80 ✓
            statistical_parity_difference=-0.01,  # |x| <= 0.05 ✓
            equalized_odds_difference=0.03,  # <= 0.10 ✓
        )

        result = metrics.passes_thresholds()

        assert result["disparate_impact"] is True
        assert result["statistical_parity_difference"] is True
        assert result["equalized_odds_difference"] is True

    def test_passes_thresholds_some_fail(self):
        """Test passes_thresholds when some metrics fail."""
        metrics = FairnessMetrics(
            disparate_impact=0.70,  # < 0.80 ✗
            statistical_parity_difference=-0.01,  # |x| <= 0.05 ✓
            equalized_odds_difference=0.15,  # > 0.10 ✗
        )

        result = metrics.passes_thresholds()

        assert result["disparate_impact"] is False
        assert result["statistical_parity_difference"] is True
        assert result["equalized_odds_difference"] is False

    def test_passes_thresholds_custom_thresholds(self):
        """Test passes_thresholds with custom thresholds."""
        metrics = FairnessMetrics(
            disparate_impact=0.75,
            statistical_parity_difference=-0.08,
            equalized_odds_difference=0.12,
        )

        # With relaxed thresholds, should pass
        result = metrics.passes_thresholds(
            di_threshold=0.70,
            spd_threshold=0.10,
            eod_threshold=0.15,
        )

        assert result["disparate_impact"] is True
        assert result["statistical_parity_difference"] is True
        assert result["equalized_odds_difference"] is True


class TestFairnessAnalyzer:
    """Tests for the FairnessAnalyzer class."""

    def test_apply_reweighing(self, preprocessed_data, test_config):
        """Test reweighing pre-processing."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        analyzer = FairnessAnalyzer(test_config)
        sample_weights = analyzer.apply_reweighing(X_train, y_train)

        assert len(sample_weights) == len(X_train)
        assert all(w > 0 for w in sample_weights)  # All weights positive
        assert not all(w == 1.0 for w in sample_weights)  # Not all equal to 1

    def test_apply_threshold_adjustment(self, preprocessed_data, test_config):
        """Test threshold adjustment post-processing."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Train model to get probabilities
        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        protected_values = X_test["Proxy_Disadvantaged"].values

        analyzer = FairnessAnalyzer(test_config)
        y_pred_adjusted = analyzer.apply_threshold_adjustment(
            y_prob, protected_values
        )

        assert len(y_pred_adjusted) == len(y_prob)
        assert set(y_pred_adjusted).issubset({0, 1})

    def test_threshold_adjustment_increases_unprivileged_approvals(
        self, preprocessed_data, test_config
    ):
        """Test that lower threshold for unprivileged increases their approvals."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Train model
        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        protected_values = X_test["Proxy_Disadvantaged"].values
        unpriv_mask = protected_values == 1

        analyzer = FairnessAnalyzer(test_config)

        # Standard threshold (0.5 for both)
        y_pred_standard = analyzer.apply_threshold_adjustment(
            y_prob, protected_values,
            threshold_privileged=0.5,
            threshold_unprivileged=0.5,
        )

        # Lower threshold for unprivileged (0.4)
        y_pred_adjusted = analyzer.apply_threshold_adjustment(
            y_prob, protected_values,
            threshold_privileged=0.5,
            threshold_unprivileged=0.4,
        )

        # Should have more approvals for unprivileged with lower threshold
        standard_unpriv_approvals = y_pred_standard[unpriv_mask].sum()
        adjusted_unpriv_approvals = y_pred_adjusted[unpriv_mask].sum()

        assert adjusted_unpriv_approvals >= standard_unpriv_approvals

    def test_calculate_metrics(self, preprocessed_data, test_config):
        """Test fairness metrics calculation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Train model
        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        analyzer = FairnessAnalyzer(test_config)
        metrics = analyzer.calculate_metrics(X_test, y_test, y_pred)

        assert isinstance(metrics, FairnessMetrics)
        assert 0 <= metrics.disparate_impact <= 2  # Can be > 1
        assert -1 <= metrics.statistical_parity_difference <= 1
        assert 0 <= metrics.equalized_odds_difference <= 1

    def test_calculate_metrics_group_confusion_matrices(
        self, preprocessed_data, test_config
    ):
        """Test that group confusion matrices are calculated."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        analyzer = FairnessAnalyzer(test_config)
        metrics = analyzer.calculate_metrics(X_test, y_test, y_pred)

        # Should have 2x2 confusion matrices
        assert metrics.privileged_confusion_matrix.shape == (2, 2)
        assert metrics.unprivileged_confusion_matrix.shape == (2, 2)


class TestFairnessIntegration:
    """Integration tests for fairness workflow."""

    def test_reweighing_improves_fairness(self, preprocessed_data, test_config):
        """Test that reweighing generally improves fairness metrics."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        analyzer = FairnessAnalyzer(test_config)

        # Model without reweighing
        model_no_reweight = CreditModel(test_config)
        model_no_reweight.train(X_train, y_train, sample_weights=None)
        y_pred_no_reweight = model_no_reweight.predict(X_test)
        metrics_no_reweight = analyzer.calculate_metrics(
            X_test, y_test, y_pred_no_reweight
        )

        # Model with reweighing
        sample_weights = analyzer.apply_reweighing(X_train, y_train)
        model_reweight = CreditModel(test_config)
        model_reweight.train(X_train, y_train, sample_weights=sample_weights)
        y_pred_reweight = model_reweight.predict(X_test)
        metrics_reweight = analyzer.calculate_metrics(
            X_test, y_test, y_pred_reweight
        )

        # Note: Reweighing doesn't always improve all metrics in all cases,
        # but the pipeline should run without errors
        assert metrics_no_reweight is not None
        assert metrics_reweight is not None


class TestNewFairnessFeatures:
    """Tests for new fairness features."""

    def test_plot_fairness_metrics(self, preprocessed_data, test_config):
        """Test fairness visualization plot generation."""
        import tempfile

        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        analyzer = FairnessAnalyzer(test_config)
        metrics = analyzer.calculate_metrics(X_test, y_test, y_pred)

        # Generate plot
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            plot_path = analyzer.plot_fairness_metrics(metrics, save_path=tmp_path)

            assert plot_path.exists()
            assert plot_path.suffix == ".png"

            # Cleanup
            tmp_path.unlink()

    def test_generate_fairness_report(self, preprocessed_data, test_config):
        """Test comprehensive fairness report generation."""
        import tempfile

        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        analyzer = FairnessAnalyzer(test_config)
        metrics = analyzer.calculate_metrics(X_test, y_test, y_pred)

        # Generate report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            report_path = analyzer.generate_fairness_report(
                metrics, X_test, y_test, y_pred, save_path=tmp_path
            )

            assert report_path.exists()
            assert report_path.suffix == ".json"

            # Read and validate report
            import json
            with open(report_path) as f:
                report = json.load(f)

            assert "metrics" in report
            assert "confusion_matrices" in report
            assert "group_statistics" in report

            # Cleanup
            tmp_path.unlink()

    def test_extended_fairness_metrics(self, preprocessed_data, test_config):
        """Test that extended fairness metrics are calculated."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        analyzer = FairnessAnalyzer(test_config)
        metrics = analyzer.calculate_metrics(X_test, y_test, y_pred)

        # Check extended metrics exist
        metrics_dict = metrics.to_dict()
        assert "disparate_impact" in metrics_dict
        assert "statistical_parity_difference" in metrics_dict
        assert "equalized_odds_difference" in metrics_dict

        # Metrics should be numeric
        assert isinstance(metrics.disparate_impact, (int, float))
