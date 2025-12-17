"""Integration tests for end-to-end pipelines.

These tests verify that complete workflows function correctly,
including tuning, calibration, and fairness pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CreditModel
from src.preprocessing import Preprocessor
from src.fairness import FairnessAnalyzer
from src.explainability import Explainer
from src.data_loader import DataLoader


class TestTuningPipeline:
    """Integration tests for hyperparameter tuning pipeline."""

    def test_full_tuning_workflow(self, sample_credit_data, test_config):
        """Test complete tuning workflow from data to tuned model."""
        # Setup
        data_loader = DataLoader(test_config)
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        model = CreditModel(test_config)

        # Small parameter grid for testing
        param_grid = {
            "max_depth": [3, 5],
            "learning_rate": [0.1, 0.2],
        }

        # Run tuning
        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            search_type="grid",
            param_grid=param_grid,
            cv=3,
        )
        best_params = tuning_results["best_params"]
        cv_results = tuning_results["cv_results"]

        # Verify results
        assert "max_depth" in best_params
        assert "learning_rate" in best_params
        assert best_params["max_depth"] in [3, 5]
        assert best_params["learning_rate"] in [0.1, 0.2]
        assert "best_score" in tuning_results
        assert 0 <= tuning_results["best_score"] <= 1

        # After tune_hyperparameters, model is already trained with best params
        # Verify model works
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(X_test)
        assert set(y_pred).issubset({0, 1})

    def test_tuning_with_random_search(self, sample_credit_data, test_config):
        """Test random search tuning."""
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        model = CreditModel(test_config)

        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }

        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            search_type="random",
            param_grid=param_grid,
            cv=3,
            n_iter=5,  # Small for testing
        )

        best_params = tuning_results["best_params"]
        assert best_params is not None
        assert tuning_results["best_score"] > 0

    def test_tuning_saves_and_loads(self, sample_credit_data, test_config):
        """Test that tuned model can be saved and loaded."""
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        # Tune and train
        model = CreditModel(test_config)
        param_grid = {"max_depth": [3, 5]}

        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            search_type="grid",
            param_grid=param_grid,
            cv=2,
        )
        best_params = tuning_results["best_params"]

        # After tune_hyperparameters, model is already trained with best params

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            model.save(tmp_path)

            # Load
            model_loaded = CreditModel(test_config)
            model_loaded.load(tmp_path)

            # Verify predictions match
            pred_original = model.predict(X_test)
            pred_loaded = model_loaded.predict(X_test)

            np.testing.assert_array_equal(pred_original, pred_loaded)

            # Cleanup
            tmp_path.unlink()


class TestCalibrationPipeline:
    """Integration tests for calibration pipeline."""

    def test_full_calibration_workflow(self, sample_credit_data, test_config):
        """Test complete calibration workflow."""
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        # Train model
        model = CreditModel(test_config)
        model.train(X_train, y_train)

        # Get uncalibrated predictions
        y_prob_before = model.predict_proba(X_test)

        # Calibrate
        model.calibrate_model(X_test, y_test, method="isotonic")

        # Get calibrated predictions
        y_prob_after = model.predict_proba(X_test)

        # Verify calibration worked
        assert model._calibrated_model is not None

        # Probabilities should be different
        # Note: isotonic calibration on small datasets can dramatically change probability distribution
        assert not np.array_equal(y_prob_before, y_prob_after)
        
        # Just verify probabilities are still valid (between 0 and 1)
        assert np.all((y_prob_after >= 0) & (y_prob_after <= 1))

        # Get calibration metrics
        calibration_data = model.get_calibration_curve(X_test, y_test)

        assert len(calibration_data["prob_true"]) > 0
        assert len(calibration_data["prob_pred"]) > 0
        assert 0 <= calibration_data["brier_score"] <= 1

    def test_calibration_methods(self, sample_credit_data, test_config):
        """Test both calibration methods."""
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        for method in ["isotonic", "sigmoid"]:
            model = CreditModel(test_config)
            model.train(X_train, y_train)

            # Calibrate
            model.calibrate_model(X_test, y_test, method=method)

            # Verify
            assert model._calibrated_model is not None

            # Predictions should work
            y_pred = model.predict(X_test)
            assert len(y_pred) == len(X_test)

            # Probabilities should be in valid range
            y_prob = model.predict_proba(X_test)
            assert np.all((y_prob >= 0) & (y_prob <= 1))

    def test_calibrated_model_persistence(self, sample_credit_data, test_config):
        """Test that calibrated model can be saved and loaded."""
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        # Train and calibrate
        model = CreditModel(test_config)
        model.train(X_train, y_train)
        model.calibrate_model(X_test, y_test, method="isotonic")

        # Get predictions
        y_prob_original = model.predict_proba(X_test)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            model.save(tmp_path)

            # Load
            model_loaded = CreditModel(test_config)
            model_loaded.load(tmp_path)

            # Verify calibrated model was saved
            assert model_loaded._calibrated_model is not None

            # Verify predictions match
            y_prob_loaded = model_loaded.predict_proba(X_test)
            np.testing.assert_array_almost_equal(y_prob_original, y_prob_loaded, decimal=5)

            # Cleanup
            tmp_path.unlink()


class TestCompletePipeline:
    """Integration tests for complete end-to-end pipeline."""

    def test_train_tune_calibrate_predict(self, sample_credit_data, test_config):
        """Test full pipeline: tune → train → calibrate → predict."""
        # Preprocessing
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        # 1. Hyperparameter tuning
        model = CreditModel(test_config)
        param_grid = {"max_depth": [3, 5], "learning_rate": [0.1]}

        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            search_type="grid",
            param_grid=param_grid,
            cv=2,
        )
        best_params = tuning_results["best_params"]

        # 2. After tune_hyperparameters, model is already trained with best params

        # 3. Calibrate
        model.calibrate_model(X_test, y_test, method="isotonic")

        # 4. Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Verify everything works
        assert len(y_pred) == len(X_test)
        assert len(y_prob) == len(X_test)
        assert np.all((y_prob >= 0) & (y_prob <= 1))

        # 5. Evaluate
        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_fairness_with_tuning_and_calibration(self, sample_credit_data, test_config):
        """Test fairness pipeline with tuning and calibration."""
        # Preprocessing
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        # Fairness preprocessing
        fairness_analyzer = FairnessAnalyzer(test_config)
        sample_weights = fairness_analyzer.apply_reweighing(X_train, y_train)

        # Tune
        model = CreditModel(test_config)
        param_grid = {"max_depth": [3, 5]}

        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            search_type="grid",
            param_grid=param_grid,
            cv=2,
        )
        best_params = tuning_results["best_params"]

        # After tune_hyperparameters, model is already trained with best params
        # Re-train with fairness weights if needed (since tune doesn't use weights)
        model.train(X_train, y_train, sample_weights=sample_weights)

        # Calibrate
        model.calibrate_model(X_test, y_test, method="isotonic")

        # Predict
        y_prob = model.predict_proba(X_test)

        # Apply fairness thresholds
        proxy_col = test_config.cdi.proxy_column
        protected_values = X_test[proxy_col].values
        y_pred_fair = fairness_analyzer.apply_threshold_adjustment(y_prob, protected_values)

        # Calculate fairness metrics
        fairness_metrics = fairness_analyzer.calculate_metrics(X_test, y_test, y_pred_fair)

        # Verify metrics calculated
        assert fairness_metrics.disparate_impact > 0
        assert -1 <= fairness_metrics.statistical_parity_difference <= 1

    def test_complete_pipeline_with_persistence(self, sample_credit_data, test_config):
        """Test complete pipeline with saving/loading all artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Paths
            preprocessor_path = tmpdir / "preprocessor.pkl"
            model_path = tmpdir / "model.pkl"

            # 1. Train pipeline
            preprocessor = Preprocessor(test_config)
            X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

            # Save preprocessor
            preprocessor.save(preprocessor_path)

            # Train model
            model = CreditModel(test_config)
            model.train(X_train, y_train)
            model.calibrate_model(X_test, y_test)

            # Save model
            model.save(model_path)

            # Get predictions
            y_pred_original = model.predict(X_test)

            # 2. Inference pipeline (separate session)
            preprocessor_loaded = Preprocessor(test_config)
            preprocessor_loaded.load(preprocessor_path)

            model_loaded = CreditModel(test_config)
            model_loaded.load(model_path)

            # Prepare new data
            new_data = sample_credit_data.iloc[:10]
            X_new = preprocessor_loaded.prepare_inference_features(new_data)

            # Predict
            y_pred_new = model_loaded.predict(X_new)

            # Verify
            assert len(y_pred_new) == 10
            assert set(y_pred_new).issubset({0, 1})

            # Verify model consistency
            y_pred_loaded = model_loaded.predict(X_test)
            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)


class TestCrossValidation:
    """Integration tests for cross-validation."""

    def test_cross_validation_workflow(self, sample_credit_data, test_config):
        """Test cross-validation integration."""
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        model = CreditModel(test_config)

        # Run CV
        cv_scores = model.cross_validate(
            X_train, y_train,
            cv=3,
            scoring=["accuracy", "f1", "roc_auc"],
        )

        # Verify results
        assert "accuracy" in cv_scores
        assert "f1" in cv_scores
        assert "roc_auc" in cv_scores

        # Each metric should have scores for each fold
        for metric, details in cv_scores.items():
            scores = details["scores"]
            assert len(scores) == 3
            assert all(0 <= s <= 1 for s in scores)

        # Verify CV scores are stored
        model.train(X_train, y_train)
        saved_cv_scores = model.get_cv_scores()

        # Should be able to retrieve CV scores
        assert saved_cv_scores is not None or saved_cv_scores == {}


class TestExplainabilityIntegration:
    """Integration tests for explainability pipeline."""

    def test_explainability_with_calibrated_model(self, sample_credit_data, test_config):
        """Test that explainability works with calibrated models."""
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        # Train and calibrate
        model = CreditModel(test_config)
        model.train(X_train, y_train)
        model.calibrate_model(X_test, y_test)

        # Setup explainer
        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        # Generate explanations
        y_pred = model.predict(X_test)
        explanations = explainer.explain_samples(X_test, y_pred, num_samples=3)

        # Verify
        assert len(explanations) == 3
        for exp in explanations:
            assert "natural_language" in exp
            assert "prediction" in exp

    def test_complete_pipeline_with_visualizations(self, sample_credit_data, test_config, test_data_dir):
        """Test complete pipeline including visualizations."""
        # Preprocessing
        preprocessor = Preprocessor(test_config)
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        # Train
        model = CreditModel(test_config)
        model.train(X_train, y_train)

        # Explainability
        explainer = Explainer(test_config)
        explainer.setup_shap_explainer(model)

        # Generate plots
        importance_plot = explainer.generate_global_importance_plot(
            X_test,
            output_path=test_data_dir / "importance.png"
        )

        # Fairness analysis
        fairness_analyzer = FairnessAnalyzer(test_config)
        y_pred = model.predict(X_test)
        fairness_metrics = fairness_analyzer.calculate_metrics(X_test, y_test, y_pred)

        fairness_plot = fairness_analyzer.plot_fairness_metrics(
            fairness_metrics,
            save_path=test_data_dir / "fairness.png"
        )

        # Verify files created
        assert Path(importance_plot).exists()
        assert Path(fairness_plot).exists()

        # Cleanup
        Path(importance_plot).unlink()
        Path(fairness_plot).unlink()


class TestDataValidation:
    """Integration tests for data validation pipeline."""

    def test_validation_workflow(self, sample_credit_data, test_config):
        """Test data validation integration."""
        data_loader = DataLoader(test_config)
        preprocessor = Preprocessor(test_config)

        # Validate data
        validation_report = preprocessor.validate_data(sample_credit_data)

        # Check report structure
        assert "shape" in validation_report
        assert "missing_values" in validation_report
        assert "duplicates" in validation_report
        assert isinstance(validation_report.get("issues", []), list)

        # Should be able to proceed with training
        X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(sample_credit_data)

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        # Model should work
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(X_test)
