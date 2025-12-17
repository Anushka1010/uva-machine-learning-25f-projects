"""Tests for model module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CreditModel
from src.preprocessing import preprocess_data


class TestCreditModel:
    """Tests for the CreditModel class."""

    def test_model_train(self, preprocessed_data, test_config):
        """Test model training."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        assert model.get_underlying_model() is not None
        assert model.get_feature_names() is not None
        assert len(model.get_feature_names()) > 0

    def test_model_train_with_sample_weights(self, preprocessed_data, test_config):
        """Test model training with sample weights."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Create sample weights
        sample_weights = np.ones(len(X_train))
        sample_weights[:len(X_train)//2] = 0.5  # Half weight for first half

        model = CreditModel(test_config)
        model.train(X_train, y_train, sample_weights=sample_weights)

        metadata = model.get_training_metadata()
        assert metadata["used_sample_weights"] is True

    def test_model_predict(self, preprocessed_data, test_config):
        """Test model prediction."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_model_predict_proba(self, preprocessed_data, test_config):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        probabilities = model.predict_proba(X_test)

        assert len(probabilities) == len(X_test)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_model_evaluate(self, preprocessed_data, test_config):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_model_feature_importance(self, preprocessed_data, test_config):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        importance_df = model.get_feature_importance()

        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) > 0

    def test_cross_validate(self, preprocessed_data, test_config):
        """Test cross-validation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        cv_scores = model.cross_validate(
            X_train, y_train,
            cv=3,
            scoring=["accuracy", "f1", "roc_auc"],
        )

        assert "accuracy" in cv_scores
        assert "f1" in cv_scores
        assert "roc_auc" in cv_scores
        accuracy_scores = cv_scores["accuracy"]["scores"]
        assert len(accuracy_scores) == 3
        assert all(0 <= score <= 1 for score in accuracy_scores)

    def test_tune_hyperparameters_grid(self, preprocessed_data, test_config):
        """Test grid search hyperparameter tuning."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Small parameter grid for faster testing
        param_grid = {
            "max_depth": [3, 5],
            "learning_rate": [0.1, 0.2],
        }

        model = CreditModel(test_config)
        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            search_type="grid",
            param_grid=param_grid,
            cv=3,
        )

        best_params = tuning_results["best_params"]
        cv_results = tuning_results["cv_results"]

        assert "max_depth" in best_params
        assert "learning_rate" in best_params
        assert best_params["max_depth"] in [3, 5]
        assert best_params["learning_rate"] in [0.1, 0.2]
        assert "best_score" in tuning_results

    def test_tune_hyperparameters_random(self, preprocessed_data, test_config):
        """Test random search hyperparameter tuning."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.1, 0.2, 0.3],
        }

        model = CreditModel(test_config)
        tuning_results = model.tune_hyperparameters(
            X_train,
            y_train,
            search_type="random",
            param_grid=param_grid,
            cv=3,
            n_iter=5,
        )

        best_params = tuning_results["best_params"]
        cv_results = tuning_results["cv_results"]

        assert "max_depth" in best_params
        assert "learning_rate" in best_params
        assert "best_score" in tuning_results

    def test_calibrate_model_isotonic(self, preprocessed_data, test_config):
        """Test isotonic calibration."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        # Calibrate
        model.calibrate_model(X_test, y_test, method="isotonic")

        # Check calibrated model exists
        assert model._calibrated_model is not None

        # Predictions should still work
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_calibrate_model_sigmoid(self, preprocessed_data, test_config):
        """Test sigmoid calibration."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        model.calibrate_model(X_test, y_test, method="sigmoid")

        assert model._calibrated_model is not None

    def test_get_calibration_curve(self, preprocessed_data, test_config):
        """Test calibration curve generation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)
        model.calibrate_model(X_test, y_test, method="isotonic")

        calibration_data = model.get_calibration_curve(X_test, y_test)

        assert len(calibration_data["prob_true"]) > 0
        assert len(calibration_data["prob_pred"]) > 0
        assert 0 <= calibration_data["brier_score"] <= 1

    def test_get_all_feature_importances(self, preprocessed_data, test_config):
        """Test getting all feature importance types."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        all_importances = model.get_all_feature_importances()

        assert "weight" in all_importances
        assert "gain" in all_importances
        assert "cover" in all_importances

        # Each type should have feature names as keys
        assert len(all_importances["weight"]) > 0
    def test_model_save_and_load(self, preprocessed_data, test_config):
        """Test model persistence."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Train and save
        model = CreditModel(test_config)
        model.train(X_train, y_train)
        original_predictions = model.predict(X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            model.save(model_path)

            # Load and predict
            loaded_model = CreditModel(test_config)
            loaded_model.load(model_path)
            loaded_predictions = loaded_model.predict(X_test)

            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)

        finally:
            Path(model_path).unlink()

    def test_model_predict_before_train_raises(self, preprocessed_data, test_config):
        """Test that predicting before training raises an error."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)

        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X_test)

    def test_model_drops_protected_attribute(self, preprocessed_data, test_config):
        """Test that model drops protected attribute during training."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        model = CreditModel(test_config)
        model.train(X_train, y_train)

        feature_names = model.get_feature_names()

        # Protected attribute should not be in feature names
        assert "Proxy_Disadvantaged" not in feature_names


class TestModelReproducibility:
    """Tests for model reproducibility."""

    def test_same_seed_same_results(self, preprocessed_data, test_config):
        """Test that same seed produces same results."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data

        # Train two models with same seed
        model1 = CreditModel(test_config)
        model1.train(X_train, y_train)
        pred1 = model1.predict(X_test)

        model2 = CreditModel(test_config)
        model2.train(X_train, y_train)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)
