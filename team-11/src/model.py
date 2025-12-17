"""Model training and prediction module.

This module handles model training, prediction, and model persistence.
Supports XGBoost with fairness-aware training via sample weights.
Includes cross-validation, hyperparameter tuning, and calibration.
"""

import logging
from pathlib import Path
from typing import Any
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from config.config import Config, ModelConfig, get_config
from src.utils import get_logger, calculate_metrics_summary
from src.protected_attribute import (
    prepare_features_without_protected,
    resolve_protected_attribute,
)


class CreditModel:
    """Handles model training, prediction, and persistence.

    This class wraps XGBoost (and potentially other models) with
    support for fairness-aware training via sample weights.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the CreditModel.

        Args:
            config: Configuration object. Uses global config if None.
        """
        self.config = config or get_config()
        self.logger = get_logger("model")
        self._model: xgb.XGBClassifier | None = None
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._training_metadata: dict[str, Any] = {}
        self._cv_scores: dict[str, list[float]] = {}
        self._version: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._calibration_blend: float = 1.0

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: np.ndarray | None = None,
        protected_attribute: str | None = None,
        **kwargs: Any,
    ) -> "CreditModel":
        """Train the credit prediction model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            sample_weights: Optional sample weights for fairness-aware training.
            protected_attribute: Column name to drop before training.
            **kwargs: Additional arguments passed to XGBClassifier.

        Returns:
            Self for method chaining.
        """
        model_config = self.config.model

        # Use DRY utility for protected attribute handling
        ctx = prepare_features_without_protected(
            X_train, protected_attribute, self.config
        )
        training_features = ctx.filtered_df
        
        if len(ctx.protected_values) > 0:
            self.logger.info(
                f"Dropped protected attribute '{ctx.protected_column}' from training"
            )

        # Merge config params with any overrides
        xgb_params = {**model_config.xgb_params, **kwargs}

        self.logger.info("Training XGBoost classifier...")
        self.logger.debug(f"XGBoost params: {xgb_params}")

        self._model = xgb.XGBClassifier(**xgb_params)
        self._calibrated_model = None
        self._calibration_blend = 1.0

        self._model.fit(
            training_features,
            y_train,
            sample_weight=sample_weights,
        )

        # Store training metadata
        self._training_metadata = {
            "num_samples": len(X_train),
            "num_features": training_features.shape[1],
            "feature_names": list(training_features.columns),
            "used_sample_weights": sample_weights is not None,
            "xgb_params": xgb_params,
        }

        self.logger.info(
            f"Model trained on {len(X_train)} samples, "
            f"{training_features.shape[1]} features"
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
        protected_attribute: str | None = None,
    ) -> np.ndarray:
        """Generate binary predictions.

        Args:
            X: Features for prediction.
            protected_attribute: Column name to drop before prediction.

        Returns:
            Array of binary predictions.
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        prediction_features = self._prepare_features_for_inference(
            X, protected_attribute
        )
        return self._model.predict(prediction_features)

    def predict_proba(
        self,
        X: pd.DataFrame,
        protected_attribute: str | None = None,
        use_calibrated: bool = True,
    ) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X: Features for prediction.
            protected_attribute: Column name to drop before prediction.
            use_calibrated: Use calibrated model if available.

        Returns:
            Array of probability predictions (for positive class).
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        prediction_features = self._prepare_features_for_inference(
            X, protected_attribute
        )
        # Use calibrated model if available and requested
        if use_calibrated and self._calibrated_model is not None:
            calibrated_probs = self._calibrated_model.predict_proba(prediction_features)[:, 1]
            if self._calibration_blend >= 1.0:
                return calibrated_probs
            base_probs = self._model.predict_proba(prediction_features)[:, 1]
            return (
                self._calibration_blend * calibrated_probs
                + (1.0 - self._calibration_blend) * base_probs
            )

        return self._model.predict_proba(prediction_features)[:, 1]

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        protected_attribute: str | None = None,
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            X_test: Test features.
            y_test: True labels.
            protected_attribute: Column name to drop before evaluation.

        Returns:
            Dictionary of metric names to values.
        """
        y_pred = self.predict(X_test, protected_attribute)
        y_prob = self.predict_proba(X_test, protected_attribute)

        metrics = calculate_metrics_summary(y_test.values, y_pred, y_prob)

        self.logger.info(f"Evaluation metrics:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")

        return metrics

    def get_feature_importance(
        self,
        importance_type: str = "weight",
    ) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover').

        Returns:
            DataFrame with feature names and importance scores.
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self._model.get_booster().get_score(importance_type=importance_type)

        df = pd.DataFrame(
            [
                {"feature": k, "importance": v}
                for k, v in importance.items()
            ]
        )

        if len(df) > 0:
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def get_all_feature_importances(self) -> dict[str, pd.DataFrame]:
        """Get all types of feature importance scores.

        Returns:
            Dictionary with importance type as key and DataFrame as value.
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance_types = ["weight", "gain", "cover"]
        importances = {}

        for imp_type in importance_types:
            try:
                importances[imp_type] = self.get_feature_importance(imp_type)
            except Exception as e:
                self.logger.warning(f"Could not get {imp_type} importance: {e}")

        return importances

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str | list[str] = "roc_auc",
        protected_attribute: str | None = None,
    ) -> dict[str, Any]:
        """Perform cross-validation.

        Args:
            X: Features.
            y: Target labels.
            cv: Number of folds.
            scoring: Scoring metric(s) to use.
            protected_attribute: Column to drop before training.

        Returns:
            Dictionary with cross-validation results.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        cv_features = X.copy()
        if protected_attribute in cv_features.columns:
            cv_features = cv_features.drop(columns=[protected_attribute])

        # Create model for CV
        model_config = self.config.model
        xgb_params = model_config.xgb_params
        model = xgb.XGBClassifier(**xgb_params)

        # Perform cross-validation
        if isinstance(scoring, str):
            scoring = [scoring]

        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(
                model,
                cv_features,
                y,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=model_config.random_state),
                scoring=metric,
                n_jobs=-1,
            )
            cv_results[metric] = {
                "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }
            self._cv_scores[metric] = scores.tolist()

            self.logger.info(
                f"CV {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})"
            )

        return cv_results

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: dict[str, list[Any]] | None = None,
        search_type: str = "grid",
        cv: int = 5,
        n_iter: int = 20,
        protected_attribute: str | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Tune hyperparameters using grid or random search.

        Args:
            X_train: Training features.
            y_train: Training labels.
            param_grid: Parameter grid/distributions to search.
            search_type: 'grid' or 'random' search.
            cv: Number of cross-validation folds.
            n_iter: Number of iterations for random search.
            protected_attribute: Column to drop before training.
            sample_weights: Optional sample weights for fairness-aware tuning.

        Returns:
            Dictionary with best parameters and scores.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        training_features = X_train.copy()
        if protected_attribute in training_features.columns:
            training_features = training_features.drop(columns=[protected_attribute])

        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "n_estimators": [50, 100, 200],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

        model_config = self.config.model
        base_model = xgb.XGBClassifier(
            random_state=model_config.random_state,
            eval_metric="logloss",
        )

        self.logger.info(f"Starting {search_type} search with {cv}-fold CV...")

        if search_type == "grid":
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
                random_state=model_config.random_state,
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        fit_kwargs: dict[str, Any] = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights

        search.fit(training_features, y_train, **fit_kwargs)

        results = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_results": {
                "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": search.cv_results_["std_test_score"].tolist(),
                "params": search.cv_results_["params"],
            },
        }

        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")

        # Update model with best parameters
        self._model = search.best_estimator_

        return results

    def calibrate_model(
        self,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        method: str = "isotonic",
        protected_attribute: str | None = None,
        blend: float = 0.5,
    ) -> "CreditModel":
        """Calibrate model probabilities.

        Args:
            X_calib: Calibration features.
            y_calib: Calibration labels.
            method: Calibration method ('isotonic' or 'sigmoid').
            protected_attribute: Column to drop.
            blend: Weight given to calibrated probabilities (0-1).

        Returns:
            Self for method chaining.
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        calib_features = self._prepare_features_for_inference(
            X_calib, protected_attribute
        )

        self.logger.info(f"Calibrating model using {method} method...")

        self._calibrated_model = CalibratedClassifierCV(
            self._model,
            method=method,
            cv="prefit",
        )

        self._calibrated_model.fit(calib_features, y_calib)
        self._calibration_blend = max(0.0, min(1.0, blend))

        self.logger.info("Model calibration completed")

        return self

    def get_calibration_curve(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_bins: int = 10,
        protected_attribute: str | None = None,
    ) -> dict[str, Any]:
        """Calculate calibration curve data.

        Args:
            X_test: Test features.
            y_test: Test labels.
            n_bins: Number of bins for calibration curve.
            protected_attribute: Column to drop.

        Returns:
            Dictionary with calibration data.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        y_prob = self.predict_proba(X_test, protected_attribute)

        prob_true, prob_pred = calibration_curve(
            y_test,
            y_prob,
            n_bins=n_bins,
            strategy="uniform",
        )

        brier_score = brier_score_loss(y_test, y_prob)

        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "brier_score": float(brier_score),
        }

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": self._model,
            "calibrated_model": self._calibrated_model,
            "calibration_blend": self._calibration_blend,
            "metadata": self._training_metadata,
            "cv_scores": self._cv_scores,
            "version": self._version,
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        self.logger.info(f"Model v{self._version} saved to {path}")

    def load(self, path: str | Path) -> "CreditModel":
        """Load a model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Self for method chaining.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            save_data = pickle.load(f)

        self._model = save_data["model"]
        self._calibrated_model = save_data.get("calibrated_model")
        self._calibration_blend = save_data.get("calibration_blend", 1.0)
        self._training_metadata = save_data.get("metadata", {})
        self._cv_scores = save_data.get("cv_scores", {})
        self._version = save_data.get("version", "unknown")

        self.logger.info(f"Model v{self._version} loaded from {path}")

        return self

    def get_underlying_model(self) -> xgb.XGBClassifier | None:
        """Get the underlying XGBoost model."""
        return self._model

    def get_training_metadata(self) -> dict[str, Any]:
        """Get metadata from training."""
        return self._training_metadata

    def get_feature_names(self) -> list[str]:
        """Get feature names used during training."""
        return self._training_metadata.get("feature_names", [])

    def _prepare_features_for_inference(
        self,
        X: pd.DataFrame,
        protected_attribute: str | None = None,
    ) -> pd.DataFrame:
        """Drop protected attribute and align columns with training features."""
        ctx = prepare_features_without_protected(X, protected_attribute, self.config)
        features = ctx.filtered_df.copy()

        trained_features = self.get_feature_names()
        if trained_features:
            missing = [col for col in trained_features if col not in features.columns]
            extra = [col for col in features.columns if col not in trained_features]

            if missing:
                self.logger.warning(
                    f"Input data missing {len(missing)} trained features: {missing}"
                )
                for col in missing:
                    features[col] = 0.0

            if extra:
                self.logger.warning(
                    f"Dropping {len(extra)} unexpected feature columns: {extra}"
                )
                features = features.drop(columns=extra, errors="ignore")

            features = features[trained_features]

        return features.apply(pd.to_numeric, errors="coerce")

    def get_cv_scores(self) -> dict[str, list[float]]:
        """Get cross-validation scores."""
        return self._cv_scores

    def get_version(self) -> str:
        """Get model version."""
        return self._version
