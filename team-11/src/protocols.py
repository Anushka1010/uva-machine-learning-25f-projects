"""Type protocols and interfaces for the credit prediction pipeline.

This module defines Protocol classes that establish contracts for
the main components of the system. Using protocols allows for better
type checking, easier testing, and cleaner dependency injection.

We're being pedantic about types because production code should be! 🐶
"""

from typing import Protocol, Any, runtime_checkable
from pathlib import Path

import numpy as np
import pandas as pd


@runtime_checkable
class Predictor(Protocol):
    """Protocol for any model that can make predictions.
    
    This allows us to swap out XGBoost for other models without
    changing the rest of the codebase. SOLID's Liskov Substitution
    Principle in action!
    """
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate binary predictions."""
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        ...


@runtime_checkable
class Trainable(Protocol):
    """Protocol for trainable models."""
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> "Trainable":
        """Train the model."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for model evaluators."""
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model performance."""
        ...


@runtime_checkable
class Persistable(Protocol):
    """Protocol for objects that can be saved/loaded."""
    
    def save(self, path: str | Path) -> None:
        """Save to disk."""
        ...
    
    def load(self, path: str | Path) -> "Persistable":
        """Load from disk."""
        ...


@runtime_checkable
class FeatureImportanceProvider(Protocol):
    """Protocol for models that provide feature importance."""
    
    def get_feature_importance(self, importance_type: str = "weight") -> pd.DataFrame:
        """Get feature importance scores."""
        ...


@runtime_checkable 
class CreditModelProtocol(
    Predictor, Trainable, Evaluator, Persistable, FeatureImportanceProvider, Protocol
):
    """Full protocol for credit prediction models.
    
    Combines all required capabilities for a production credit model.
    """
    
    def get_underlying_model(self) -> Any:
        """Get the underlying ML model for explainability tools."""
        ...
    
    def get_training_metadata(self) -> dict[str, Any]:
        """Get metadata from training."""
        ...
    
    def get_feature_names(self) -> list[str]:
        """Get feature names used during training."""
        ...


@runtime_checkable
class ExplainerProtocol(Protocol):
    """Protocol for model explainers (SHAP, LIME, etc.)."""
    
    def explain_individual(
        self,
        X: pd.DataFrame,
        index: int,
        prediction: int,
    ) -> dict[str, Any]:
        """Generate explanation for a single prediction."""
        ...
    
    def generate_natural_language_explanation(
        self,
        explanation: dict[str, Any],
    ) -> str:
        """Convert explanation to natural language."""
        ...


@runtime_checkable
class FairnessAnalyzerProtocol(Protocol):
    """Protocol for fairness analyzers."""
    
    def apply_reweighing(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> np.ndarray:
        """Apply reweighing pre-processing."""
        ...
    
    def apply_threshold_adjustment(
        self,
        y_prob: np.ndarray,
        protected_values: np.ndarray,
    ) -> np.ndarray:
        """Apply post-processing threshold adjustment."""
        ...


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for data loaders."""
    
    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Load data from CSV."""
        ...
    
    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate loaded data."""
        ...


@runtime_checkable  
class PreprocessorProtocol(Protocol):
    """Protocol for data preprocessors."""
    
    def convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to numeric types."""
        ...
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        ...
    
    def calculate_cdi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Composite Disadvantage Index."""
        ...
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        ...
