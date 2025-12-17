"""Model Registry for A/B Testing and Version Management.

This module provides functionality for:
- Registering multiple model versions
- A/B testing with traffic splitting
- Model version comparison
- Rollback capabilities
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

from src.model import CreditModel
from src.preprocessing import Preprocessor
from src.explainability import Explainer
from src.utils import get_logger


logger = get_logger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    version: str
    name: str
    description: str
    created_at: str
    updated_at: str
    status: ModelStatus
    model_path: str
    preprocessor_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    traffic_weight: float = 0.0  # For A/B testing (0.0 - 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data


class ModelRegistry:
    """Registry for managing multiple model versions."""

    def __init__(self, registry_path: Path = Path("models/registry.json")):
        """Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, CreditModel] = {}
        self.loaded_preprocessors: Dict[str, Preprocessor] = {}
        self.loaded_explainers: Dict[str, Explainer] = {}

        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)

                for version, meta_dict in data.items():
                    meta_dict['status'] = ModelStatus(meta_dict['status'])
                    self.models[version] = ModelMetadata(**meta_dict)

                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.models = {}
        else:
            logger.info("No existing registry found, starting fresh")

    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {version: meta.to_dict() for version, meta in self.models.items()}

            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved registry with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_model(
        self,
        version: str,
        name: str,
        model_path: str | Path,
        description: str = "",
        preprocessor_path: Optional[str | Path] = None,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        status: ModelStatus = ModelStatus.INACTIVE,
    ) -> ModelMetadata:
        """Register a new model version.

        Args:
            version: Version identifier (e.g., "v1.0.0", "2024-01-15-prod")
            name: Human-readable model name
            model_path: Path to model file
            description: Model description
            preprocessor_path: Path to preprocessor file
            metrics: Model performance metrics
            tags: Model tags for categorization
            status: Initial deployment status

        Returns:
            ModelMetadata object
        """
        if version in self.models:
            raise ValueError(f"Model version '{version}' already registered")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        metadata = ModelMetadata(
            version=version,
            name=name,
            description=description,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            status=status,
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path) if preprocessor_path else None,
            metrics=metrics or {},
            tags=tags or [],
            traffic_weight=0.0,
        )

        self.models[version] = metadata
        self._save_registry()

        logger.info(f"Registered model version '{version}': {name}")
        return metadata

    def load_model(self, version: str, config: Any) -> CreditModel:
        """Load a model version into memory.

        Args:
            version: Model version to load
            config: Configuration object

        Returns:
            Loaded CreditModel
        """
        if version not in self.models:
            raise ValueError(f"Model version '{version}' not found in registry")

        # Check if already loaded
        if version in self.loaded_models:
            logger.info(f"Model '{version}' already loaded")
            return self.loaded_models[version]

        metadata = self.models[version]
        model_path = Path(metadata.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        model = CreditModel(config)
        model.load(model_path)
        self.loaded_models[version] = model

        # Load preprocessor if available
        if metadata.preprocessor_path:
            preprocessor_path = Path(metadata.preprocessor_path)
            if preprocessor_path.exists():
                preprocessor = Preprocessor(config)
                preprocessor.load(preprocessor_path)
                self.loaded_preprocessors[version] = preprocessor

        # Setup explainer
        explainer = Explainer(config)
        explainer.setup_shap_explainer(model)
        self.loaded_explainers[version] = explainer

        logger.info(f"Loaded model version '{version}' into memory")
        return model

    def unload_model(self, version: str):
        """Unload a model from memory.

        Args:
            version: Model version to unload
        """
        if version in self.loaded_models:
            del self.loaded_models[version]
        if version in self.loaded_preprocessors:
            del self.loaded_preprocessors[version]
        if version in self.loaded_explainers:
            del self.loaded_explainers[version]

        logger.info(f"Unloaded model version '{version}' from memory")

    def set_traffic_weights(self, weights: Dict[str, float]):
        """Set traffic weights for A/B testing.

        Args:
            weights: Dictionary mapping version to traffic weight (0.0 - 1.0)
                    Weights should sum to 1.0

        Raises:
            ValueError: If weights don't sum to 1.0 or contain invalid versions
        """
        total_weight = sum(weights.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Traffic weights must sum to 1.0, got {total_weight}")

        for version in weights:
            if version not in self.models:
                raise ValueError(f"Model version '{version}' not found in registry")

        # Update weights
        for version, weight in weights.items():
            self.models[version].traffic_weight = weight
            self.models[version].updated_at = datetime.utcnow().isoformat()

        self._save_registry()
        logger.info(f"Updated traffic weights: {weights}")

    def get_model_for_request(self, request_hash: str) -> Optional[str]:
        """Get model version for a request based on traffic weights.

        Uses consistent hashing to assign requests to model versions.

        Args:
            request_hash: Hash of request (for consistent routing)

        Returns:
            Model version to use, or None if no active models
        """
        # Get models with traffic weight > 0
        weighted_models = [
            (version, meta.traffic_weight)
            for version, meta in self.models.items()
            if meta.traffic_weight > 0
        ]

        if not weighted_models:
            # Fall back to first active model
            active_models = [
                version for version, meta in self.models.items()
                if meta.status == ModelStatus.ACTIVE
            ]
            return active_models[0] if active_models else None

        # Consistent hashing based on request
        hash_value = int(hashlib.md5(request_hash.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 1000) / 1000.0  # 0.0 - 1.0

        # Select model based on cumulative weights
        cumulative_weight = 0.0
        for version, weight in weighted_models:
            cumulative_weight += weight
            if normalized_hash <= cumulative_weight:
                return version

        # Fallback to last model
        return weighted_models[-1][0]

    def set_status(self, version: str, status: ModelStatus):
        """Set model status.

        Args:
            version: Model version
            status: New status
        """
        if version not in self.models:
            raise ValueError(f"Model version '{version}' not found")

        self.models[version].status = status
        self.models[version].updated_at = datetime.utcnow().isoformat()
        self._save_registry()

        logger.info(f"Set model '{version}' status to {status.value}")

    def update_metrics(self, version: str, metrics: Dict[str, Any]):
        """Update model metrics.

        Args:
            version: Model version
            metrics: Metrics to update
        """
        if version not in self.models:
            raise ValueError(f"Model version '{version}' not found")

        self.models[version].metrics.update(metrics)
        self.models[version].updated_at = datetime.utcnow().isoformat()
        self._save_registry()

        logger.info(f"Updated metrics for model '{version}'")

    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """List registered models.

        Args:
            status: Filter by status
            tags: Filter by tags (must have all tags)

        Returns:
            List of ModelMetadata objects
        """
        models = list(self.models.values())

        if status:
            models = [m for m in models if m.status == status]

        if tags:
            models = [m for m in models if all(tag in m.tags for tag in tags)]

        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def get_model_metadata(self, version: str) -> ModelMetadata:
        """Get metadata for a model version.

        Args:
            version: Model version

        Returns:
            ModelMetadata object
        """
        if version not in self.models:
            raise ValueError(f"Model version '{version}' not found")

        return self.models[version]

    def compare_models(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """Compare two model versions.

        Args:
            version_a: First model version
            version_b: Second model version

        Returns:
            Comparison dictionary
        """
        if version_a not in self.models or version_b not in self.models:
            raise ValueError("One or both model versions not found")

        meta_a = self.models[version_a]
        meta_b = self.models[version_b]

        comparison = {
            "version_a": {
                "version": version_a,
                "name": meta_a.name,
                "created_at": meta_a.created_at,
                "status": meta_a.status.value,
                "traffic_weight": meta_a.traffic_weight,
                "metrics": meta_a.metrics,
            },
            "version_b": {
                "version": version_b,
                "name": meta_b.name,
                "created_at": meta_b.created_at,
                "status": meta_b.status.value,
                "traffic_weight": meta_b.traffic_weight,
                "metrics": meta_b.metrics,
            },
            "metric_differences": {},
        }

        # Calculate metric differences
        common_metrics = set(meta_a.metrics.keys()) & set(meta_b.metrics.keys())
        for metric in common_metrics:
            val_a = meta_a.metrics[metric]
            val_b = meta_b.metrics[metric]
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_b - val_a
                pct_change = ((val_b - val_a) / val_a * 100) if val_a != 0 else 0
                comparison["metric_differences"][metric] = {
                    "value_a": val_a,
                    "value_b": val_b,
                    "absolute_difference": diff,
                    "percent_change": pct_change,
                }

        return comparison
