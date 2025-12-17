"""Centralized configuration for the Fair Credit Score Prediction Model.

This module contains all configurable parameters for the credit prediction
pipeline, including CDI definitions, thresholds, fairness settings, and paths.
No more magic numbers scattered throughout the codebase!
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """File and directory paths configuration."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    explanations_dir: Path = Path("explanations")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")

    def __post_init__(self) -> None:
        """Convert strings to Path objects if necessary."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.explanations_dir = Path(self.explanations_dir)
        self.reports_dir = Path(self.reports_dir)
        self.models_dir = Path(self.models_dir)

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        for dir_path in [
            self.data_dir,
            self.output_dir,
            self.explanations_dir,
            self.reports_dir,
            self.reports_dir / "figures",
            self.models_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class CDIConfig:
    """Composite Disadvantage Index (CDI) configuration.

    The CDI is a proxy measure for disadvantaged groups based on structural
    demographic factors that reflect long-term socioeconomic barriers without
    directly using protected attributes like race.
    """

    # Factors that contribute to the disadvantage index (1 point each)
    factors: dict[str, str] = field(
        default_factory=lambda: {
            "Education": "High School",
            "Residence_Type": "Rented",
            "Marital_Status": "Single",
            "Gender": "Female",
        }
    )

    # Minimum CDI score to be considered "disadvantaged"
    threshold: int = 2

    # Column name for the proxy disadvantaged indicator
    proxy_column: str = "Proxy_Disadvantaged"


@dataclass
class FairnessConfig:
    """Fairness-related configuration settings."""

    # Post-processing threshold adjustments
    threshold_privileged: float = 0.5
    threshold_unprivileged: float = 0.4

    # Fairness targets (from master plan success metrics)
    target_disparate_impact_ratio: float = 0.80  # Minimum acceptable (ideal > 0.95)
    target_statistical_parity_diff: float = 0.05  # Maximum acceptable (ideal < 0.02)
    target_equalized_odds_diff: float = 0.10  # Maximum acceptable (ideal < 0.05)

    # Whether to apply fairness-aware reweighting during training
    apply_reweighting: bool = True

    # Protected attribute configuration for AIF360
    unprivileged_groups: list[dict[str, int]] = field(
        default_factory=lambda: [{"Proxy_Disadvantaged": 1}]
    )
    privileged_groups: list[dict[str, int]] = field(
        default_factory=lambda: [{"Proxy_Disadvantaged": 0}]
    )


@dataclass
class ModelConfig:
    """Model training configuration."""

    # XGBoost parameters
    xgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
        }
    )

    # Train/test split settings
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # Columns configuration
    target_column: str = "Creditworthiness"
    columns_to_drop_for_training: list[str] = field(
        default_factory=lambda: ["Proxy_Disadvantaged", "Credit_Score"]
    )


@dataclass
class ExplainabilityConfig:
    """Explainability settings for SHAP and LIME."""

    # Number of samples to explain (set to -1 for all)
    num_samples_to_explain: int = 5

    # Random seed for sampling (ensures reproducibility)
    random_seed: int = 42

    # SHAP settings
    use_shap: bool = True
    save_shap_summary_plot: bool = True

    # LIME settings
    use_lime: bool = True
    lime_num_features: int = 10

    # Natural language explanation thresholds
    strong_influence_threshold: float = 0.5
    moderate_influence_threshold: float = 0.1


@dataclass
class DataConfig:
    """Data schema and preprocessing configuration."""

    # Numeric columns to convert
    numeric_columns: list[str] = field(
        default_factory=lambda: [
            "Income",
            "Debt",
            "Loan_Amount",
            "Loan_Term",
            "Num_Credit_Cards",
            "Credit_Score",
            "Creditworthiness",
        ]
    )

    # Categorical columns to encode
    categorical_columns: list[str] = field(
        default_factory=lambda: [
            "Gender",
            "Education",
            "Payment_History",
            "Employment_Status",
            "Residence_Type",
            "Marital_Status",
        ]
    )

    # Derived features to create
    derived_features: list[str] = field(
        default_factory=lambda: ["Debt_to_Income", "Loan_to_Income"]
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file: Path = Path("logs/credit_prediction.log")

    def __post_init__(self) -> None:
        """Convert strings to Path objects if necessary."""
        self.log_file = Path(self.log_file)


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration."""

    search_type: str = "random"  # or "grid"
    cv_folds: int = 5
    n_iter: int = 50
    param_grid: dict[str, list] = field(
        default_factory=lambda: {
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
    )


@dataclass
class CalibrationConfig:
    """Model calibration configuration."""

    enabled: bool = True
    method: str = "isotonic"  # or "sigmoid"
    cv: str = "prefit"
    calibration_fraction: float = 0.3


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration."""

    enabled: bool = True
    cv_folds: int = 5
    metrics: list[str] = field(
        default_factory=lambda: ["roc_auc", "f1", "accuracy", "precision", "recall"]
    )
    stratified: bool = True


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""

    validation: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "fail_on_errors": False,
        }
    )
    scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "method": "standard",  # or "robust"
            "columns": [],
        }
    )
    handle_unseen_categories: bool = True


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    fairness_plots: bool = True
    calibration_plots: bool = True
    feature_importance_plots: bool = True
    dpi: int = 150
    figsize: list[int] = field(default_factory=lambda: [14, 10])
    style: str = "whitegrid"


@dataclass
class Config:
    """Master configuration container."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    cdi: CDIConfig = field(default_factory=CDIConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file {path} not found, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from a dictionary."""
        config = cls()

        if "paths" in data:
            config.paths = PathsConfig(**data["paths"])
        if "cdi" in data:
            config.cdi = CDIConfig(**data["cdi"])
        if "fairness" in data:
            config.fairness = FairnessConfig(**data["fairness"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "explainability" in data:
            config.explainability = ExplainabilityConfig(**data["explainability"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "tuning" in data:
            config.tuning = TuningConfig(**data["tuning"])
        if "calibration" in data:
            config.calibration = CalibrationConfig(**data["calibration"])
        if "cross_validation" in data:
            config.cross_validation = CrossValidationConfig(**data["cross_validation"])
        if "preprocessing" in data:
            config.preprocessing = PreprocessingConfig(**data["preprocessing"])
        if "visualization" in data:
            config.visualization = VisualizationConfig(**data["visualization"])

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import dataclasses

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [_to_dict(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            return obj

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)


# Default global config instance
_default_config: Config | None = None


def get_config() -> Config:
    """Get the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _default_config
    _default_config = config


def load_config(path: str | Path | None = None) -> Config:
    """Load and set the global configuration from a YAML file."""
    if path is None:
        config = Config()
    else:
        config = Config.from_yaml(path)
    set_config(config)
    return config
