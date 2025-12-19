"""Configuration module for Fair Credit Score Prediction."""

from config.config import (
    Config,
    CDIConfig,
    DataConfig,
    ExplainabilityConfig,
    FairnessConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    get_config,
    load_config,
    set_config,
)

__all__ = [
    "Config",
    "CDIConfig",
    "DataConfig",
    "ExplainabilityConfig",
    "FairnessConfig",
    "LoggingConfig",
    "ModelConfig",
    "PathsConfig",
    "get_config",
    "load_config",
    "set_config",
]
