"""Fair Credit Score Prediction - Source Package.

This package contains the modular components for building a fair,
transparent, and production-ready credit scoring system.

Modules:
    model: Credit model training and prediction
    preprocessing: Data preprocessing and feature engineering
    fairness: Fairness analysis and mitigation
    explainability: SHAP and LIME explanations
    data_loader: Data loading and validation
    utils: Utility functions and helpers
    api: FastAPI REST endpoints
    validators: Input validation utilities
    protocols: Type protocols for better type safety
    protected_attribute: DRY utilities for protected attribute handling
"""

from __future__ import annotations

import os
from pathlib import Path


def _ensure_writable_dir(path: Path) -> bool:
    """Attempt to create and write to a directory, returning success."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _configure_runtime_environment() -> None:
    """Provide writable cache/config directories for plotting libraries."""
    home_dir = Path.home()
    default_mpl_dir = home_dir / ".matplotlib"
    default_cache_dir = home_dir / ".cache"
    default_fontconfig_dir = home_dir / ".fontconfig"

    has_default_access = all(
        [
            _ensure_writable_dir(default_mpl_dir),
            _ensure_writable_dir(default_cache_dir / "fontconfig"),
            _ensure_writable_dir(default_fontconfig_dir),
        ]
    )

    if has_default_access:
        os.environ.setdefault("MPLCONFIGDIR", str(default_mpl_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(default_cache_dir))
        return

    workspace = Path.cwd()
    fallback_home = workspace / ".runtime-home"
    fallback_mpl_dir = fallback_home / ".matplotlib"
    fallback_cache_dir = fallback_home / ".cache"
    fallback_fontconfig_dir = fallback_home / ".fontconfig"
    fallback_config_dir = fallback_home / ".config"

    for directory in [
        fallback_home,
        fallback_mpl_dir,
        fallback_cache_dir / "fontconfig",
        fallback_fontconfig_dir,
        fallback_config_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(fallback_home)
    os.environ["MPLCONFIGDIR"] = str(fallback_mpl_dir)
    os.environ["XDG_CACHE_HOME"] = str(fallback_cache_dir)
    os.environ["XDG_CONFIG_HOME"] = str(fallback_config_dir)


_configure_runtime_environment()

__version__ = "0.1.0"

# Convenience imports
from src.model import CreditModel
from src.preprocessing import Preprocessor, preprocess_data
from src.fairness import FairnessAnalyzer, FairnessMetrics
from src.explainability import Explainer
from src.data_loader import DataLoader, load_data
from src.validators import ValidationError
from src.protected_attribute import (
    prepare_features_without_protected,
    without_protected_attribute,
    resolve_protected_attribute,
)

__all__ = [
    "__version__",
    # Core classes
    "CreditModel",
    "Preprocessor",
    "FairnessAnalyzer",
    "FairnessMetrics",
    "Explainer",
    "DataLoader",
    # Convenience functions
    "preprocess_data",
    "load_data",
    # Utilities
    "ValidationError",
    "prepare_features_without_protected",
    "without_protected_attribute",
    "resolve_protected_attribute",
]
