"""Tests for configuration module."""

import pytest
from pathlib import Path
import tempfile
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    Config,
    CDIConfig,
    FairnessConfig,
    ModelConfig,
    ExplainabilityConfig,
    PathsConfig,
    get_config,
    set_config,
    load_config,
)


class TestConfigDataclasses:
    """Tests for configuration dataclasses."""

    def test_cdi_config_defaults(self):
        """Test CDI config has sensible defaults."""
        config = CDIConfig()

        assert config.threshold == 2
        assert config.proxy_column == "Proxy_Disadvantaged"
        assert "Education" in config.factors
        assert "Gender" in config.factors

    def test_fairness_config_defaults(self):
        """Test fairness config defaults match master plan targets."""
        config = FairnessConfig()

        assert config.threshold_privileged == 0.5
        assert config.threshold_unprivileged == 0.4
        assert config.target_disparate_impact_ratio == 0.80
        assert config.target_statistical_parity_diff == 0.05
        assert config.apply_reweighting is True

    def test_model_config_defaults(self):
        """Test model config defaults."""
        config = ModelConfig()

        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.target_column == "Creditworthiness"
        assert "Proxy_Disadvantaged" in config.columns_to_drop_for_training

    def test_explainability_config_defaults(self):
        """Test explainability config defaults."""
        config = ExplainabilityConfig()

        assert config.num_samples_to_explain == 5
        assert config.random_seed == 42
        assert config.use_shap is True
        assert config.use_lime is True

    def test_paths_config_creates_path_objects(self):
        """Test paths config converts strings to Path objects."""
        config = PathsConfig(data_dir="my/data/path")

        assert isinstance(config.data_dir, Path)
        assert config.data_dir == Path("my/data/path")


class TestConfig:
    """Tests for the master Config class."""

    def test_config_creates_all_subconfigs(self):
        """Test Config initializes all sub-configurations."""
        config = Config()

        assert isinstance(config.cdi, CDIConfig)
        assert isinstance(config.fairness, FairnessConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.explainability, ExplainabilityConfig)
        assert isinstance(config.paths, PathsConfig)

    def test_config_to_yaml_and_from_yaml(self):
        """Test config can be saved and loaded from YAML."""
        original = Config()
        original.cdi.threshold = 3
        original.fairness.threshold_unprivileged = 0.35

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            original.to_yaml(temp_path)
            loaded = Config.from_yaml(temp_path)

            assert loaded.cdi.threshold == 3
            assert loaded.fairness.threshold_unprivileged == 0.35
        finally:
            Path(temp_path).unlink()

    def test_config_from_yaml_nonexistent_returns_defaults(self):
        """Test loading nonexistent YAML returns defaults."""
        config = Config.from_yaml("/nonexistent/path.yaml")

        assert config.cdi.threshold == 2  # Default value


class TestConfigGlobalState:
    """Tests for global config management."""

    def test_get_config_returns_default(self):
        """Test get_config returns a Config instance."""
        config = get_config()

        assert isinstance(config, Config)

    def test_set_config_updates_global(self):
        """Test set_config updates the global config."""
        custom = Config()
        custom.cdi.threshold = 99

        set_config(custom)
        retrieved = get_config()

        assert retrieved.cdi.threshold == 99

        # Reset to avoid affecting other tests
        set_config(Config())

    def test_load_config_sets_global(self):
        """Test load_config sets the global config."""
        config = load_config(None)

        assert isinstance(config, Config)
        assert get_config() is config
