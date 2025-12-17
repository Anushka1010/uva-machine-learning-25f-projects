"""Data loading and validation module.

This module handles loading credit data from various sources,
validating the schema, and providing clean data for preprocessing.
No more google.colab.files.upload() - we're production-ready now!
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import Config, DataConfig, get_config
from src.utils import get_logger, validate_dataframe, report_coercion_failures


class DataLoader:
    """Handles data loading, validation, and initial processing.

    This class replaces the Colab-specific upload flow with a flexible
    data loading system that supports local files, cloud storage, etc.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the DataLoader.

        Args:
            config: Configuration object. Uses global config if None.
        """
        self.config = config or get_config()
        self.logger = get_logger("data_loader")
        self._raw_data: pd.DataFrame | None = None
        self._validation_report: dict[str, Any] = {}

    def load_csv(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            **kwargs: Additional arguments passed to pd.read_csv.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If validation fails and strict mode is enabled.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")

        self.logger.info(f"Loading data from {file_path}")

        # Default to reading as string initially for safe type handling
        default_kwargs = {"dtype": str}
        default_kwargs.update(kwargs)

        self._raw_data = pd.read_csv(file_path, **default_kwargs)

        self.logger.info(
            f"Loaded {len(self._raw_data)} rows, {len(self._raw_data.columns)} columns"
        )

        return self._raw_data

    def validate(
        self,
        df: pd.DataFrame | None = None,
        data_config: DataConfig | None = None,
    ) -> dict[str, Any]:
        """Validate the loaded data against expected schema.

        Args:
            df: DataFrame to validate. Uses loaded data if None.
            data_config: Data configuration. Uses config if None.

        Returns:
            Validation report dictionary.
        """
        if df is None:
            if self._raw_data is None:
                raise ValueError("No data loaded. Call load_csv() first.")
            df = self._raw_data

        if data_config is None:
            data_config = self.config.data

        report: dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "issues": [],
        }

        # Check for required numeric columns
        for col in data_config.numeric_columns:
            if col not in df.columns:
                report["issues"].append(
                    {"type": "missing_column", "column": col, "expected_type": "numeric"}
                )
            else:
                # Check if convertible to numeric
                test_series = pd.to_numeric(df[col], errors="coerce")
                nan_count = test_series.isna().sum()
                if nan_count > 0:
                    original_nan = df[col].isna().sum()
                    coercion_failures = nan_count - original_nan
                    if coercion_failures > 0:
                        report["issues"].append(
                            {
                                "type": "coercion_warning",
                                "column": col,
                                "failed_count": int(coercion_failures),
                                "percentage": round(
                                    coercion_failures / len(df) * 100, 2
                                ),
                            }
                        )

        # Check for required categorical columns
        for col in data_config.categorical_columns:
            if col not in df.columns:
                report["issues"].append(
                    {
                        "type": "missing_column",
                        "column": col,
                        "expected_type": "categorical",
                    }
                )
            else:
                # Record unique values
                unique_values = df[col].unique().tolist()
                report[f"unique_{col}"] = unique_values

        # Check for missing values
        missing_counts = df.isna().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            report["missing_values"] = missing_cols.to_dict()

        self._validation_report = report

        # Log validation results
        if report["issues"]:
            self.logger.warning(f"Validation found {len(report['issues'])} issues")
            for issue in report["issues"]:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info("Validation passed with no issues")

        return report

    def get_validation_report(self) -> dict[str, Any]:
        """Get the most recent validation report."""
        return self._validation_report

    def get_raw_data(self) -> pd.DataFrame | None:
        """Get the raw loaded data."""
        return self._raw_data


def load_data(
    file_path: str | Path,
    config: Config | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience function to load and optionally validate data.

    Args:
        file_path: Path to the data file.
        config: Configuration object.
        validate: Whether to run validation.

    Returns:
        Tuple of (DataFrame, validation_report).
    """
    loader = DataLoader(config)
    df = loader.load_csv(file_path)

    validation_report = {}
    if validate:
        validation_report = loader.validate(df)

    return df, validation_report
