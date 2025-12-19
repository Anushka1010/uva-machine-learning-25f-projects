"""Data preprocessing and feature engineering module.

This module handles all data transformations including:
- Numeric type coercion with proper NaN handling
- Derived feature creation (Debt_to_Income, Loan_to_Income)
- Composite Disadvantage Index (CDI) calculation
- Categorical encoding
- Train/test splitting
- Pipeline persistence for inference
- Data quality validation
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from config.config import Config, CDIConfig, DataConfig, ModelConfig, get_config
from src.utils import get_logger, report_coercion_failures


class Preprocessor:
    """Handles all data preprocessing operations.

    This class transforms raw data into model-ready features while
    maintaining the ability to track and reproduce all transformations.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the Preprocessor.

        Args:
            config: Configuration object. Uses global config if None.
        """
        self.config = config or get_config()
        self.logger = get_logger("preprocessing")
        self._coercion_report: dict[str, int] = {}
        self._encoded_columns: list[str] = []
        self._categorical_mappings: dict[str, list[str]] = {}
        self._scaler: StandardScaler | RobustScaler | None = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _ensure_fitted(self, action: str) -> None:
        """Ensure the preprocessor has been fitted before inference ops."""
        if not self._is_fitted or not self._feature_names:
            raise RuntimeError(
                f"Preprocessor must be fitted before calling {action}. "
                "Run fit_transform() on training data or load a saved preprocessor."
            )

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """Run the full preprocessing pipeline and mark the preprocessor fitted."""
        processed = self.convert_numeric_columns(df)
        processed = self.create_derived_features(processed)
        processed = self.calculate_cdi(processed)
        processed = self.encode_categorical(processed)

        X, y = self.prepare_features_and_target(processed)

        model_feature_names = list(X.columns)

        # Persist feature metadata (without protected attribute) for alignment
        self._feature_names = model_feature_names
        self._is_fitted = True

        # Preserve protected attribute column for fairness analysis on splits
        proxy_col = self.config.cdi.proxy_column
        if proxy_col in processed.columns and proxy_col not in X.columns:
            X[proxy_col] = processed.loc[X.index, proxy_col].astype(float)

        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train = X_train.apply(pd.to_numeric, errors="coerce")
        X_test = X_test.apply(pd.to_numeric, errors="coerce")

        return X_train, X_test, y_train, y_test, processed

    def convert_numeric_columns(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Convert specified columns to numeric types.

        Args:
            df: Input DataFrame.
            columns: Columns to convert. Uses config if None.

        Returns:
            DataFrame with converted columns.
        """
        df = df.copy()

        if columns is None:
            columns = self.config.data.numeric_columns

        original_count = len(df)

        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                nan_count = report_coercion_failures(
                    df, col, original_count, self.logger
                )
                self._coercion_report[col] = nan_count

        self.logger.info(f"Converted {len(columns)} columns to numeric")
        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with derived features added.
        """
        df = df.copy()

        # Debt to Income ratio
        if "Debt" in df.columns and "Income" in df.columns:
            # Avoid division by zero
            df["Debt_to_Income"] = np.where(
                df["Income"] != 0, df["Debt"] / df["Income"], np.nan
            )
            self.logger.info("Created Debt_to_Income feature")

        # Loan to Income ratio
        if "Loan_Amount" in df.columns and "Income" in df.columns:
            df["Loan_to_Income"] = np.where(
                df["Income"] != 0, df["Loan_Amount"] / df["Income"], np.nan
            )
            self.logger.info("Created Loan_to_Income feature")

        return df

    def calculate_cdi(
        self,
        df: pd.DataFrame,
        cdi_config: CDIConfig | None = None,
    ) -> pd.DataFrame:
        """Calculate the Composite Disadvantage Index (CDI).

        The CDI is a proxy measure for disadvantaged groups based on
        structural demographic factors. Each factor contributes 1 point.

        Args:
            df: Input DataFrame.
            cdi_config: CDI configuration. Uses config if None.

        Returns:
            DataFrame with CDI and Proxy_Disadvantaged columns added.
        """
        df = df.copy()

        if cdi_config is None:
            cdi_config = self.config.cdi

        # Calculate CDI as sum of matching factors
        cdi_score = pd.Series(0, index=df.index)

        for column, value in cdi_config.factors.items():
            if column in df.columns:
                cdi_score += (df[column] == value).astype(int)
                self.logger.debug(f"CDI factor: {column} == '{value}'")
            else:
                self.logger.warning(
                    f"CDI factor column '{column}' not found in data"
                )

        df["CDI"] = cdi_score

        # Create binary proxy disadvantaged indicator
        df[cdi_config.proxy_column] = (cdi_score >= cdi_config.threshold).astype(int)

        # Log distribution
        disadvantaged_count = df[cdi_config.proxy_column].sum()
        total = len(df)
        self.logger.info(
            f"CDI calculated: {disadvantaged_count}/{total} "
            f"({disadvantaged_count/total*100:.1f}%) classified as disadvantaged "
            f"(threshold >= {cdi_config.threshold})"
        )

        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        drop_first: bool = True,
        fit: bool = True,
    ) -> pd.DataFrame:
        """One-hot encode categorical columns.

        Args:
            df: Input DataFrame.
            columns: Columns to encode. Uses config if None.
            drop_first: Whether to drop first category (avoid multicollinearity).
            fit: Whether to update internal categorical mappings.

        Returns:
            DataFrame with encoded columns.
        """
        df = df.copy()

        if columns is None:
            columns = self.config.data.categorical_columns

        # Filter to columns that exist in the dataframe
        columns_to_encode = [col for col in columns if col in df.columns]

        if not columns_to_encode:
            self.logger.warning("No categorical columns found to encode")
            return df

        # Ensure all categorical columns are strings
        for col in columns_to_encode:
            df[col] = df[col].astype(str)
            # Store unique categories during fitting for later validation
            if fit and col not in self._categorical_mappings:
                self._categorical_mappings[col] = df[col].unique().tolist()

        # Store original columns for reference
        original_cols = set(df.columns)

        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=drop_first)

        # Track new encoded columns
        new_cols = set(df.columns) - original_cols
        if fit:
            self._encoded_columns = list(new_cols)

        message = (
            f"Encoded {len(columns_to_encode)} categorical columns into "
            f"{len(new_cols)} dummy variables"
        )
        if fit:
            self.logger.info(message)
        else:
            self.logger.debug(message)

        return df

    def prepare_features_and_target(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        columns_to_drop: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features and target.

        Args:
            df: Input DataFrame.
            target_column: Name of target column. Uses config if None.
            columns_to_drop: Additional columns to drop. Uses config if None.

        Returns:
            Tuple of (features_df, target_series).
        """
        if target_column is None:
            target_column = self.config.model.target_column

        if columns_to_drop is None:
            columns_to_drop = self.config.model.columns_to_drop_for_training

        # Extract target
        y = df[target_column].astype(float)

        # Prepare features (drop target + specified columns)
        cols_to_drop = [target_column] + [
            c for c in columns_to_drop if c in df.columns and c != target_column
        ]
        X = df.drop(columns=cols_to_drop)

        self.logger.info(
            f"Prepared features: {X.shape[1]} features, {len(y)} samples"
        )

        if not self._feature_names:
            self._feature_names = list(X.columns)

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attribute: pd.Series | None = None,
        test_size: float | None = None,
        random_state: int | None = None,
        stratify: bool | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            protected_attribute: Protected attribute for stratification tracking.
            test_size: Fraction for test set. Uses config if None.
            random_state: Random seed. Uses config if None.
            stratify: Whether to stratify. Uses config if None.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        model_config = self.config.model

        if test_size is None:
            test_size = model_config.test_size
        if random_state is None:
            random_state = model_config.random_state
        if stratify is None:
            stratify = model_config.stratify

        stratify_col = y if stratify else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_col,
            )
        except ValueError as exc:
            if stratify_col is not None and "least populated class" in str(exc):
                self.logger.warning(
                    "Stratified split failed due to insufficient class members; "
                    "falling back to non-stratified split."
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=None,
                )
            else:
                raise

        self.logger.info(
            f"Split data: {len(X_train)} train, {len(X_test)} test "
            f"(test_size={test_size}, seed={random_state})"
        )

        return X_train, X_test, y_train, y_test

    def validate_data(self, df: pd.DataFrame, required_columns: list[str] | None = None) -> dict[str, Any]:
        """Validate input data quality.

        Args:
            df: DataFrame to validate.
            required_columns: List of required columns.

        Returns:
            Dictionary containing validation results.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")

        duplicate_rows = int(df.duplicated().sum())
        validation_report = {
            "shape": (len(df), len(df.columns)),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": 0,
            "missing_value_columns": {},
            "duplicate_rows": duplicate_rows,
            "duplicates": duplicate_rows,
            "constant_columns": [],
            "high_cardinality_columns": [],
            "issues": [],
        }

        # Check missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                validation_report["missing_values"] += int(missing_count)
                validation_report["missing_value_columns"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_count / len(df) * 100),
                }
                validation_report["issues"].append(
                    f"Column '{col}' has {missing_count} missing values"
                )

        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                validation_report["constant_columns"].append(col)
                validation_report["issues"].append(
                    f"Column '{col}' has constant value {df[col].iloc[0]!r}"
                )

        # Check for high cardinality categorical columns
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                validation_report["high_cardinality_columns"].append({
                    "column": col,
                    "unique_count": int(df[col].nunique()),
                    "unique_ratio": float(unique_ratio),
                })
                validation_report["issues"].append(
                    f"Column '{col}' has high cardinality ({unique_ratio:.2%})"
                )

        if validation_report["duplicate_rows"] > 0:
            validation_report["issues"].append(
                f"Found {int(validation_report['duplicate_rows'])} duplicate rows"
            )

        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        self.logger.info(f"Data validation completed: {len(df)} rows, {len(df.columns)} columns")
        if validation_report["missing_values"]:
            self.logger.warning(
                f"Found missing values in {len(validation_report['missing_value_columns'])} columns"
            )
        if validation_report["constant_columns"]:
            self.logger.warning(f"Found {len(validation_report['constant_columns'])} constant columns")

        return validation_report

    def apply_scaling(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: str = "standard",
        fit: bool = True,
    ) -> pd.DataFrame:
        """Apply feature scaling to numeric columns.

        Args:
            df: Input DataFrame.
            columns: Columns to scale. If None, scales all numeric columns.
            method: Scaling method ('standard' or 'robust').
            fit: Whether to fit the scaler (True for training, False for test).

        Returns:
            DataFrame with scaled features.
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        columns = [col for col in columns if col in df.columns]

        if not columns:
            self.logger.warning("No numeric columns found for scaling")
            return df

        if fit or self._scaler is None:
            if method == "standard":
                self._scaler = StandardScaler()
            elif method == "robust":
                self._scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            df[columns] = self._scaler.fit_transform(df[columns])
            self.logger.info(f"Fitted and transformed {len(columns)} columns using {method} scaling")
        else:
            df[columns] = self._scaler.transform(df[columns])
            self.logger.info(f"Transformed {len(columns)} columns using fitted scaler")

        return df

    def align_to_feature_space(
        self,
        df: pd.DataFrame,
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        """Align columns to the fitted feature space, adding or removing columns.

        Args:
            df: DataFrame to align.
            fill_value: Value to use for missing columns.

        Returns:
            DataFrame aligned to training feature order.
        """
        self._ensure_fitted("align_to_feature_space()")

        current_cols = set(df.columns)
        missing = [col for col in self._feature_names if col not in current_cols]
        extra = [col for col in df.columns if col not in self._feature_names]

        if missing:
            self.logger.debug(
                f"Adding {len(missing)} missing feature columns with fill value {fill_value}"
            )
            for col in missing:
                df[col] = fill_value

        if extra:
            self.logger.debug(
                f"Dropping {len(extra)} extra feature columns not seen during training: {extra}"
            )
            df = df.drop(columns=extra)

        aligned = df[self._feature_names].copy()
        aligned = aligned.apply(pd.to_numeric, errors="coerce")
        return aligned

    def prepare_inference_features(
        self,
        df: pd.DataFrame,
        handle_unseen: bool = True,
    ) -> pd.DataFrame:
        """Full preprocessing pipeline for inference data aligned to training features.

        Args:
            df: Raw inference DataFrame.
            handle_unseen: Whether to map unseen categories to 'Unknown'.

        Returns:
            DataFrame ready for model prediction (still includes protected column).
        """
        self._ensure_fitted("prepare_inference_features()")

        processed = self.convert_numeric_columns(df)
        processed = self.create_derived_features(processed)
        processed = self.calculate_cdi(processed)

        if handle_unseen:
            processed = self.handle_unseen_categories(processed)

        processed = self.encode_categorical(processed, fit=False)

        target_column = self.config.model.target_column
        if target_column in processed.columns:
            processed = processed.drop(columns=[target_column])

        proxy_col = self.config.cdi.proxy_column
        proxy_series = processed[proxy_col] if proxy_col in processed.columns else None

        features = processed.drop(columns=[proxy_col], errors="ignore")
        features = self.align_to_feature_space(features)

        if proxy_series is not None:
            features[proxy_col] = proxy_series

        return features.apply(pd.to_numeric, errors="coerce")

    def handle_unseen_categories(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Handle unseen categories in test data by mapping to 'Unknown'.

        Args:
            df: Input DataFrame.
            categorical_columns: Columns to process.

        Returns:
            DataFrame with unseen categories handled.
        """
        df = df.copy()

        if categorical_columns is None:
            categorical_columns = self.config.data.categorical_columns

        for col in categorical_columns:
            if col not in df.columns:
                continue

            if col in self._categorical_mappings:
                known_categories = self._categorical_mappings[col]
                mask = ~df[col].isin(known_categories)
                if mask.any():
                    unseen_count = mask.sum()
                    self.logger.warning(
                        f"Found {unseen_count} unseen categories in '{col}', mapping to 'Unknown'"
                    )
                    df.loc[mask, col] = "Unknown"

        return df

    def save(self, path: str | Path) -> None:
        """Save preprocessor state for inference.

        Args:
            path: Path to save the preprocessor.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "coercion_report": self._coercion_report,
            "encoded_columns": self._encoded_columns,
            "categorical_mappings": self._categorical_mappings,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        self.logger.info(f"Preprocessor saved to {path}")

    def load(self, path: str | Path) -> "Preprocessor":
        """Load preprocessor state from disk.

        Args:
            path: Path to the saved preprocessor.

        Returns:
            Self for method chaining.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.config = state.get("config", self.config)
        self._coercion_report = state.get("coercion_report", {})
        self._encoded_columns = state.get("encoded_columns", [])
        self._categorical_mappings = state.get("categorical_mappings", {})
        self._scaler = state.get("scaler")
        self._feature_names = state.get("feature_names", [])
        self._is_fitted = state.get("is_fitted", False)

        self.logger.info(f"Preprocessor loaded from {path}")

        return self

    def get_coercion_report(self) -> dict[str, int]:
        """Get the numeric coercion failure report."""
        return self._coercion_report

    def get_encoded_columns(self) -> list[str]:
        """Get list of columns created during encoding."""
        return self._encoded_columns

    def get_feature_names(self) -> list[str]:
        """Get list of final feature names after preprocessing."""
        return self._feature_names

    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._is_fitted


def preprocess_data(
    df: pd.DataFrame,
    config: Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Convenience function for full preprocessing pipeline.

    Args:
        df: Raw input DataFrame.
        config: Configuration object.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, processed_df).
    """
    preprocessor = Preprocessor(config)
    return preprocessor.fit_transform(df)
