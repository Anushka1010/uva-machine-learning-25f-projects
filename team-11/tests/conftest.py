"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, load_config


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_credit_data() -> pd.DataFrame:
    """Generate sample credit data for testing."""
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame(
        {
            "Income": np.random.randint(30000, 150000, n_samples).astype(str),
            "Debt": np.random.randint(0, 50000, n_samples).astype(str),
            "Loan_Amount": np.random.randint(5000, 100000, n_samples).astype(str),
            "Loan_Term": np.random.choice([12, 24, 36, 48, 60], n_samples).astype(str),
            "Num_Credit_Cards": np.random.randint(0, 10, n_samples).astype(str),
            "Credit_Score": np.random.randint(300, 850, n_samples).astype(str),
            "Gender": np.random.choice(["Male", "Female"], n_samples),
            "Education": np.random.choice(
                ["High School", "Bachelor's", "Master's", "PhD"], n_samples
            ),
            "Payment_History": np.random.choice(
                ["Excellent", "Good", "Fair", "Poor"], n_samples
            ),
            "Employment_Status": np.random.choice(
                ["Employed", "Self-Employed", "Unemployed"], n_samples
            ),
            "Residence_Type": np.random.choice(["Owned", "Rented", "Mortgage"], n_samples),
            "Marital_Status": np.random.choice(
                ["Single", "Married", "Divorced"], n_samples
            ),
            "Creditworthiness": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]).astype(
                str
            ),
        }
    )

    return data


@pytest.fixture
def sample_csv_path(test_data_dir: Path, sample_credit_data: pd.DataFrame) -> Path:
    """Save sample data to CSV and return path."""
    csv_path = test_data_dir / "test_credit_data.csv"
    sample_credit_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def test_config(test_data_dir: Path) -> Config:
    """Create a test configuration."""
    config = Config()
    config.paths.data_dir = test_data_dir / "data"
    config.paths.output_dir = test_data_dir / "outputs"
    config.paths.explanations_dir = test_data_dir / "explanations"
    config.paths.reports_dir = test_data_dir / "reports"
    config.paths.models_dir = test_data_dir / "models"
    config.logging.log_to_file = False  # Don't create log files during tests
    config.explainability.num_samples_to_explain = 2  # Fewer for faster tests
    return config


@pytest.fixture
def preprocessed_data(sample_credit_data: pd.DataFrame, test_config: Config):
    """Return preprocessed data ready for model training."""
    from src.preprocessing import preprocess_data

    X_train, X_test, y_train, y_test, processed_df = preprocess_data(
        sample_credit_data, test_config
    )
    return X_train, X_test, y_train, y_test, processed_df
