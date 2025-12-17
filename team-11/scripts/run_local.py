#!/usr/bin/env python3
"""Local development runner for Fair Credit Score Prediction.

This script provides a simple way to run the complete local development
workflow including data generation, model training, and API startup.

Usage:
    python scripts/run_local.py              # Run everything
    python scripts/run_local.py --train      # Just train
    python scripts/run_local.py --api        # Just start API
    python scripts/run_local.py --test       # Run tests
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list[str], check: bool = True) -> int:
    """Run a command and return exit code."""
    print(f"\n\033[94m> {' '.join(cmd)}\033[0m")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"\033[91mCommand failed with exit code {result.returncode}\033[0m")
        sys.exit(result.returncode)
    return result.returncode


def ensure_dependencies():
    """Ensure all dependencies are installed."""
    try:
        import pandas
        import numpy
        import xgboost
        import fastapi
    except ImportError:
        print("\033[93mInstalling missing dependencies...\033[0m")
        run_command([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
        run_command([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn"])


def generate_sample_data():
    """Generate sample data if it doesn't exist."""
    data_path = PROJECT_ROOT / "data" / "sample_credit_data.csv"
    if not data_path.exists():
        print("\033[93mGenerating sample data...\033[0m")
        run_command([sys.executable, "scripts/generate_sample_data.py"])
    else:
        print(f"\033[92m✓ Sample data exists at {data_path}\033[0m")


def train_model():
    """Train the credit prediction model."""
    print("\n\033[92m" + "=" * 50 + "\033[0m")
    print("\033[92m  TRAINING MODEL\033[0m")
    print("\033[92m" + "=" * 50 + "\033[0m")
    
    generate_sample_data()
    run_command([
        sys.executable, "-m", "src.main", "run",
        "--data-path", "data/sample_credit_data.csv"
    ])


def start_api():
    """Start the FastAPI server."""
    model_path = PROJECT_ROOT / "models" / "credit_model.pkl"
    
    if not model_path.exists():
        print("\033[93mModel not found. Training first...\033[0m")
        train_model()
    
    print("\n\033[92m" + "=" * 50 + "\033[0m")
    print("\033[92m  STARTING API SERVER\033[0m")
    print("\033[92m" + "=" * 50 + "\033[0m")
    print("\n\033[92mAPI running at: http://127.0.0.1:8000\033[0m")
    print("\033[92mSwagger docs: http://127.0.0.1:8000/docs\033[0m")
    print("\033[93mPress Ctrl+C to stop\033[0m\n")
    
    os.environ["MODEL_PATH"] = str(model_path)
    
    run_command([
        sys.executable, "-m", "uvicorn", "src.api:app",
        "--reload", "--host", "127.0.0.1", "--port", "8000"
    ], check=False)


def run_tests():
    """Run the test suite."""
    print("\n\033[92m" + "=" * 50 + "\033[0m")
    print("\033[92m  RUNNING TESTS\033[0m")
    print("\033[92m" + "=" * 50 + "\033[0m")
    run_command([sys.executable, "-m", "pytest", "-v"])


def test_api():
    """Test the API endpoints."""
    import urllib.request
    import json
    
    base_url = "http://127.0.0.1:8000"
    
    print("\n\033[92mTesting API endpoints...\033[0m")
    
    # Health check
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            data = json.loads(response.read())
            print(f"\033[92m✓ Health: {data['status']}\033[0m")
    except Exception as e:
        print(f"\033[91m✗ Health check failed: {e}\033[0m")
        return
    
    # Fairness metrics
    try:
        with urllib.request.urlopen(f"{base_url}/fairness_metrics", timeout=5) as response:
            data = json.loads(response.read())
            print(f"\033[92m✓ Fairness metrics retrieved\033[0m")
    except Exception as e:
        print(f"\033[91m✗ Fairness metrics failed: {e}\033[0m")
    
    # Prediction
    try:
        test_data = json.dumps({
            "Income": 75000,
            "Debt": 15000,
            "Loan_Amount": 25000,
            "Loan_Term": 36,
            "Num_Credit_Cards": 3,
            "Gender": "Female",
            "Education": "Bachelor's",
            "Payment_History": "Good",
            "Employment_Status": "Employed",
            "Residence_Type": "Rented",
            "Marital_Status": "Single"
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            result = "Approved" if data['creditworthy'] else "Denied"
            print(f"\033[92m✓ Prediction: {result} (prob={data['probability']:.2f}, group={data['group']})\033[0m")
    except Exception as e:
        print(f"\033[91m✗ Prediction failed: {e}\033[0m")
    
    print("\n\033[92mAPI tests complete!\033[0m")


def main():
    parser = argparse.ArgumentParser(
        description="Local development runner for Fair Credit Score Prediction"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the model only"
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Start the API server only"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run tests only"
    )
    parser.add_argument(
        "--test-api", action="store_true",
        help="Test API endpoints (requires running server)"
    )
    parser.add_argument(
        "--generate-data", action="store_true",
        help="Generate sample data only"
    )
    
    args = parser.parse_args()
    
    print("\033[94m")
    print("=" * 50)
    print("  Fair Credit Score Prediction - Local Dev")
    print("=" * 50)
    print("\033[0m")
    
    ensure_dependencies()
    
    if args.generate_data:
        generate_sample_data()
    elif args.train:
        train_model()
    elif args.api:
        start_api()
    elif args.test:
        run_tests()
    elif args.test_api:
        test_api()
    else:
        # Default: train then start API
        train_model()
        start_api()


if __name__ == "__main__":
    main()
