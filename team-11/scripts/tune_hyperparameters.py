#!/usr/bin/env python3
"""Automated hyperparameter tuning script.

This script provides automated hyperparameter tuning with multiple search strategies.
It can save the best parameters directly to a config file for easy integration.

Usage:
    python scripts/tune_hyperparameters.py --data-path data/credit.csv --search-type random
    python scripts/tune_hyperparameters.py --data-path data/credit.csv --search-type grid --save-to-config
"""

import sys
from pathlib import Path
import argparse
import json
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from src.utils import setup_logging, set_random_seed
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import CreditModel


def main():
    parser = argparse.ArgumentParser(
        description="Automated hyperparameter tuning for credit prediction model"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--search-type",
        choices=["random", "grid"],
        default="random",
        help="Search strategy (random or grid)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of iterations for random search"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--save-to-config",
        action="store_true",
        help="Save best parameters to config file"
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("config/tuned_config.yaml"),
        help="Path to save updated config"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.paths.ensure_dirs()

    logger = setup_logging(config.logging)
    logger.info("="*70)
    logger.info("AUTOMATED HYPERPARAMETER TUNING")
    logger.info("="*70)
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Search type: {args.search_type}")
    logger.info(f"Iterations: {args.n_iter if args.search_type == 'random' else 'all combinations'}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info("="*70)

    set_random_seed(config.model.random_state)

    # Load and preprocess data
    logger.info("\nLoading and preprocessing data...")
    data_loader = DataLoader(config)
    raw_data = data_loader.load_csv(args.data_path)

    preprocessor = Preprocessor(config)
    X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(raw_data)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Run hyperparameter tuning
    logger.info(f"\nStarting {args.search_type} search...")
    model = CreditModel(config)

    tuning_results = model.tune_hyperparameters(
        X_train,
        y_train,
        search_type=args.search_type,
        param_grid=config.tuning.param_grid,
        cv=args.cv_folds,
        n_iter=args.n_iter if args.search_type == "random" else None,
    )
    best_params = tuning_results["best_params"]
    cv_results = tuning_results["cv_results"]

    # Display results
    logger.info("\n" + "="*70)
    logger.info("TUNING RESULTS")
    logger.info("="*70)
    logger.info(f"Best CV Score: {tuning_results['best_score']:.6f}")
    logger.info("\nBest Parameters:")
    for param, value in sorted(best_params.items()):
        logger.info(f"  {param:20s}: {value}")
    logger.info("="*70)

    # Prepare output
    output_data = {
        "search_type": args.search_type,
        "cv_folds": args.cv_folds,
        "best_score": tuning_results["best_score"],
        "best_params": best_params,
        "all_results": cv_results,
        "data_path": str(args.data_path),
    }

    # Save results JSON
    output_path = args.output or config.paths.output_dir / f"tuning_results_{args.search_type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")

    # Optionally save to config file
    if args.save_to_config:
        logger.info(f"\nSaving best parameters to config: {args.config_output}")

        # Load existing config or create new one
        if args.config and Path(args.config).exists():
            with open(args.config) as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config._to_dict(config)

        # Update XGBoost parameters
        if "model" not in config_data:
            config_data["model"] = {}
        if "xgb_params" not in config_data["model"]:
            config_data["model"]["xgb_params"] = {}

        # Merge best params
        config_data["model"]["xgb_params"].update(best_params)

        # Save updated config
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.config_output, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Config saved to: {args.config_output}")
        logger.info("\nTo use the tuned parameters, run:")
        logger.info(f"  python -m src.main run --data-path {args.data_path} --config {args.config_output}")

    logger.info("\nTuning complete!")


if __name__ == "__main__":
    main()
