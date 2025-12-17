"""Explainability module for SHAP and LIME explanations.

This module provides model explainability through:
- SHAP (SHapley Additive exPlanations) for global and local explanations
- LIME (Local Interpretable Model-agnostic Explanations)
- Natural language explanation generation
- Visualization and reporting
- Waterfall plots and force plots
- Interactive HTML reports
- Comparison explanations
"""

import logging
import random
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

from config.config import Config, ExplainabilityConfig, get_config
from src.utils import get_logger, set_random_seed
from src.model import CreditModel

from utils.serialization import to_python

class Explainer:
    """Handles model explanations using SHAP and LIME.

    This class generates both global feature importance explanations
    and individual decision explanations with natural language summaries.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the Explainer.

        Args:
            config: Configuration object. Uses global config if None.
        """
        self.config = config or get_config()
        self.logger = get_logger("explainability")
        self._shap_explainer: shap.TreeExplainer | None = None
        self._lime_explainer: lime.lime_tabular.LimeTabularExplainer | None = None
        self._shap_values: np.ndarray | None = None

    def setup_shap_explainer(self, model: CreditModel) -> None:
        """Initialize SHAP explainer for the model.

        Args:
            model: Trained CreditModel instance.
        """
        underlying_model = model.get_underlying_model()
        if underlying_model is None:
            raise ValueError("Model not trained")

        self._shap_explainer = shap.TreeExplainer(underlying_model)
        self.logger.info("SHAP TreeExplainer initialized")

    def setup_lime_explainer(
        self,
        X_train: pd.DataFrame,
        feature_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
    ) -> None:
        """Initialize LIME explainer.

        Args:
            X_train: Training data for LIME background.
            feature_names: Names of features.
            categorical_features: Indices of categorical features.
        """
        if feature_names is None:
            feature_names = list(X_train.columns)

        self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=["Not Creditworthy", "Creditworthy"],
            categorical_features=categorical_features,
            mode="classification",
            random_state=self.config.explainability.random_seed,
        )
        self.logger.info("LIME explainer initialized")

    def calculate_shap_values(
        self,
        X: pd.DataFrame,
        protected_attribute: str | None = None,
    ) -> np.ndarray:
        """Calculate SHAP values for the given data.

        Args:
            X: Features to explain.
            protected_attribute: Column to drop before explaining.

        Returns:
            Array of SHAP values.
        """
        if self._shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call setup_shap_explainer() first.")

        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        self._shap_values = self._shap_explainer.shap_values(explain_features)
        self.logger.info(f"SHAP values calculated for {len(X)} samples")

        return self._shap_values

    def generate_global_importance_plot(
        self,
        X: pd.DataFrame,
        output_path: str | Path | None = None,
        protected_attribute: str | None = None,
    ) -> Path:
        """Generate global feature importance plot using SHAP.

        Args:
            X: Features to explain.
            output_path: Path to save the plot.
            protected_attribute: Column to drop before explaining.

        Returns:
            Path to the saved plot.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        # Calculate SHAP values if not already done
        if self._shap_values is None:
            self.calculate_shap_values(X, protected_attribute)

        if output_path is None:
            output_dir = self.config.paths.reports_dir / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "feature_importance_shap.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(self._shap_values, explain_features, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Global feature importance plot saved to {output_path}")
        return output_path

    def generate_detailed_shap_plot(
        self,
        X: pd.DataFrame,
        output_path: str | Path | None = None,
        protected_attribute: str | None = None,
    ) -> Path:
        """Generate detailed SHAP summary plot showing feature value impacts.

        Args:
            X: Features to explain.
            output_path: Path to save the plot.
            protected_attribute: Column to drop before explaining.

        Returns:
            Path to the saved plot.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        # Calculate SHAP values if not already done
        if self._shap_values is None:
            self.calculate_shap_values(X, protected_attribute)

        if output_path is None:
            output_dir = self.config.paths.reports_dir / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "feature_importance_detailed.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 10))
        shap.summary_plot(self._shap_values, explain_features, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Detailed SHAP plot saved to {output_path}")
        return output_path

    def explain_individual(
        self,
        X: pd.DataFrame,
        index: int,
        prediction: int,
        shap_values: np.ndarray | None = None,
        protected_attribute: str | None = None,
    ) -> dict[str, Any]:
        """Generate explanation for a single prediction.

        Args:
            X: Features DataFrame.
            index: Index of the sample to explain.
            prediction: The model's prediction (0 or 1).
            shap_values: Pre-computed SHAP values (optional).
            protected_attribute: Column to exclude.

        Returns:
            Dictionary containing the explanation.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        if shap_values is None:
            if self._shap_values is None:
                self.calculate_shap_values(X, protected_attribute)
            shap_values = self._shap_values

        # Get this sample's values
        sample_idx = list(X.index).index(index) if index in X.index else index
        sample_values = explain_features.iloc[sample_idx]
        sample_shap = shap_values[sample_idx]

        # Rank features by importance
        feature_impacts = []
        for feat, value, sv in zip(
            explain_features.columns, sample_values.values, sample_shap
        ):
            sv_py = to_python(sv)
            value_py = to_python(value)

            feature_impacts.append(
                {
                    "feature": feat,
                    "value": value_py,
                    "shap_value": sv_py,
                    "direction": "positive" if sv_py > 0 else "negative",
                    "strength": abs(sv_py),
                }
            )

        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: x["strength"], reverse=True)

        return {
            "index": int(index),
            "prediction": int(prediction),
            "prediction_text": "approved for credit" if prediction == 1 else "denied credit",
            "feature_impacts": feature_impacts,
            "top_positive_factors": [
                f for f in feature_impacts if f["direction"] == "positive"
            ][:3],
            "top_negative_factors": [
                f for f in feature_impacts if f["direction"] == "negative"
            ][:3],
        }

    def generate_natural_language_explanation(
        self,
        explanation: dict[str, Any],
    ) -> str:
        """Convert SHAP explanation to natural language.

        Args:
            explanation: Explanation dictionary from explain_individual.

        Returns:
            Human-readable explanation string.
        """
        exp_config = self.config.explainability

        def _describe_impact(impact: dict[str, Any]) -> str:
            """Generate description for a single feature impact."""
            feat = impact["feature"]
            value = impact["value"]
            strength = impact["strength"]
            direction = "helped" if impact["direction"] == "positive" else "hurt"

            if strength > exp_config.strong_influence_threshold:
                level = "strongly"
            elif strength > exp_config.moderate_influence_threshold:
                level = "moderately"
            else:
                level = "slightly"

            # Format value nicely
            if isinstance(value, float):
                if value == int(value):
                    value_str = str(int(value))
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)

            return f"- {feat} (value: {value_str}) {level} {direction} the decision"

        lines = [
            f"Explanation for Person #{explanation['index']}:",
            f"Model Decision: {explanation['prediction_text'].upper()}",
            "",
            "Factors that HELPED the approval:",
        ]

        for impact in explanation["top_positive_factors"]:
            lines.append(_describe_impact(impact))

        if not explanation["top_positive_factors"]:
            lines.append("- No significant positive factors")

        lines.append("")
        lines.append("Factors that HURT the approval:")

        for impact in explanation["top_negative_factors"]:
            lines.append(_describe_impact(impact))

        if not explanation["top_negative_factors"]:
            lines.append("- No significant negative factors")

        return "\n".join(lines)

    def explain_samples(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        num_samples: int | None = None,
        output_dir: str | Path | None = None,
        protected_attribute: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate explanations for multiple samples.

        Args:
            X: Features DataFrame.
            predictions: Model predictions.
            num_samples: Number of samples to explain.
            output_dir: Directory to save explanation files.
            protected_attribute: Column to exclude.

        Returns:
            List of explanation dictionaries.
        """
        exp_config = self.config.explainability

        if num_samples is None:
            num_samples = exp_config.num_samples_to_explain

        if output_dir is None:
            output_dir = self.config.paths.explanations_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducible sampling
        set_random_seed(exp_config.random_seed)

        # Calculate SHAP values once
        self.calculate_shap_values(X, protected_attribute)

        # Select samples to explain
        if num_samples == -1 or num_samples >= len(X):
            sample_indices = list(X.index)
        else:
            sample_indices = random.sample(list(range(len(X))), num_samples)

        explanations = []

        for i, idx in enumerate(sample_indices):
            actual_index = X.index[idx] if idx < len(X.index) else idx

            explanation = self.explain_individual(
                X,
                actual_index,
                predictions[idx],
                self._shap_values,
                protected_attribute,
            )

            nl_explanation = self.generate_natural_language_explanation(explanation)

            # Save to file
            output_path = output_dir / f"person_{actual_index}_explanation.txt"
            with open(output_path, "w") as f:
                f.write(nl_explanation)

            explanation["natural_language"] = nl_explanation
            explanation["output_path"] = str(output_path)
            explanations.append(explanation)

            self.logger.debug(f"Explanation saved to {output_path}")

        self.logger.info(
            f"Generated {len(explanations)} explanations in {output_dir}"
        )

        return explanations

    def generate_lime_explanation(
        self,
        model: CreditModel,
        X: pd.DataFrame,
        index: int,
        protected_attribute: str | None = None,
        num_features: int | None = None,
    ) -> dict[str, Any]:
        """Generate LIME explanation for a single sample.

        Args:
            model: Trained model for predictions.
            X: Features DataFrame.
            index: Index of sample to explain.
            protected_attribute: Column to exclude.
            num_features: Number of features to show.

        Returns:
            LIME explanation dictionary.
        """
        if self._lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call setup_lime_explainer() first.")

        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        if num_features is None:
            num_features = self.config.explainability.lime_num_features

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        # Get the sample
        sample_idx = list(X.index).index(index) if index in X.index else index
        sample = explain_features.iloc[sample_idx].values

        # Get underlying model for prediction function
        underlying_model = model.get_underlying_model()

        # Generate LIME explanation
        lime_exp = self._lime_explainer.explain_instance(
            sample,
            underlying_model.predict_proba,
            num_features=num_features,
        )

        return {
            "index": index,
            "lime_explanation": lime_exp,
            "feature_weights": lime_exp.as_list(),
            "intercept": lime_exp.intercept,
        }

    def generate_waterfall_plot(
        self,
        X: pd.DataFrame,
        index: int,
        output_path: str | Path | None = None,
        protected_attribute: str | None = None,
    ) -> Path:
        """Generate SHAP waterfall plot for a single prediction.

        Args:
            X: Features DataFrame.
            index: Index of the sample to explain.
            output_path: Path to save the plot.
            protected_attribute: Column to exclude.

        Returns:
            Path to the saved plot.
        """
        if self._shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")

        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        # Calculate SHAP values if needed
        if self._shap_values is None:
            self.calculate_shap_values(X, protected_attribute)

        if output_path is None:
            output_dir = self.config.paths.explanations_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"waterfall_person_{index}.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get sample index
        sample_idx = list(X.index).index(index) if index in X.index else index

        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self._shap_values[sample_idx],
                base_values=self._shap_explainer.expected_value,
                data=explain_features.iloc[sample_idx].values,
                feature_names=list(explain_features.columns),
            ),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Waterfall plot saved to {output_path}")
        return Path(output_path)

    def generate_force_plot(
        self,
        X: pd.DataFrame,
        index: int,
        output_path: str | Path | None = None,
        protected_attribute: str | None = None,
    ) -> Path:
        """Generate SHAP force plot for a single prediction.

        Args:
            X: Features DataFrame.
            index: Index of the sample to explain.
            output_path: Path to save the plot.
            protected_attribute: Column to exclude.

        Returns:
            Path to the saved plot.
        """
        if self._shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")

        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        # Calculate SHAP values if needed
        if self._shap_values is None:
            self.calculate_shap_values(X, protected_attribute)

        if output_path is None:
            output_dir = self.config.paths.explanations_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"force_plot_person_{index}.html"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get sample index
        sample_idx = list(X.index).index(index) if index in X.index else index

        # Create force plot
        force_plot = shap.force_plot(
            self._shap_explainer.expected_value,
            self._shap_values[sample_idx],
            explain_features.iloc[sample_idx],
            matplotlib=False,
        )

        # Save as HTML
        shap.save_html(str(output_path), force_plot)

        self.logger.info(f"Force plot saved to {output_path}")
        return Path(output_path)

    def compare_explanations(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        indices: Sequence[int] | None = None,
        output_path: str | Path | None = None,
        protected_attribute: str | None = None,
    ) -> dict[str, Any]:
        """Compare explanations for multiple samples.

        Args:
            X: Features DataFrame.
            predictions: Model predictions.
            indices: List of sample indices to compare.
            output_path: Path to save comparison report.
            protected_attribute: Column to exclude.

        Returns:
            Dictionary with comparison data.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column
        if indices is None:
            indices = list(range(min(5, len(X))))

        # Calculate SHAP values if needed
        if self._shap_values is None:
            self.calculate_shap_values(X, protected_attribute)

        explanations = []
        for idx in indices:
            sample_idx = list(X.index).index(idx) if idx in X.index else idx
            exp = self.explain_individual(
                X,
                idx,
                predictions[sample_idx],
                self._shap_values,
                protected_attribute,
            )
            explanations.append(exp)

        # Find common features that influence decisions
        all_positive_features = set()
        all_negative_features = set()

        for exp in explanations:
            for factor in exp["top_positive_factors"]:
                all_positive_features.add(factor["feature"])
            for factor in exp["top_negative_factors"]:
                all_negative_features.add(factor["feature"])

        comparison = {
            "num_samples": len(indices),
            "indices": indices,
            "common_positive_features": list(all_positive_features),
            "common_negative_features": list(all_negative_features),
            "individual_explanations": explanations,
        }
        comparison["common_patterns"] = {
            "positive": comparison["common_positive_features"],
            "negative": comparison["common_negative_features"],
        }

        # Save comparison report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(comparison, f, indent=2, default=str)

            self.logger.info(f"Comparison report saved to {output_path}")

        return comparison

    def generate_interactive_report(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        output_path: str | Path | None = None,
        protected_attribute: str | None = None,
    ) -> Path:
        """Generate interactive HTML report with all explanations.

        Args:
            X: Features DataFrame.
            predictions: Model predictions.
            output_path: Path to save the report.
            protected_attribute: Column to exclude.

        Returns:
            Path to the saved report.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        # Calculate SHAP values if needed
        if self._shap_values is None:
            self.calculate_shap_values(X, protected_attribute)

        if output_path is None:
            output_dir = self.config.paths.reports_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "explainability_report.html"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create force plot for all samples
        force_plot = shap.force_plot(
            self._shap_explainer.expected_value,
            self._shap_values,
            explain_features,
            matplotlib=False,
        )

        # Save as HTML
        shap.save_html(str(output_path), force_plot)

        self.logger.info(f"Interactive report saved to {output_path}")
        return Path(output_path)

    def get_top_features_summary(
        self,
        X: pd.DataFrame,
        top_n: int | None = 10,
        *,
        n_features: int | None = None,
        protected_attribute: str | None = None,
    ) -> dict[str, Any]:
        """Get summary of top features by average absolute SHAP value.

        Args:
            X: Features DataFrame.
            top_n: Number of top features to return (positional alias).
            n_features: Optional keyword alias for the number of features.
            protected_attribute: Column to exclude.

        Returns:
            Dictionary containing top feature list and summary DataFrame.
        """
        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Prepare features
        explain_features = X.copy()
        if protected_attribute in explain_features.columns:
            explain_features = explain_features.drop(columns=[protected_attribute])

        # Calculate SHAP values if needed
        if self._shap_values is None:
            self.calculate_shap_values(X, protected_attribute)

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self._shap_values).mean(axis=0)

        # Create summary DataFrame
        summary_df = pd.DataFrame({
            "feature": explain_features.columns,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": self._shap_values.mean(axis=0),
        })

        limit = n_features if n_features is not None else top_n or 10
        summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).head(limit)
        summary_df = summary_df.reset_index(drop=True)

        return {
            "top_features": summary_df.to_dict("records"),
            "summary_df": summary_df,
        }

    def get_shap_values(self) -> np.ndarray | None:
        """Get calculated SHAP values."""
        return self._shap_values
