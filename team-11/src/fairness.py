"""Fairness analysis and mitigation module.

This module handles all fairness-related operations including:
- Pre-processing: Reweighing samples
- In-processing: Fairness-aware training constraints
- Post-processing: Threshold adjustment per group
- Metrics: Disparate impact, statistical parity, equalized odds, PPV/FNR/FPR parity
- Visualization: Fairness metric plots
- Reporting: Comprehensive fairness reports
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

from config.config import Config, FairnessConfig, get_config
from src.utils import get_logger, format_confusion_matrix

# Set plotting style
sns.set_style("whitegrid")


@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""

    disparate_impact: float = 0.0
    statistical_parity_difference: float = 0.0
    equalized_odds_difference: float = 0.0
    average_odds_difference: float = 0.0

    # Group-specific metrics
    privileged_accuracy: float = 0.0
    unprivileged_accuracy: float = 0.0
    privileged_positive_rate: float = 0.0
    unprivileged_positive_rate: float = 0.0

    # Additional fairness metrics
    ppv_parity: float = 0.0  # Positive Predictive Value parity
    fnr_parity: float = 0.0  # False Negative Rate parity
    fpr_parity: float = 0.0  # False Positive Rate parity
    tpr_privileged: float = 0.0  # True Positive Rate for privileged
    tpr_unprivileged: float = 0.0  # True Positive Rate for unprivileged
    fpr_privileged: float = 0.0  # False Positive Rate for privileged
    fpr_unprivileged: float = 0.0  # False Positive Rate for unprivileged

    # Confusion matrices
    privileged_confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2))
    )
    unprivileged_confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2))
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "disparate_impact": self.disparate_impact,
            "statistical_parity_difference": self.statistical_parity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "average_odds_difference": self.average_odds_difference,
            "privileged_accuracy": self.privileged_accuracy,
            "unprivileged_accuracy": self.unprivileged_accuracy,
            "privileged_positive_rate": self.privileged_positive_rate,
            "unprivileged_positive_rate": self.unprivileged_positive_rate,
            "ppv_parity": self.ppv_parity,
            "fnr_parity": self.fnr_parity,
            "fpr_parity": self.fpr_parity,
            "tpr_privileged": self.tpr_privileged,
            "tpr_unprivileged": self.tpr_unprivileged,
            "fpr_privileged": self.fpr_privileged,
            "fpr_unprivileged": self.fpr_unprivileged,
            "privileged_confusion_matrix": self.privileged_confusion_matrix.tolist(),
            "unprivileged_confusion_matrix": self.unprivileged_confusion_matrix.tolist(),
        }

    def passes_thresholds(
        self,
        di_threshold: float = 0.80,
        spd_threshold: float = 0.05,
        eod_threshold: float = 0.10,
    ) -> dict[str, bool]:
        """Check if metrics pass fairness thresholds.

        Args:
            di_threshold: Minimum disparate impact ratio.
            spd_threshold: Maximum statistical parity difference.
            eod_threshold: Maximum equalized odds difference.

        Returns:
            Dictionary of metric names to pass/fail status.
        """
        return {
            "disparate_impact": self.disparate_impact >= di_threshold,
            "statistical_parity_difference": abs(self.statistical_parity_difference)
            <= spd_threshold,
            "equalized_odds_difference": abs(self.equalized_odds_difference)
            <= eod_threshold,
        }


class FairnessAnalyzer:
    """Handles fairness analysis and mitigation techniques."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the FairnessAnalyzer.

        Args:
            config: Configuration object. Uses global config if None.
        """
        self.config = config or get_config()
        self.logger = get_logger("fairness")
        self._sample_weights: np.ndarray | None = None

    def apply_reweighing(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        protected_attribute: str | None = None,
    ) -> np.ndarray:
        """Apply reweighing pre-processing to training data.

        Reweighing adjusts sample weights to achieve demographic parity
        before training, without modifying the actual data.

        Args:
            X_train: Training features (must include protected attribute).
            y_train: Training labels.
            protected_attribute: Name of protected attribute column.

        Returns:
            Array of sample weights for training.
        """
        fairness_config = self.config.fairness

        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        if protected_attribute not in X_train.columns:
            self.logger.error(
                f"Protected attribute '{protected_attribute}' not in features"
            )
            raise ValueError(
                f"Protected attribute '{protected_attribute}' not in features"
            )

        # Create AIF360 BinaryLabelDataset
        train_df = pd.concat([X_train, y_train], axis=1)

        train_bld = BinaryLabelDataset(
            df=train_df,
            label_names=[self.config.model.target_column],
            protected_attribute_names=[protected_attribute],
        )

        # Apply reweighing
        reweigher = Reweighing(
            unprivileged_groups=fairness_config.unprivileged_groups,
            privileged_groups=fairness_config.privileged_groups,
        )

        train_bld_transformed = reweigher.fit_transform(train_bld)
        self._sample_weights = train_bld_transformed.instance_weights

        self.logger.info(
            f"Reweighing applied: weights range [{self._sample_weights.min():.3f}, "
            f"{self._sample_weights.max():.3f}]"
        )

        return self._sample_weights

    def apply_threshold_adjustment(
        self,
        y_prob: np.ndarray,
        protected_values: np.ndarray,
        threshold_privileged: float | None = None,
        threshold_unprivileged: float | None = None,
    ) -> np.ndarray:
        """Apply group-specific threshold adjustment (post-processing).

        Different thresholds for privileged vs unprivileged groups
        can help equalize positive prediction rates.

        Args:
            y_prob: Predicted probabilities.
            protected_values: Protected attribute values (0=privileged, 1=unprivileged).
            threshold_privileged: Threshold for privileged group.
            threshold_unprivileged: Threshold for unprivileged group.

        Returns:
            Adjusted binary predictions.
        """
        fairness_config = self.config.fairness

        if threshold_privileged is None:
            threshold_privileged = fairness_config.threshold_privileged
        if threshold_unprivileged is None:
            threshold_unprivileged = fairness_config.threshold_unprivileged

        y_pred_adjusted = np.zeros(len(y_prob), dtype=int)

        # Apply different thresholds per group
        privileged_mask = protected_values == 0
        unprivileged_mask = protected_values == 1

        y_pred_adjusted[privileged_mask] = (
            y_prob[privileged_mask] >= threshold_privileged
        ).astype(int)
        y_pred_adjusted[unprivileged_mask] = (
            y_prob[unprivileged_mask] >= threshold_unprivileged
        ).astype(int)

        self.logger.info(
            f"Threshold adjustment applied: "
            f"privileged={threshold_privileged}, unprivileged={threshold_unprivileged}"
        )

        return y_pred_adjusted

    def calculate_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        protected_attribute: str | None = None,
    ) -> FairnessMetrics:
        """Calculate comprehensive fairness metrics.

        Args:
            X_test: Test features (must include protected attribute).
            y_test: True labels.
            y_pred: Predicted labels.
            protected_attribute: Name of protected attribute column.

        Returns:
            FairnessMetrics object with all metrics.
        """
        fairness_config = self.config.fairness

        if protected_attribute is None:
            protected_attribute = self.config.cdi.proxy_column

        # Create AIF360 datasets
        test_df = pd.concat([X_test, y_test], axis=1)

        test_bld = BinaryLabelDataset(
            df=test_df,
            label_names=[self.config.model.target_column],
            protected_attribute_names=[protected_attribute],
        )

        # Create predicted dataset
        test_bld_pred = test_bld.copy()
        test_bld_pred.labels = y_pred.reshape(-1, 1)

        # Calculate AIF360 metrics
        metric = ClassificationMetric(
            test_bld,
            test_bld_pred,
            unprivileged_groups=fairness_config.unprivileged_groups,
            privileged_groups=fairness_config.privileged_groups,
        )

        # Get protected attribute values
        protected_values = X_test[protected_attribute].values

        # Calculate group-specific metrics
        priv_mask = protected_values == 0
        unpriv_mask = protected_values == 1

        # Calculate confusion matrices
        cm_priv = (
            confusion_matrix(y_test.values[priv_mask], y_pred[priv_mask])
            if priv_mask.sum() > 0
            else np.zeros((2, 2))
        )
        cm_unpriv = (
            confusion_matrix(y_test.values[unpriv_mask], y_pred[unpriv_mask])
            if unpriv_mask.sum() > 0
            else np.zeros((2, 2))
        )

        # Extract confusion matrix values
        # cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
        def safe_divide(num, denom):
            return num / denom if denom > 0 else 0.0

        # Privileged group rates
        tn_p, fp_p, fn_p, tp_p = cm_priv[0,0], cm_priv[0,1], cm_priv[1,0], cm_priv[1,1]
        tpr_priv = safe_divide(tp_p, (tp_p + fn_p))  # TPR = TP / (TP + FN)
        fpr_priv = safe_divide(fp_p, (fp_p + tn_p))  # FPR = FP / (FP + TN)
        fnr_priv = safe_divide(fn_p, (fn_p + tp_p))  # FNR = FN / (FN + TP)
        ppv_priv = safe_divide(tp_p, (tp_p + fp_p))  # PPV = TP / (TP + FP)

        # Unprivileged group rates
        tn_u, fp_u, fn_u, tp_u = cm_unpriv[0,0], cm_unpriv[0,1], cm_unpriv[1,0], cm_unpriv[1,1]
        tpr_unpriv = safe_divide(tp_u, (tp_u + fn_u))
        fpr_unpriv = safe_divide(fp_u, (fp_u + tn_u))
        fnr_unpriv = safe_divide(fn_u, (fn_u + tp_u))
        ppv_unpriv = safe_divide(tp_u, (tp_u + fp_u))

        metrics = FairnessMetrics(
            disparate_impact=metric.disparate_impact(),
            statistical_parity_difference=metric.statistical_parity_difference(),
            equalized_odds_difference=metric.equalized_odds_difference(),
            average_odds_difference=metric.average_odds_difference(),
            privileged_accuracy=(
                (y_pred[priv_mask] == y_test.values[priv_mask]).mean()
                if priv_mask.sum() > 0
                else 0.0
            ),
            unprivileged_accuracy=(
                (y_pred[unpriv_mask] == y_test.values[unpriv_mask]).mean()
                if unpriv_mask.sum() > 0
                else 0.0
            ),
            privileged_positive_rate=(
                y_pred[priv_mask].mean() if priv_mask.sum() > 0 else 0.0
            ),
            unprivileged_positive_rate=(
                y_pred[unpriv_mask].mean() if unpriv_mask.sum() > 0 else 0.0
            ),
            ppv_parity=abs(ppv_priv - ppv_unpriv),
            fnr_parity=abs(fnr_priv - fnr_unpriv),
            fpr_parity=abs(fpr_priv - fpr_unpriv),
            tpr_privileged=tpr_priv,
            tpr_unprivileged=tpr_unpriv,
            fpr_privileged=fpr_priv,
            fpr_unprivileged=fpr_unpriv,
            privileged_confusion_matrix=cm_priv,
            unprivileged_confusion_matrix=cm_unpriv,
        )

        return metrics

    def log_metrics(self, metrics: FairnessMetrics, title: str = "Fairness Metrics") -> None:
        """Log fairness metrics in a readable format.

        Args:
            metrics: FairnessMetrics object.
            title: Title for the log output.
        """
        fairness_config = self.config.fairness
        results = metrics.passes_thresholds(
            di_threshold=fairness_config.target_disparate_impact_ratio,
            spd_threshold=fairness_config.target_statistical_parity_diff,
            eod_threshold=fairness_config.target_equalized_odds_diff,
        )

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{title}")
        self.logger.info(f"{'='*60}")

        # Disparate Impact
        status = "✓ PASS" if results["disparate_impact"] else "✗ FAIL"
        self.logger.info(
            f"Disparate Impact Ratio: {metrics.disparate_impact:.4f} "
            f"(target >= {fairness_config.target_disparate_impact_ratio}) [{status}]"
        )

        # Statistical Parity
        status = "✓ PASS" if results["statistical_parity_difference"] else "✗ FAIL"
        self.logger.info(
            f"Statistical Parity Diff: {metrics.statistical_parity_difference:.4f} "
            f"(target < {fairness_config.target_statistical_parity_diff}) [{status}]"
        )

        # Equalized Odds
        status = "✓ PASS" if results["equalized_odds_difference"] else "✗ FAIL"
        self.logger.info(
            f"Equalized Odds Diff: {metrics.equalized_odds_difference:.4f} "
            f"(target < {fairness_config.target_equalized_odds_diff}) [{status}]"
        )

        self.logger.info(f"\nGroup Metrics:")
        self.logger.info(
            f"  Privileged accuracy: {metrics.privileged_accuracy:.4f}, "
            f"positive rate: {metrics.privileged_positive_rate:.4f}"
        )
        self.logger.info(
            f"  Unprivileged accuracy: {metrics.unprivileged_accuracy:.4f}, "
            f"positive rate: {metrics.unprivileged_positive_rate:.4f}"
        )

        self.logger.info(f"{'='*60}\n")

    def plot_fairness_metrics(
        self,
        metrics: FairnessMetrics,
        output_path: str | Path | None = None,
        *,
        save_path: str | Path | None = None,
    ) -> Path:
        """Generate fairness metrics visualization.

        Args:
            metrics: FairnessMetrics object.
            output_path: Path to save the plot.
            save_path: Optional alias for output_path.

        Returns:
            Path to the saved plot.
        """
        if save_path is not None:
            output_path = save_path

        if output_path is None:
            output_dir = self.config.paths.reports_dir / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "fairness_metrics.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Accuracy comparison
        groups = ["Privileged", "Unprivileged"]
        accuracies = [metrics.privileged_accuracy, metrics.unprivileged_accuracy]
        axes[0, 0].bar(groups, accuracies, color=["#3498db", "#e74c3c"])
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Accuracy by Group")
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

        # Plot 2: Positive prediction rates
        pos_rates = [metrics.privileged_positive_rate, metrics.unprivileged_positive_rate]
        axes[0, 1].bar(groups, pos_rates, color=["#3498db", "#e74c3c"])
        axes[0, 1].set_ylabel("Positive Prediction Rate")
        axes[0, 1].set_title("Positive Prediction Rate by Group")
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(pos_rates):
            axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

        # Plot 3: Fairness metrics comparison
        fairness_metrics_names = ["Disparate\nImpact", "Stat. Parity\nDiff", "Eq. Odds\nDiff"]
        fairness_values = [
            metrics.disparate_impact,
            abs(metrics.statistical_parity_difference),
            abs(metrics.equalized_odds_difference),
        ]
        targets = [0.80, 0.05, 0.10]
        x_pos = np.arange(len(fairness_metrics_names))

        bars = axes[1, 0].bar(x_pos, fairness_values, color=["#3498db", "#e74c3c", "#2ecc71"])
        axes[1, 0].axhline(y=targets[0], color="gray", linestyle="--", alpha=0.5, label="Target")
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(fairness_metrics_names)
        axes[1, 0].set_ylabel("Metric Value")
        axes[1, 0].set_title("Fairness Metrics vs Targets")
        for i, (v, t) in enumerate(zip(fairness_values, targets)):
            axes[1, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

        # Plot 4: TPR and FPR comparison
        tpr_values = [metrics.tpr_privileged, metrics.tpr_unprivileged]
        fpr_values = [metrics.fpr_privileged, metrics.fpr_unprivileged]

        x = np.arange(len(groups))
        width = 0.35

        axes[1, 1].bar(x - width/2, tpr_values, width, label="TPR", color="#2ecc71")
        axes[1, 1].bar(x + width/2, fpr_values, width, label="FPR", color="#e74c3c")
        axes[1, 1].set_ylabel("Rate")
        axes[1, 1].set_title("True Positive Rate vs False Positive Rate")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(groups)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Fairness metrics plot saved to {output_path}")
        return Path(output_path)

    def generate_fairness_report(
        self,
        metrics: FairnessMetrics,
        X: pd.DataFrame | None = None,
        y_true: pd.Series | None = None,
        y_pred: np.ndarray | None = None,
        output_path: str | Path | None = None,
        *,
        save_path: str | Path | None = None,
    ) -> Path:
        """Generate comprehensive fairness report.

        Args:
            metrics: FairnessMetrics object.
            X: Feature data used for evaluation (optional).
            y_true: True labels (optional).
            y_pred: Predicted labels (optional).
            output_path: Path to save the report.
            save_path: Optional alias for output_path.

        Returns:
            Path to the saved report.
        """
        if save_path is not None:
            output_path = save_path

        if output_path is None:
            output_dir = self.config.paths.reports_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "fairness_report.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        fairness_config = self.config.fairness
        thresholds = {
            "disparate_impact": fairness_config.target_disparate_impact_ratio,
            "statistical_parity_difference": fairness_config.target_statistical_parity_diff,
            "equalized_odds_difference": fairness_config.target_equalized_odds_diff,
        }
        passes = metrics.passes_thresholds(
            di_threshold=thresholds["disparate_impact"],
            spd_threshold=thresholds["statistical_parity_difference"],
            eod_threshold=thresholds["equalized_odds_difference"],
        )

        report = {
            "summary": {
                "num_samples": int(len(X)) if X is not None else None,
                "overall_status": "PASS" if all(passes.values()) else "FAIL",
                "thresholds": thresholds,
                "passes_thresholds": passes,
            },
            "metrics": metrics.to_dict(),
            "confusion_matrices": {
                "privileged": metrics.privileged_confusion_matrix.astype(float).tolist(),
                "unprivileged": metrics.unprivileged_confusion_matrix.astype(float).tolist(),
            },
            "group_statistics": {
                "privileged": {
                    "accuracy": metrics.privileged_accuracy,
                    "positive_rate": metrics.privileged_positive_rate,
                    "tpr": metrics.tpr_privileged,
                    "fpr": metrics.fpr_privileged,
                },
                "unprivileged": {
                    "accuracy": metrics.unprivileged_accuracy,
                    "positive_rate": metrics.unprivileged_positive_rate,
                    "tpr": metrics.tpr_unprivileged,
                    "fpr": metrics.fpr_unprivileged,
                },
            },
        }

        if X is not None:
            report["input_metadata"] = {
                "columns": list(X.columns),
                "num_rows": int(len(X)),
            }
        if y_true is not None:
            report["input_metadata"] = report.get("input_metadata", {})
            report["input_metadata"]["class_balance"] = {
                "positive": int((y_true == 1).sum()),
                "negative": int((y_true == 0).sum()),
            }
        if y_pred is not None:
            report["predictions"] = {
                "positive_rate": float(np.mean(y_pred)),
            }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=float)

        self.logger.info(f"Fairness report saved to {output_path}")
        return Path(output_path)

    def get_sample_weights(self) -> np.ndarray | None:
        """Get the sample weights from reweighing."""
        return self._sample_weights
