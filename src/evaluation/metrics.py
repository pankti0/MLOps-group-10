"""
Classification and evaluation metrics for credit risk prediction.

Exports:
    compute_classification_metrics(labels, predicted_scores, threshold) -> dict
    compute_roc_curve(labels, predicted_scores) -> Tuple[array, array, float]
    compare_approaches(results) -> pd.DataFrame
    plot_roc_curves(results, output_path) -> None
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # non-interactive backend safe for cluster/CI
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# Colours used for each approach in ROC plots
_APPROACH_COLORS: Dict[str, str] = {
    "baseline": "#1f77b4",
    "rag": "#ff7f0e",
    "lora": "#2ca02c",
    "altman_zscore": "#d62728",
}
_DEFAULT_COLOR = "#9467bd"


def compute_classification_metrics(
    labels: np.ndarray,
    predicted_scores: np.ndarray,
    threshold: float = 70.0,
) -> dict:
    """Compute a standard suite of binary classification metrics.

    Args:
        labels: Ground-truth binary labels (0 = low risk, 1 = high risk).
        predicted_scores: Continuous risk scores in [0, 100] where higher
            values indicate higher predicted risk.
        threshold: Score threshold above which a prediction is classified
            as high risk (label=1). Defaults to 70.0.

    Returns:
        Dictionary with keys:
            - ``auc_roc`` (float)
            - ``f1`` (float)
            - ``precision`` (float)
            - ``recall`` (float)
            - ``accuracy`` (float)
            - ``precision_at_high_risk`` (float): precision on the top
              tercile of predicted scores
    """
    labels = np.asarray(labels, dtype=int)
    predicted_scores = np.asarray(predicted_scores, dtype=float)

    if len(np.unique(labels)) < 2:
        logger.warning(
            "Only one class present in labels. AUC-ROC is undefined; returning 0.0."
        )
        auc_roc = 0.0
    else:
        auc_roc = float(roc_auc_score(labels, predicted_scores))

    predicted_labels = (predicted_scores >= threshold).astype(int)

    f1 = float(f1_score(labels, predicted_labels, zero_division=0))
    precision = float(precision_score(labels, predicted_labels, zero_division=0))
    recall = float(recall_score(labels, predicted_labels, zero_division=0))
    accuracy = float(accuracy_score(labels, predicted_labels))

    # Precision at high risk: restrict to the top tercile by predicted score
    tercile_cutoff = np.percentile(predicted_scores, 66.67)
    high_risk_mask = predicted_scores >= tercile_cutoff
    if high_risk_mask.sum() > 0:
        prec_high_risk = float(
            precision_score(
                labels[high_risk_mask],
                predicted_labels[high_risk_mask],
                zero_division=0,
            )
        )
    else:
        prec_high_risk = 0.0

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "precision_at_high_risk": round(prec_high_risk, 4),
    }
    logger.info("Metrics: %s", metrics)
    return metrics


def compute_roc_curve(
    labels: np.ndarray,
    predicted_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve points and AUC.

    Args:
        labels: Ground-truth binary labels.
        predicted_scores: Continuous risk scores in [0, 100].

    Returns:
        Tuple of (fpr, tpr, auc_value) where fpr and tpr are numpy arrays.
    """
    labels = np.asarray(labels, dtype=int)
    predicted_scores = np.asarray(predicted_scores, dtype=float)

    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present — ROC curve is degenerate.")
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc_value = 0.0
        return fpr, tpr, auc_value

    fpr, tpr, _ = roc_curve(labels, predicted_scores)
    auc_value = float(auc(fpr, tpr))
    return fpr, tpr, auc_value


def compare_approaches(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a comparison table of metrics across approaches.

    Args:
        results: Mapping of approach name → predictions DataFrame.
            Each DataFrame must have columns ``predicted_score`` and
            ``label`` (ground-truth).

    Returns:
        DataFrame with one row per approach and metric columns.
    """
    rows = []
    for approach, df in results.items():
        if df.empty:
            logger.warning("Empty DataFrame for approach '%s', skipping.", approach)
            continue
        if "label" not in df.columns or "predicted_score" not in df.columns:
            logger.warning(
                "DataFrame for '%s' missing required columns ('label', 'predicted_score').",
                approach,
            )
            continue

        metrics = compute_classification_metrics(
            df["label"].values,
            df["predicted_score"].values,
        )
        metrics["approach"] = approach
        rows.append(metrics)

    if not rows:
        logger.warning("No valid approach data to compare.")
        return pd.DataFrame()

    comparison = pd.DataFrame(rows).set_index("approach")
    logger.info("Comparison table built for approaches: %s", list(comparison.index))
    return comparison


def plot_roc_curves(results: Dict[str, pd.DataFrame], output_path: str) -> None:
    """Plot ROC curves for multiple approaches on a single figure and save.

    Args:
        results: Mapping of approach name → predictions DataFrame.
            Each DataFrame must have columns ``predicted_score`` and
            ``label`` (ground-truth).
        output_path: File path where the PNG figure will be saved.
            Parent directories are created if they do not exist.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")

    plotted_any = False
    for approach, df in results.items():
        if df.empty or "label" not in df.columns or "predicted_score" not in df.columns:
            logger.warning("Skipping ROC plot for approach '%s' — missing data.", approach)
            continue

        try:
            fpr, tpr, auc_value = compute_roc_curve(
                df["label"].values, df["predicted_score"].values
            )
            color = _APPROACH_COLORS.get(approach, _DEFAULT_COLOR)
            ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=2,
                label=f"{approach} (AUC={auc_value:.3f})",
            )
            plotted_any = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to plot ROC for '%s': %s", approach, exc)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Credit Risk Detection", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)

    if not plotted_any:
        logger.warning("No valid approach data to plot.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", output_path)
