#!/usr/bin/env python3
"""
Evaluate all approaches by aggregating results from all folds and all companies.
"""

import os
import sys
import logging
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.evaluation.metrics import compute_classification_metrics

RESULTS_DIR = os.path.join(REPO_ROOT, "data", "results")
LABELS_PATH = os.path.join(REPO_ROOT, "data", "labels", "company_labels.csv")


FOLD_FILES = {
    "baseline": [f"baseline_fold{i}_test.csv" for i in range(1, 6)],
    "rag": [f"rag_fold{i}_test.csv" for i in range(1, 6)],
    "lora_r32": [f"lora_r32_fold{i}_test.csv" for i in range(1, 6)],
    "lora_r32_rag": [f"lora_r32_rag_fold{i}_test.csv" for i in range(1, 6)],
    "altman_zscore": ["altman_zscore_predictions.csv"],
}


def load_and_concat_folds(approach, files):
    """
    Load all available fold CSVs for an approach and concatenate them.
    """
    dfs = []

    for fname in files:
        fpath = os.path.join(RESULTS_DIR, fname)

        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["approach"] = approach
    return df


def compute_metrics(df):
    """
    Compute metrics for a dataframe.
    Uses saved predicted_label if present.
    Uses predicted_score for ROC-AUC.
    """
    y_true = df["label"].values
    y_score = df["predicted_score"].values

    # Preferred: use actual saved predicted labels
    if "predicted_label" in df.columns:
        y_pred = df["predicted_label"].values

        auc_roc = (
            round(roc_auc_score(y_true, y_score), 4)
            if len(set(y_true)) > 1
            else 0.0
        )

        metrics = {
            "auc_roc": auc_roc,
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(
                precision_score(y_true, y_pred, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true, y_pred, zero_division=0), 4
            ),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
        }

    else:
        # fallback
        metrics = compute_classification_metrics(
            y_true,
            y_score,
            threshold=50.0,
        )

    return metrics


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_folds")

    if not os.path.exists(LABELS_PATH):
        logger.error("Labels file not found: %s", LABELS_PATH)
        return

    labels_df = pd.read_csv(LABELS_PATH)

    all_results = []

    # Load + merge each approach
    for approach, files in FOLD_FILES.items():
        df = load_and_concat_folds(approach, files)

        if df.empty:
            logger.warning("No results found for %s", approach)
            continue

        merged = df.merge(labels_df, on="ticker", how="inner")

        if merged.empty:
            logger.warning("Merge empty for %s", approach)
            continue

        all_results.append(merged)
        logger.info(
            "Loaded and merged %d rows for %s",
            len(merged),
            approach,
        )

    if not all_results:
        logger.error("No results to evaluate.")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # Comparison table
    comparison_rows = []

    for approach in FOLD_FILES.keys():
        df = results_df[results_df["approach"] == approach]

        if df.empty:
            continue

        metrics = compute_metrics(df)
        metrics["approach"] = approach
        comparison_rows.append(metrics)

    comparison = pd.DataFrame(comparison_rows).set_index("approach")

    print("\n" + "=" * 70)
    print("  EVALUATION OF ALL FOLDS AND COMPANIES")
    print("=" * 70)
    print(comparison)
    print("=" * 70)

    # Detailed metrics
    for approach in comparison.index:
        df = results_df[results_df["approach"] == approach]
        metrics = compute_metrics(df)

        print(f"\nMetrics for {approach}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()