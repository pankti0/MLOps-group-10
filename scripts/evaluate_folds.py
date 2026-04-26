#!/usr/bin/env python3
"""
Evaluate all approaches by aggregating results from all folds and all companies.
"""

import os
import sys
import logging
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.evaluation.metrics import compute_classification_metrics, compare_approaches

RESULTS_DIR = os.path.join(REPO_ROOT, "data", "results")
LABELS_PATH = os.path.join(REPO_ROOT, "data", "labels", "company_labels.csv")


FOLD_FILES = {
    "baseline": [f"baseline_fold{i}_test.csv" for i in range(1, 6)],
    "rag": [f"rag_fold{i}_test.csv" for i in range(1, 6)],
    "lora_r32": [f"lora_r32_fold{i}_test.csv" for i in range(1, 6)],
    "altman_zscore": ["altman_zscore_predictions.csv"],
}


def load_and_concat_folds(approach, files):
    dfs = []
    for fname in files:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["approach"] = approach
        return df
    return pd.DataFrame()


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_folds")

    labels_df = pd.read_csv(LABELS_PATH)
    all_results = []
    for approach, files in FOLD_FILES.items():
        df = load_and_concat_folds(approach, files)
        if not df.empty:
            merged = df.merge(labels_df, on="ticker", how="inner")
            all_results.append(merged)
            logger.info(f"Loaded and merged {len(merged)} rows for {approach}")
        else:
            logger.warning(f"No results found for {approach}")

    if not all_results:
        logger.error("No results to evaluate.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    comparison = compare_approaches({a: results_df[results_df["approach"] == a] for a in FOLD_FILES.keys()})

    print("\n" + "=" * 70)
    print("  EVALUATION OF ALL FOLDS AND COMPANIES")
    print("=" * 70)
    print(comparison)
    print("=" * 70)

    # Per-approach metrics
    for approach in FOLD_FILES.keys():
        df = results_df[results_df["approach"] == approach]
        if not df.empty and "label" in df.columns and "predicted_score" in df.columns:
            metrics = compute_classification_metrics(df["label"].values, df["predicted_score"].values)
            print(f"\nMetrics for {approach}:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
