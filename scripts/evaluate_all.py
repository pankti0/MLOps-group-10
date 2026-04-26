#!/usr/bin/env python3
"""
Run evaluation for all approaches.
"""

import os
import sys
import logging
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.evaluation.altman_zscore import save_zscore_predictions
from src.evaluation.evaluator import run_full_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--sections_dir", type=str, default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_all")

    args = parse_args()

    results_dir = args.results_dir or os.path.join(REPO_ROOT, "data", "results")
    labels_path = args.labels or os.path.join(
        REPO_ROOT, "data", "labels", "company_labels.csv"
    )
    sections_dir = args.sections_dir or os.path.join(
        REPO_ROOT, "data", "processed"
    )

    os.makedirs(results_dir, exist_ok=True)

    logger.info("Results directory  : %s", results_dir)
    logger.info("Labels path        : %s", labels_path)
    logger.info("Sections directory : %s", sections_dir)

    # Altman
    zscore_path = os.path.join(results_dir, "altman_zscore_predictions.csv")
    logger.info("Generating Altman Z-Score predictions → %s", zscore_path)

    zscore_df = save_zscore_predictions(zscore_path)

    logger.info(
        "Z-Score predictions saved: %d companies.",
        len(zscore_df),
    )

    # Full eval
    logger.info("Running full evaluation pipeline...")

    run_full_evaluation(
        results_dir=results_dir,
        labels_path=labels_path,
        sections_dir=sections_dir,
    )

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()