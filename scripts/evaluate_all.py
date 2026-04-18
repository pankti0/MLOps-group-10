#!/usr/bin/env python3
"""
evaluate_all.py

Runs the complete evaluation pipeline for all credit risk detection approaches:
    - Baseline prompting
    - RAG
    - LoRA fine-tuning
    - Altman Z-Score (generated if predictions are absent)

Outputs:
    data/results/altman_zscore_predictions.csv  (generated)
    data/results/metrics_summary.csv
    data/results/roc_curves.png
    Console: formatted comparison table

Usage:
    python scripts/evaluate_all.py [--results-dir data/results]
                                       [--labels data/labels/company_labels.csv]
                                       [--sections-dir data/processed]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.evaluation.altman_zscore import save_zscore_predictions
from src.evaluation.evaluator import run_full_evaluation
from src.utils.config_loader import get_repo_root, load_all_configs
from src.utils.wandb_logger import init_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate all credit risk detection approaches."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing {approach}_predictions.csv files.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to company_labels.csv.",
    )
    parser.add_argument(
        "--sections-dir",
        type=str,
        default=None,
        help="Directory containing {ticker}_sections.json files.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="credit-risk-detection",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default="evaluation",
        help="W&B run name.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    return parser.parse_args()


def main() -> None:
   
    args = parse_args()
    repo_root = get_repo_root()

    # Load merged configs to resolve default paths
    configs = load_all_configs()
    eval_config = configs.get("eval", {})
    data_config = configs.get("data", {})

    # Resolve paths from args or config defaults
    results_dir = args.results_dir or eval_config.get(
        "output", {}
    ).get("results_dir") or str(repo_root / "data" / "results")

    labels_path = args.labels or data_config.get(
        "output_paths", {}
    ).get("labels") or str(repo_root / "data" / "labels" / "company_labels.csv")

    sections_dir = args.sections_dir or data_config.get(
        "output_paths", {}
    ).get("processed") or str(repo_root / "data" / "processed")

    os.makedirs(results_dir, exist_ok=True)

    logger.info("Results directory  : %s", results_dir)
    logger.info("Labels path        : %s", labels_path)
    logger.info("Sections directory : %s", sections_dir)

    # 1. Generate Altman Z-Score predictions 
    zscore_path = os.path.join(results_dir, "altman_zscore_predictions.csv")
    logger.info("Generating Altman Z-Score predictions → %s", zscore_path)
    zscore_df = save_zscore_predictions(zscore_path)
    logger.info("Z-Score predictions saved: %d companies.", len(zscore_df))

    # 2. Initialize W&B (optional)
    if not args.no_wandb:
        init_run(
            project=args.wandb_project,
            run_name=args.wandb_run,
            config={
                "results_dir": results_dir,
                "labels_path": labels_path,
                "sections_dir": sections_dir,
            },
        )

    # 3. Run full evaluation 
    logger.info("Running full evaluation pipeline...")
    all_metrics = run_full_evaluation(
        results_dir=results_dir,
        labels_path=labels_path,
        sections_dir=sections_dir,
    )

    #  4. Print final summary 
    comparison = all_metrics.pop("comparison", None)

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)

    for approach, metrics in all_metrics.items():
        print(f"\n  [{approach.upper()}]")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for sub_key, sub_val in value.items():
                        print(f"      {sub_key:35s}: {sub_val}")
                else:
                    print(f"    {key:37s}: {value}")

    print(f"  Outputs saved to: {results_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
