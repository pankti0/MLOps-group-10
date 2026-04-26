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
"""


import os
import numpy as np
import pandas as pd
import logging
from src.evaluation.metrics import compute_classification_metrics

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate all credit risk detection approaches.")
    parser.add_argument('--results_dir', type=str, default=None, help='Directory for results output')
    parser.add_argument('--labels', type=str, default=None, help='Path to labels CSV')
    parser.add_argument('--sections_dir', type=str, default=None, help='Directory for processed sections')
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging.')
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb_run', type=str, default=None, help='W&B run name')
    return parser.parse_args()

def get_repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_all_configs():
    # Dummy implementation, replace with actual config loading
    return {"eval": {}, "data": {}}

def save_zscore_predictions(zscore_path):
    # Dummy implementation, replace with actual logic
    import pandas as pd
    df = pd.DataFrame({"ticker": [], "zscore": []})
    df.to_csv(zscore_path, index=False)
    return df

def init_run(project, run_name, config):
    # Dummy implementation for W&B init
    pass

def run_full_evaluation(results_dir, labels_path, sections_dir):
    # Dummy implementation for full evaluation
    return {}

def main():
    approaches = [
            "baseline", "rag", "lora_r32", "rag_lora_r32"
    ]
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_all")
    args = parse_args()
    repo_root = get_repo_root()

    # Load merged configs to resolve default paths
    configs = load_all_configs()
    eval_config = configs.get("eval", {})
    data_config = configs.get("data", {})

    # Resolve paths from args or config defaults
    results_dir = args.results_dir or eval_config.get(
        "output", {}
    ).get("results_dir") or os.path.join(repo_root, "data", "results")

    labels_path = args.labels or data_config.get(
        "output_paths", {}
    ).get("labels") or os.path.join(repo_root, "data", "labels", "company_labels.csv")

    sections_dir = args.sections_dir or data_config.get(
        "output_paths", {}
    ).get("processed") or os.path.join(repo_root, "data", "processed")

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
    comparison = all_metrics.pop("comparison", None) if isinstance(all_metrics, dict) else None

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)

    display_names = {
        "lora": "RAG + LORA",
    }

    if isinstance(all_metrics, dict):
        for approach, metrics in all_metrics.items():
            section_title = display_names.get(approach, approach.upper())
            print(f"\n  [{section_title}]")
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

    display_names = {
        "lora": "RAG + LORA",
    }

    for approach, metrics in all_metrics.items():
        section_title = display_names.get(approach, approach.upper())
        print(f"\n  [{section_title}]")
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
