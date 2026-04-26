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

        "--no-wandb",
        import numpy as np
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

        # 3. Cross-validation evaluation
        folds = [1, 2, 3, 4, 5]
        approaches = [
            "baseline", "rag", "lora_r8", "lora_r16", "lora_r32",
            "rag_lora_r8", "rag_lora_r16", "rag_lora_r32"
        ]
        metrics_summary = []
        all_results = {}

        # Load ground truth labels
        import pandas as pd
        labels_df = pd.read_csv(labels_path)

        for approach in approaches:
            fold_metrics = []
            for fold in folds:
                # Determine file name
                if approach.startswith("rag_lora_"):
                    lora_variant = approach.replace("rag_", "")
                    pred_file = f"rag_fold{fold}_test.csv"  # Default for rag
                    # Try variant file name
                    alt_file = f"rag_{lora_variant}_fold{fold}_test.csv"
                    path = os.path.join(results_dir, alt_file)
                    if not os.path.exists(path):
                        path = os.path.join(results_dir, pred_file)
                else:
                    path = os.path.join(results_dir, f"{approach}_fold{fold}_test.csv")
                if not os.path.exists(path):
                    logger.warning(f"Missing predictions for {approach} fold {fold}: {path}")
                    continue
                df = pd.read_csv(path)
                # Merge with labels
                if "ticker" not in df.columns:
                    logger.warning(f"No 'ticker' column in {path}, skipping.")
                    continue
                merged = df.merge(labels_df[["ticker", "label"]], on="ticker", how="inner")
                if merged.empty:
                    logger.warning(f"No matching tickers for {approach} fold {fold} after merge.")
                    continue
                # Compute metrics
                from src.evaluation.metrics import compute_classification_metrics
                metrics = compute_classification_metrics(
                    merged["label"].values, merged["predicted_score"].values
                )
                metrics["fold"] = fold
                fold_metrics.append(metrics)
            if fold_metrics:
                # Aggregate
                keys = fold_metrics[0].keys()
                agg = {k: np.mean([m[k] for m in fold_metrics if k in m]) for k in keys if k != "fold"}
                agg_std = {f"{k}_std": np.std([m[k] for m in fold_metrics if k in m]) for k in keys if k != "fold"}
                agg["folds"] = len(fold_metrics)
                agg.update(agg_std)
                agg["approach"] = approach
                metrics_summary.append(agg)
                all_results[approach] = fold_metrics

        # 4. Print and save summary
        summary_df = pd.DataFrame(metrics_summary)
        summary_path = os.path.join(results_dir, "metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 70)
        print("  CROSS-VALIDATION EVALUATION COMPLETE")
        print("=" * 70)
        print(summary_df)
        print(f"\nMetrics summary saved to {summary_path}\n")

    if __name__ == "__main__":
        main()
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
