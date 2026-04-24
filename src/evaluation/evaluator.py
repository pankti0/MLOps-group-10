"""
Unified evaluation runner for the Credit Risk Detection project.

Loads predictions from all approaches, merges with ground truth, computes
classification and hallucination metrics, logs to W&B, and saves outputs.

Exports:
    load_predictions(approach, results_dir) -> pd.DataFrame
    run_full_evaluation(results_dir, labels_path, sections_dir) -> dict
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from src.evaluation.altman_zscore import compute_all_zscores, save_zscore_predictions
from src.evaluation.hallucination_checker import score_response
from src.evaluation.metrics import (
    compare_approaches,
    compute_classification_metrics,
    compute_roc_curve,
    plot_roc_curves,
)
from src.utils.wandb_logger import log_metrics, log_predictions, log_roc_curve

logger = logging.getLogger(__name__)

_LLM_APPROACHES: List[str] = ["baseline", "rag", "lora_r8", "lora_r16", "lora_r32"]
_ALL_APPROACHES: List[str] = ["baseline", "rag", "lora_r8", "lora_r16", "lora_r32", "altman_zscore"]


def load_predictions(approach: str, results_dir: str) -> pd.DataFrame:
    """Load predictions CSV for a single approach.

    Args:
        approach: Approach identifier (e.g. 'baseline', 'rag', 'lora',
            'altman_zscore').
        results_dir: Directory containing ``{approach}_predictions.csv`` files.

    Returns:
        Loaded DataFrame, or an empty DataFrame if the file is missing.
    """
    path = os.path.join(results_dir, f"{approach}_predictions.csv")
    if not os.path.exists(path):
        logger.warning("Predictions file not found for approach '%s': %s", approach, path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info("Loaded %d predictions for approach '%s' from %s", len(df), approach, path)
    return df


def _load_sections(ticker: str, sections_dir: str) -> Optional[dict]:
    """Load processed section JSON for a ticker.

    Args:
        ticker: Company ticker symbol.
        sections_dir: Directory containing ``{ticker}_sections.json`` files.

    Returns:
        Parsed sections dict, or None if the file is missing or invalid.
    """
    path = os.path.join(sections_dir, f"{ticker}_sections.json")
    if not os.path.exists(path):
        logger.debug("Sections file not found for %s at %s", ticker, path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse sections JSON for %s: %s", ticker, exc)
        return None


def _sections_to_chunks(sections: dict) -> List[dict]:
    """Convert a sections dict into a flat list of source chunks.

    Args:
        sections: Dict with keys item_1a, item_7, item_8, full_text.

    Returns:
        List of chunk dicts with keys ``text`` and ``source``.
    """
    chunks: List[dict] = []
    for key in ("item_1a", "item_7", "item_8"):
        text = sections.get(key, "")
        if text:
            chunks.append({"text": text, "source": key})
    return chunks


def _parse_json_list(value: str) -> list:
    """Safely parse a JSON list from a string, returning empty list on failure.

    Args:
        value: String representation of a JSON list.

    Returns:
        Parsed list, or empty list.
    """
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _compute_hallucination_metrics(
    df: pd.DataFrame, sections_dir: str
) -> Dict[str, dict]:
    """Compute per-company hallucination scores for an LLM predictions DataFrame.

    Args:
        df: Predictions DataFrame with columns raw_output, citations, ticker.
        sections_dir: Directory containing processed section JSON files.

    Returns:
        Dict mapping ticker → hallucination score dict.
    """
    results: Dict[str, dict] = {}
    for _, row in df.iterrows():
        ticker = row.get("ticker", "UNKNOWN")
        raw_output = row.get("raw_output", "")
        citations_raw = row.get("citations", "[]")
        citations = _parse_json_list(str(citations_raw))

        sections = _load_sections(ticker, sections_dir)
        source_chunks = _sections_to_chunks(sections) if sections else []

        if not source_chunks:
            logger.debug("No source chunks for %s; skipping hallucination check.", ticker)
            continue

        score = score_response(
            output_text=str(raw_output),
            citations=citations,
            source_chunks=source_chunks,
        )
        results[ticker] = score

    return results


def _aggregate_hallucination(scores: Dict[str, dict]) -> dict:
    """Aggregate per-company hallucination scores to means.

    Args:
        scores: Dict of ticker → score dict.

    Returns:
        Aggregated metrics dict.
    """
    if not scores:
        return {}
    import numpy as np

    keys = ["fabrication_rate", "citation_accuracy", "grounding_score"]
    aggregated = {}
    for key in keys:
        values = [v[key] for v in scores.values() if key in v]
        if values:
            aggregated[f"mean_{key}"] = float(np.mean(values))
    return aggregated


def _print_comparison_table(comparison: pd.DataFrame) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        comparison: DataFrame from compare_approaches().
    """
    separator = "-" * 90
    print("\n" + separator)
    print(" CREDIT RISK DETECTION — EVALUATION RESULTS")
    print(separator)
    if comparison.empty:
        print("  No results to display.")
    else:
        print(comparison.to_string())
    print(separator + "\n")


def run_full_evaluation(
    results_dir: str,
    labels_path: str,
    sections_dir: str,
) -> dict:
    """Run the complete evaluation pipeline.

    Steps:
        1. Load predictions from all approaches (skip missing files).
        2. Ensure altman_zscore predictions exist (generate if absent).
        3. Load ground truth labels.
        4. Merge predictions with labels.
        5. Compute classification metrics per approach.
        6. Compute hallucination metrics for LLM approaches.
        7. Log everything to W&B.
        8. Save metrics summary CSV.
        9. Plot and save ROC curves.
        10. Print formatted comparison table.

    Args:
        results_dir: Directory containing ``{approach}_predictions.csv`` files.
        labels_path: Path to ``company_labels.csv``.
        sections_dir: Directory containing ``{ticker}_sections.json`` files.

    Returns:
        Dictionary mapping approach → metrics dict, plus a ``comparison``
        key containing the full comparison DataFrame.
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Load all predictions ---
    all_predictions: Dict[str, pd.DataFrame] = {}
    for approach in _ALL_APPROACHES:
        df = load_predictions(approach, results_dir)
        if not df.empty:
            all_predictions[approach] = df

    # --- 2. Ensure altman_zscore predictions ---
    if "altman_zscore" not in all_predictions:
        logger.info("Generating altman_zscore predictions...")
        zscore_path = os.path.join(results_dir, "altman_zscore_predictions.csv")
        df_zscore = save_zscore_predictions(zscore_path)
        all_predictions["altman_zscore"] = df_zscore

    # --- 3. Load ground truth labels ---
    if not os.path.exists(labels_path):
        logger.error("Labels file not found: %s", labels_path)
        return {}

    labels_df = pd.read_csv(labels_path)
    logger.info("Loaded %d ground truth labels from %s", len(labels_df), labels_path)

    # --- 4. Merge predictions with labels ---
    merged_predictions: Dict[str, pd.DataFrame] = {}
    for approach, pred_df in all_predictions.items():
        if "ticker" not in pred_df.columns:
            logger.warning("Predictions for '%s' missing 'ticker' column; skipping.", approach)
            continue
        merged = pred_df.merge(
            labels_df[["ticker", "label"]],
            on="ticker",
            how="inner",
        )
        if merged.empty:
            logger.warning("No matching tickers for approach '%s' after merge.", approach)
        else:
            merged_predictions[approach] = merged
            logger.info("Merged %d rows for approach '%s'.", len(merged), approach)

    # --- 5. Compute classification metrics ---
    all_metrics: Dict[str, dict] = {}
    for approach, df in merged_predictions.items():
        if "label" not in df.columns or "predicted_score" not in df.columns:
            logger.warning("Skipping metrics for '%s' — missing required columns.", approach)
            continue
        metrics = compute_classification_metrics(
            df["label"].values, df["predicted_score"].values
        )
        all_metrics[approach] = metrics
        logger.info("Metrics for '%s': %s", approach, metrics)

        # Log to W&B
        log_metrics(metrics, approach)
        log_predictions(df, approach)

        # Log ROC curve
        fpr, tpr, roc_auc = compute_roc_curve(df["label"].values, df["predicted_score"].values)
        log_roc_curve(fpr, tpr, roc_auc, approach)

    # --- 6. Compute hallucination metrics for LLM approaches ---
    for approach in _LLM_APPROACHES:
        if approach not in merged_predictions:
            continue
        df = merged_predictions[approach]
        logger.info("Computing hallucination metrics for '%s'...", approach)
        per_company_scores = _compute_hallucination_metrics(df, sections_dir)
        aggregated = _aggregate_hallucination(per_company_scores)
        if aggregated:
            all_metrics[approach]["hallucination"] = aggregated
            log_metrics({f"hallucination_{k}": v for k, v in aggregated.items()}, approach)
            logger.info("Hallucination metrics for '%s': %s", approach, aggregated)

    # --- 7. Build comparison table ---
    comparison = compare_approaches(merged_predictions)

    # --- 8. Save metrics summary CSV ---
    metrics_path = os.path.join(results_dir, "metrics_summary.csv")
    metrics_rows = []
    for approach, metrics in all_metrics.items():
        flat = {"approach": approach}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}_{sub_k}"] = sub_v
            else:
                flat[k] = v
        metrics_rows.append(flat)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Saved metrics summary to %s", metrics_path)

    # --- 9. Plot and save ROC curves ---
    roc_plot_path = os.path.join(results_dir, "roc_curves.png")
    plot_roc_curves(merged_predictions, roc_plot_path)

    # --- 10. Print formatted comparison table ---
    _print_comparison_table(comparison)

    return {
        **all_metrics,
        "comparison": comparison,
    }
