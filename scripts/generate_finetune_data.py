#!/usr/bin/env python3
"""
generate_finetune_data.py

Generates LoRA fine-tuning training data from processed 10-K sections and
company labels. Each company's sections are combined with a rule-based
template ideal output (based on label and risk_category) to form a
supervised training example.

Splits: 70/15/15 

Output:
    data/finetune/train.jsonl
    data/finetune/val.jsonl
    data/finetune/test.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from typing import Dict, List

import pandas as pd


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.prompts.lora_prompt import build_training_example
from src.utils.config_loader import get_repo_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)



def _high_risk_output(company_name: str, ticker: str, score: int) -> dict:
    """Build a high-risk ideal output template.

    Args:
        company_name: Company name.
        ticker: Ticker symbol.
        score: Numeric risk score (expected in 67–100 range).

    Returns:
        Ideal output dict matching the JSON schema.
    """
    return {
        "predicted_score": float(score),
        "risk_level": "high",
        "key_signals": [
            "Significant debt burden with elevated leverage ratios",
            "Going concern uncertainty or liquidity risk disclosed",
            "Declining revenue or negative operating trends",
            "Negative retained earnings and stockholders deficit",
            "High interest expense relative to operating income",
        ],
        "citations": [
            "We have incurred significant indebtedness and may not be able to service our obligations.",
            "Our substantial leverage could adversely affect our financial condition.",
        ],
        "rationale": (
            f"{company_name} ({ticker}) exhibits elevated credit risk driven by a combination "
            "of high financial leverage, liquidity pressures, and deteriorating operating performance. "
            "The filing discloses material uncertainties around debt servicing capacity and potential "
            "covenant violations. Negative retained earnings and declining revenue trends indicate "
            "sustained financial stress that warrants classification as high credit risk."
        ),
    }


def _medium_risk_output(company_name: str, ticker: str, score: int) -> dict:
    """Build a medium-risk ideal output template.

    Args:
        company_name: Company name.
        ticker: Ticker symbol.
        score: Numeric risk score (expected in 34–66 range).

    Returns:
        Ideal output dict matching the JSON schema.
    """
    return {
        "predicted_score": float(score),
        "risk_level": "medium",
        "key_signals": [
            "Moderate leverage with manageable debt maturities",
            "Mixed revenue growth with some segment-level weakness",
            "Exposure to cyclical or macroeconomic headwinds",
            "Adequate but not robust liquidity position",
            "Some regulatory or competitive pressures noted",
        ],
        "citations": [
            "We face significant competition and our results of operations may be adversely affected.",
            "Economic downturns or disruptions could negatively impact our business.",
        ],
        "rationale": (
            f"{company_name} ({ticker}) presents mixed credit risk signals. While the company "
            "maintains adequate liquidity and manageable leverage, it faces headwinds from competitive "
            "pressures, macroeconomic uncertainty, and some operational challenges. Revenue growth "
            "is uneven across segments. The overall profile warrants a medium risk classification "
            "pending sustained improvement in operating metrics."
        ),
    }


def _low_risk_output(company_name: str, ticker: str, score: int) -> dict:
    """Build a low-risk ideal output template.

    Args:
        company_name: Company name.
        ticker: Ticker symbol.
        score: Numeric risk score (expected in 0–33 range).

    Returns:
        Ideal output dict matching the JSON schema.
    """
    return {
        "predicted_score": float(score),
        "risk_level": "low",
        "key_signals": [
            "Strong balance sheet with substantial cash and low net debt",
            "Consistent positive free cash flow generation",
            "Diversified and resilient revenue streams",
            "Investment-grade credit profile with ample liquidity",
            "Stable or growing earnings with strong operating margins",
        ],
        "citations": [
            "We have generated strong cash flow from operations enabling continued investment.",
            "Our diversified business model provides resilience against market downturns.",
        ],
        "rationale": (
            f"{company_name} ({ticker}) demonstrates low credit risk with a robust financial "
            "profile characterized by strong cash generation, manageable leverage, and diversified "
            "revenue. The company maintains ample liquidity headroom and an investment-grade "
            "credit profile. Consistent profitability and positive free cash flow support "
            "confidence in debt servicing capacity over the near and medium term."
        ),
    }


def _build_ideal_output(company_name: str, ticker: str, risk_category: str, risk_score: int) -> dict:
    """Route to the appropriate template based on risk category."""
    if risk_category == "high":
        return _high_risk_output(company_name, ticker, risk_score)
    elif risk_category == "low":
        return _low_risk_output(company_name, ticker, risk_score)
    else:
        return _medium_risk_output(company_name, ticker, risk_score)


def _load_sections(ticker: str, sections_dir: str) -> dict:
    """Load processed section JSON for a ticker."""
    path = os.path.join(sections_dir, f"{ticker}_sections.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _write_jsonl(examples: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    """Generate fine-tuning data and perform stratified k-fold cross-validation splits."""
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter

    repo_root = get_repo_root()
    labels_path = repo_root / "data" / "labels" / "company_labels.csv"
    sections_dir = str(repo_root / "data" / "processed")
    output_dir = str(repo_root / "data" / "finetune")

    # Load labels
    if not labels_path.exists():
        logger.error("Labels file not found: %s", labels_path)
        sys.exit(1)


    labels_df = pd.read_csv(labels_path)
    logger.info("Loaded %d companies from %s", len(labels_df), labels_path)

    # Build one training example per company
    examples: List[dict] = []
    y = []  # stratification labels

    for _, row in labels_df.iterrows():
        ticker = str(row["ticker"])
        company_name = str(row["company_name"])
        risk_category = str(row.get("risk_category", "medium"))
        risk_score = int(row.get("risk_score", 50))

        sections = _load_sections(ticker, sections_dir)
        if not sections:
            logger.warning("No sections for %s — using placeholder text.", ticker)
            item_1a = f"Risk factors section not available for {company_name}."
            item_7 = f"Management discussion not available for {company_name}."
        else:
            item_1a = sections.get("item_1a", "")
            item_7 = sections.get("item_7", "")

        ideal_output = _build_ideal_output(company_name, ticker, risk_category, risk_score)

        example = build_training_example(
            company_name=company_name,
            ticker=ticker,
            item_1a=item_1a,
            item_7=item_7,
            ideal_output=ideal_output,
        )
        example["ticker"] = ticker
        example["risk_category"] = risk_category
        examples.append(example)
        y.append(risk_category)

    if not examples:
        logger.error("No training examples generated. Exiting.")
        sys.exit(1)

    # Stratified K-Fold Cross-Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print(f"  STRATIFIED {n_splits}-FOLD CROSS-VALIDATION SPLITS")
    print("=" * 60)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(examples, y), 1):
        # Further split train_val into train/val (e.g., 80/20)
        train_val_y = [y[i] for i in train_val_idx]
        train_val_examples = [examples[i] for i in train_val_idx]
        test_examples = [examples[i] for i in test_idx]

        # Stratified split for val
        val_size = max(1, int(0.2 * len(train_val_examples)))
        if val_size >= len(train_val_examples):
            val_size = max(1, len(train_val_examples) // 5)
        val_skf = StratifiedKFold(n_splits=len(train_val_examples) // val_size, shuffle=True, random_state=fold)
        val_split = next(val_skf.split(train_val_examples, train_val_y))
        train_idx, val_idx = val_split
        train_examples = [train_val_examples[i] for i in train_idx]
        val_examples = [train_val_examples[i] for i in val_idx]

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        train_path = os.path.join(fold_dir, "train.jsonl")
        val_path = os.path.join(fold_dir, "val.jsonl")
        test_path = os.path.join(fold_dir, "test.jsonl")

        _write_jsonl(train_examples, train_path)
        _write_jsonl(val_examples, val_path)
        _write_jsonl(test_examples, test_path)

        print(f"FOLD {fold}")
        print(f"  Train: {len(train_examples)}  Val: {len(val_examples)}  Test: {len(test_examples)}")
        for split_name, split_examples in [
            ("train", train_examples),
            ("val", val_examples),
            ("test", test_examples),
        ]:
            cats = Counter(e.get("risk_category", "unknown") for e in split_examples)
            print(f"    {split_name:5s} risk categories: {dict(cats)}")
        print(f"  Saved to: {fold_dir}")
        print("-" * 60)

    print("\nAll folds complete. Use fold_X/train.jsonl etc. for training and evaluation.")


if __name__ == "__main__":
    main()
