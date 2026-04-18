"""
Baseline credit-risk inference agent (no retrieval, direct prompting).

Exports:
    BaselineAgent — runs Mistral 7B with a direct prompting strategy over
    pre-extracted 10-K sections and saves predictions to CSV.
"""

import json
import logging
import os
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "results",
)
_OUTPUT_CSV = os.path.join(_RESULTS_DIR, "baseline_predictions.csv")

_CSV_COLUMNS = [
    "ticker",
    "company_name",
    "predicted_score",
    "predicted_label",
    "risk_level",
    "key_signals",
    "citations",
    "raw_output",
    "approach",
]


def _score_to_label(score: float) -> int:
    """Convert a 0-100 risk score to a binary label (1 = high risk)."""
    return 1 if score >= 50 else 0


class BaselineAgent:
    """Direct-prompting credit-risk analyser using Mistral 7B Instruct.

    Args:
        model: A loaded causal LM (from ``base_loader.load_model``).
        tokenizer: The corresponding tokenizer.
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Single-company analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        company_name: str,
        sections: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyse a single company's 10-K and return a prediction row.

        Args:
            ticker: Stock ticker symbol.
            company_name: Full company name.
            sections: Dict with keys ``item_1a``, ``item_7``, ``item_8``,
                ``full_text`` (as produced by the data pipeline).

        Returns:
            Dict matching the prediction CSV schema.
        """
        from src.models.base_loader import generate_response
        from src.prompts.baseline_prompt import build_baseline_prompt, parse_model_output

        item_1a = sections.get("item_1a", "") or ""
        item_7 = sections.get("item_7", "") or ""

        logger.info("[baseline] Analyzing %s (%s).", company_name, ticker)

        prompt = build_baseline_prompt(
            company_name=company_name,
            ticker=ticker,
            item_1a=item_1a,
            item_7=item_7,
        )

        raw_output = generate_response(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
        )

        parsed = parse_model_output(raw_output)

        predicted_score = float(parsed.get("risk_score", 50))
        predicted_label = _score_to_label(predicted_score)
        risk_level = parsed.get("risk_level", "medium")
        key_signals = parsed.get("key_signals", [])
        citations = parsed.get("citations", [])

        logger.info(
            "[baseline] %s -> score=%.1f, label=%d, level=%s",
            ticker,
            predicted_score,
            predicted_label,
            risk_level,
        )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "predicted_score": predicted_score,
            "predicted_label": predicted_label,
            "risk_level": risk_level,
            "key_signals": json.dumps(key_signals),
            "citations": json.dumps(citations),
            "raw_output": raw_output,
            "approach": "baseline",
        }

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_all(
        self,
        companies_df: pd.DataFrame,
        sections_dir: str,
    ) -> pd.DataFrame:
        """Analyse every company in ``companies_df`` and save predictions CSV.

        Sections are loaded from ``{sections_dir}/{ticker}_sections.json``.
        Failed companies receive a default score of 50 and a warning is logged.

        Args:
            companies_df: DataFrame with at minimum ``ticker`` and
                ``company_name`` columns (as produced by the data pipeline).
            sections_dir: Directory containing ``{ticker}_sections.json`` files.

        Returns:
            DataFrame of predictions with columns matching the output CSV schema.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore

        rows = []
        iterable = companies_df.iterrows()
        if tqdm is not None:
            iterable = tqdm(
                list(iterable),
                desc="[baseline] Analyzing companies",
                unit="company",
            )

        for _, row in iterable:
            ticker = str(row["ticker"])
            company_name = str(row["company_name"])

            sections_path = os.path.join(sections_dir, f"{ticker}_sections.json")
            try:
                with open(sections_path, "r", encoding="utf-8") as fh:
                    sections = json.load(fh)
            except FileNotFoundError:
                logger.warning("[baseline] Sections file not found: %s", sections_path)
                sections = {}
            except json.JSONDecodeError as exc:
                logger.warning("[baseline] Failed to parse sections JSON for %s: %s", ticker, exc)
                sections = {}

            try:
                result = self.analyze(ticker, company_name, sections)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[baseline] Analysis failed for %s (%s): %s",
                    ticker,
                    company_name,
                    exc,
                    exc_info=True,
                )
                result = {
                    "ticker": ticker,
                    "company_name": company_name,
                    "predicted_score": 50.0,
                    "predicted_label": 0,
                    "risk_level": "medium",
                    "key_signals": json.dumps([]),
                    "citations": json.dumps([]),
                    "raw_output": f"ERROR: {exc}",
                    "approach": "baseline",
                }

            rows.append(result)

        predictions_df = pd.DataFrame(rows, columns=_CSV_COLUMNS)

        os.makedirs(_RESULTS_DIR, exist_ok=True)
        predictions_df.to_csv(_OUTPUT_CSV, index=False)
        logger.info("[baseline] Predictions saved to '%s'.", _OUTPUT_CSV)

        return predictions_df
