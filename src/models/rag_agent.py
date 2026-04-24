"""
RAG-based credit-risk inference agent.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "results",
)
_OUTPUT_CSV = os.path.join(_RESULTS_DIR, "rag_predictions.csv")

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

# 🔥 STRONGER QUERY
_RAG_QUERY_SUFFIX = (
    " HIGH RISK debt leverage liquidity crisis financial distress "
    "going concern warning covenant breach losses negative cash flow "
    "refinancing risk bankruptcy risk debt obligations"
)

_MIN_CHUNKS = 3
_RETRIEVE_K = 7


# ------------------------------------------------------------------
# Thresholds aligned with dataset
# ------------------------------------------------------------------

def _score_to_label(score: float) -> int:
    return 1 if score >= 70 else 0


def _score_to_risk_level(score: float) -> str:
    if score < 35:
        return "low"
    elif score < 70:
        return "medium"
    else:
        return "high"


class RAGAgent:
    def __init__(self, model, tokenizer, faiss_store, embedder) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.faiss_store = faiss_store
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_chunks_for_company(
        self,
        ticker: str,
        query: str,
    ) -> List[Dict[str, Any]]:

        candidates = self.faiss_store.query(
            query_text=query,
            embedder=self.embedder,
            k=100,
        )

        logger.info("[rag] Retrieved %d raw candidates for %s.", len(candidates), ticker)

        company_chunks = [
            c for c in candidates
            if c.get("metadata", {}).get("ticker", "").upper() == ticker.upper()
        ]
        logger.info("[rag] %d chunks after ticker filter.", len(company_chunks))

        VALID_SECTIONS = {"item_1a", "item_7", "item_8"}

        company_chunks = [
            c for c in company_chunks
            if c.get("metadata", {}).get("section", "").lower() in VALID_SECTIONS
        ]
        logger.info("[rag] %d chunks after section filter.", len(company_chunks))

        def is_text_heavy(text: str) -> bool:
            words = text.split()
            if not words:
                return False
            digit_words = sum(1 for w in words if any(c.isdigit() for c in w))
            return (digit_words / len(words)) < 0.5

        company_chunks = [
            c for c in company_chunks
            if is_text_heavy(c.get("text", ""))
        ]
        logger.info("[rag] %d chunks after text filter.", len(company_chunks))

        if len(company_chunks) < _MIN_CHUNKS:
            logger.warning(
                "[rag] Too few filtered chunks (%d) for %s, using top FAISS results.",
                len(company_chunks),
                ticker,
            )
            return candidates[:_RETRIEVE_K]

        company_chunks = sorted(
            company_chunks,
            key=lambda c: c.get("score", 0),
            reverse=True,
        )

        return company_chunks[:_RETRIEVE_K]

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        company_name: str,
        sections: Dict[str, Any],
    ) -> Dict[str, Any]:

        from src.models.base_loader import generate_response
        from src.prompts.rag_prompt import build_rag_prompt, parse_rag_output

        query = company_name + _RAG_QUERY_SUFFIX
        logger.info("[rag] Retrieving chunks for %s (%s).", company_name, ticker)

        retrieved_chunks = self._retrieve_chunks_for_company(ticker, query)
        logger.info("[rag] Using %d chunks for %s.", len(retrieved_chunks), ticker)

        if not retrieved_chunks:
            logger.error("[rag] No usable data for %s (%s).", company_name, ticker)

            return {
                "ticker": ticker,
                "company_name": company_name,
                "predicted_score": -1,
                "predicted_label": 0,
                "risk_level": "unknown",
                "key_signals": json.dumps([]),
                "citations": json.dumps([]),
                "raw_output": "NO_DATA",
                "approach": "rag",
            }

        prompt = build_rag_prompt(
            company_name=company_name,
            ticker=ticker,
            retrieved_chunks=retrieved_chunks,
        )

        raw_output = generate_response(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
        )

        logger.debug("\n=== RAW MODEL OUTPUT ===\n%s\n========================\n", raw_output)

        if not re.search(r"\{.*?\}", raw_output, re.DOTALL):
            logger.warning("[rag] Model did not return JSON. Using fallback.")

            return {
                "ticker": ticker,
                "company_name": company_name,
                "predicted_score": 50.0,
                "predicted_label": 0,
                "risk_level": "medium",
                "key_signals": json.dumps([
                    "Model failed to produce structured output",
                    raw_output[:200]
                ]),
                "citations": json.dumps([]),
                "raw_output": raw_output,
                "approach": "rag",
            }

        parsed = parse_rag_output(raw_output)

        predicted_score = float(parsed.get("risk_score", 50))

        # 🔥 FINAL CALIBRATION (HIGH + LOW)

        strong_high_risk_keywords = [
            "indebtedness",
            "liquidity",
            "not be sufficient",
            "unable to fund",
            "financial distress",
            "loss",
            "going concern"
        ]

        strong_low_risk_keywords = [
            "strong cash",
            "sufficient liquidity",
            "low debt",
            "strong balance sheet",
            "consistent profitability",
            "positive cash flow"
        ]

        text_blob = (
            " ".join(parsed.get("key_signals", [])).lower()
            + " "
            + parsed.get("reasoning", "").lower()
        )

        if any(keyword in text_blob for keyword in strong_high_risk_keywords):
            predicted_score = max(predicted_score, 75)

        elif any(keyword in text_blob for keyword in strong_low_risk_keywords):
            predicted_score = min(predicted_score, 30)

        predicted_label = _score_to_label(predicted_score)
        risk_level = _score_to_risk_level(predicted_score)

        key_signals = parsed.get("key_signals", [])
        citations = parsed.get("citations", [])

        logger.info(
            "[rag] %s -> score=%.1f, label=%d, level=%s",
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
            "approach": "rag",
        }

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def analyze_all(
        self,
        companies_df: pd.DataFrame,
        sections_dir: str,
    ) -> pd.DataFrame:

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        rows = []
        iterable = companies_df.iterrows()

        if tqdm is not None:
            iterable = tqdm(list(iterable), desc="[rag] Analyzing companies")

        for _, row in iterable:
            ticker = str(row["ticker"])
            company_name = str(row["company_name"])

            try:
                result = self.analyze(ticker, company_name, sections={})
            except Exception as exc:
                logger.error(
                    "[rag] Analysis failed for %s (%s): %s",
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
                    "approach": "rag",
                }

            rows.append(result)

        predictions_df = pd.DataFrame(rows, columns=_CSV_COLUMNS)

        os.makedirs(_RESULTS_DIR, exist_ok=True)
        predictions_df.to_csv(_OUTPUT_CSV, index=False)
        logger.info("[rag] Predictions saved to '%s'.", _OUTPUT_CSV)

        return predictions_df