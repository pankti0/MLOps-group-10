"""
RAG-based credit-risk inference agent.

Exports:
    RAGAgent — retrieves relevant passages per company from a FAISS index,
    then prompts Mistral 7B with those passages and saves predictions to CSV.
"""

import json
import logging
import os
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

_RAG_QUERY_SUFFIX = " credit risk financial health debt liquidity going concern"
_MIN_CHUNKS = 3
_RETRIEVE_K = 20  # retrieve broadly, then filter to this company


def _score_to_label(score: float) -> int:
    """Convert a 0-100 risk score to a binary label (1 = high risk)."""
    return 1 if score >= 50 else 0


class RAGAgent:
    """Retrieval-augmented credit-risk analyser using Mistral 7B Instruct.

    Args:
        model: A loaded causal LM (from ``base_loader.load_model``).
        tokenizer: The corresponding tokenizer.
        faiss_store: A built/loaded ``FAISSStore`` instance.
        embedder: The ``Embedder`` instance used to build the FAISS store.
    """

    def __init__(self, model, tokenizer, faiss_store, embedder) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.faiss_store = faiss_store
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_chunks_for_company(
        self,
        ticker: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Return relevant chunks belonging to ``ticker``.

        Retrieves ``_RETRIEVE_K`` candidates, filters to those whose metadata
        matches the ticker, and falls back to all chunks for this company if
        fewer than ``_MIN_CHUNKS`` are found.

        Args:
            ticker: Ticker symbol used to filter chunks.
            query: Natural-language query for vector retrieval.

        Returns:
            List of chunk dicts (subset of the FAISS store's metadata).
        """
        candidates = self.faiss_store.query(
            query_text=query,
            embedder=self.embedder,
            k=_RETRIEVE_K,
        )

        # Filter to this company
        company_chunks = [
            c for c in candidates
            if c.get("metadata", {}).get("ticker", "") == ticker
            or c.get("ticker", "") == ticker
        ]

        if len(company_chunks) >= _MIN_CHUNKS:
            return company_chunks

        # Fallback: pull ALL stored chunks for this ticker
        all_company_chunks = [
            dict(c)
            for c in self.faiss_store._chunks  # type: ignore[attr-defined]
            if c.get("metadata", {}).get("ticker", "") == ticker
            or c.get("ticker", "") == ticker
        ]

        if all_company_chunks:
            logger.warning(
                "[rag] Only %d relevant chunks found for %s via query; "
                "using all %d chunks for this company.",
                len(company_chunks),
                ticker,
                len(all_company_chunks),
            )
            return all_company_chunks

        logger.warning("[rag] No chunks found at all for ticker '%s'.", ticker)
        return candidates[:_MIN_CHUNKS] if candidates else []

    # ------------------------------------------------------------------
    # Single-company analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        company_name: str,
        sections: Dict[str, Any],  # noqa: ARG002 — kept for API consistency
    ) -> Dict[str, Any]:
        """Analyse a single company using RAG and return a prediction row.

        ``sections`` is accepted for interface parity with ``BaselineAgent``
        but is not used — the retrieved chunks serve as context.

        Args:
            ticker: Stock ticker symbol.
            company_name: Full company name.
            sections: Dict of 10-K sections (ignored; kept for interface parity).

        Returns:
            Dict matching the prediction CSV schema.
        """
        from src.models.base_loader import generate_response
        from src.prompts.rag_prompt import build_rag_prompt, parse_rag_output

        query = company_name + _RAG_QUERY_SUFFIX
        logger.info("[rag] Retrieving chunks for %s (%s).", company_name, ticker)

        retrieved_chunks = self._retrieve_chunks_for_company(ticker, query)
        logger.info("[rag] Using %d chunks for %s.", len(retrieved_chunks), ticker)

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

        parsed = parse_rag_output(raw_output)

        predicted_score = float(parsed.get("risk_score", 50))
        predicted_label = _score_to_label(predicted_score)
        risk_level = parsed.get("risk_level", "medium")
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
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_all(
        self,
        companies_df: pd.DataFrame,
        sections_dir: str,  # noqa: ARG002 — kept for interface parity
    ) -> pd.DataFrame:
        """Analyse every company and save predictions CSV.

        ``sections_dir`` is accepted for interface parity with
        ``BaselineAgent`` but is not required for RAG inference.

        Args:
            companies_df: DataFrame with at minimum ``ticker`` and
                ``company_name`` columns.
            sections_dir: Path to processed sections (not used by RAG).

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
                desc="[rag] Analyzing companies",
                unit="company",
            )

        for _, row in iterable:
            ticker = str(row["ticker"])
            company_name = str(row["company_name"])

            try:
                result = self.analyze(ticker, company_name, sections={})
            except Exception as exc:  # noqa: BLE001
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
