"""
RAG-based credit-risk inference agent (IMPROVED VERSION).

Enhancements:
- BGE query prefix for better embeddings
- Metadata-aware reranking (company + section boosting)
- STRICT company filtering (FIXED)
- Retrieval → filter → rerank → select pipeline
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

_RETRIEVE_K = 20  # you can increase to 40 if needed


def _score_to_label(score: float) -> int:
    return 1 if score >= 50 else 0


class RAGAgent:
    def __init__(self, model, tokenizer, faiss_store, embedder) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.faiss_store = faiss_store
        self.embedder = embedder

    # -----------------------------------------------------------
    # Metadata-aware reranking
    # -----------------------------------------------------------
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        target_ticker: str,
    ) -> List[Dict[str, Any]]:
        scored = []

        for r in results:
            meta = r.get("metadata", {})
            base_score = r.get("score", 0)

            bonus = 0.0

            # Boost same company (redundant now but still good)
            if meta.get("ticker") == target_ticker:
                bonus += 0.25

            # Boost important sections
            if meta.get("section") == "item_1a":
                bonus += 0.15
            elif meta.get("section") in ["item_7", "item_7a"]:
                bonus += 0.05

            scored.append((base_score + bonus, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    # -----------------------------------------------------------
    # 🔥 FIXED retrieval pipeline
    # -----------------------------------------------------------
    def _retrieve_chunks_for_company(
        self,
        ticker: str,
        query: str,
    ) -> List[Dict[str, Any]]:

        # BGE best practice
        query = "Represent this sentence for searching relevant passages: " + query

        # Step 1: Broad retrieval
        candidates = self.faiss_store.query(
            query_text=query,
            embedder=self.embedder,
            k=_RETRIEVE_K,
        )

        if not candidates:
            logger.warning("[rag] No candidates retrieved for %s.", ticker)
            return []

        # -----------------------------------------------------------
        # 🔥 STRICT company filtering BEFORE reranking
        # -----------------------------------------------------------
        company_candidates = [
            c for c in candidates
            if c.get("metadata", {}).get("ticker") == ticker
        ]

        if not company_candidates:
            logger.warning("[rag] No company-specific chunks found for %s.", ticker)
            return []

        # -----------------------------------------------------------
        # Rerank ONLY within same company
        # -----------------------------------------------------------
        reranked = self._rerank_results(company_candidates, target_ticker=ticker)

        return reranked[:10]

    # -----------------------------------------------------------
    # Single company analysis
    # -----------------------------------------------------------
    def analyze(
        self,
        ticker: str,
        company_name: str,
        sections: Dict[str, Any],
    ) -> Dict[str, Any]:

        from src.models.base_loader import generate_response
        from src.prompts.rag_prompt import build_rag_prompt, parse_rag_output

        query = f"{company_name} financial risks liquidity debt going concern risk factors"

        logger.info("[rag] Retrieving chunks for %s (%s).", company_name, ticker)

        retrieved_chunks = self._retrieve_chunks_for_company(ticker, query)

        logger.info("[rag] Using %d chunks for %s.", len(retrieved_chunks), ticker)

        # Debug preview
        logger.info("[rag] Top retrieved chunks preview:")
        for c in retrieved_chunks[:3]:
            meta = c.get("metadata", {})
            logger.info(
                "  -> ticker=%s | section=%s",
                meta.get("ticker"),
                meta.get("section"),
            )

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

    # -----------------------------------------------------------
    # Batch analysis
    # -----------------------------------------------------------
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
            iterable = tqdm(list(iterable), desc="[rag] Analyzing", unit="company")

        for _, row in iterable:
            ticker = str(row["ticker"])
            company_name = str(row["company_name"])

            try:
                result = self.analyze(ticker, company_name, sections={})
            except Exception as exc:
                logger.error(
                    "[rag] Failed for %s (%s): %s",
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