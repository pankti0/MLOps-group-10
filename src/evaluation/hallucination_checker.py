"""
Hallucination detection utilities for LLM-generated credit risk analyses.

Checks whether numeric claims in model outputs are grounded in source
documents (10-K sections) and whether citations match source text.

Exports:
    contains_number(text: str) -> bool
    check_fabrication(output_text, source_chunks, fuzzy_threshold) -> dict
    check_citation_accuracy(citations, source_chunks, fuzzy_threshold) -> dict
    score_response(output_text, citations, source_chunks) -> dict
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz  # type: ignore

    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning(
        "rapidfuzz is not installed. Hallucination checking will be disabled. "
        "Install with: pip install rapidfuzz"
    )

# Sentence splitter – split on period/exclamation/question followed by space + capital
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Detects any numeric token: digits, percentages, currency values
_NUMBER_RE = re.compile(r"\$?[\d,]+(?:\.\d+)?%?|\b\d+\b")


def contains_number(text: str) -> bool:
    """Return True if *text* contains at least one numeric token.

    Args:
        text: Input string to inspect.

    Returns:
        True if a numeric token is found, False otherwise.
    """
    return bool(_NUMBER_RE.search(text))


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex.

    Args:
        text: Arbitrary text.

    Returns:
        List of sentence strings (stripped, non-empty).
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _fuzzy_in_chunks(query: str, source_chunks: List[dict], threshold: int) -> bool:
    """Return True if *query* fuzzy-matches any source chunk above *threshold*.

    Args:
        query: Text to search for.
        source_chunks: List of chunk dicts, each expected to have a ``text``
            key containing the source passage.
        threshold: Minimum rapidfuzz partial_ratio score (0–100) for a match.

    Returns:
        True if any source chunk yields a score >= threshold.
    """
    if not _RAPIDFUZZ_AVAILABLE:
        return False
    for chunk in source_chunks:
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        score = fuzz.partial_ratio(query.lower(), chunk_text.lower())
        if score >= threshold:
            return True
    return False


def check_fabrication(
    output_text: str,
    source_chunks: List[dict],
    fuzzy_threshold: int = 85,
) -> dict:
    """Detect fabricated quantitative claims in a model output.

    Sentences that contain a numeric token are checked against source chunks
    using fuzzy string matching. Sentences with no matching source passage are
    counted as fabrications.

    Args:
        output_text: Full text of the LLM-generated response.
        source_chunks: List of source chunk dicts with a ``text`` key.
        fuzzy_threshold: Minimum rapidfuzz partial_ratio score (0–100)
            required to consider a sentence grounded.

    Returns:
        Dictionary with keys:
            - ``fabrication_rate`` (float): fraction of quantitative claims
              that could not be grounded in source chunks.
            - ``fabrication_count`` (int)
            - ``total_quantitative_claims`` (int)
            - ``fabricated_sentences`` (List[str])
    """
    sentences = _split_sentences(output_text)
    quantitative = [s for s in sentences if contains_number(s) and len(s) >= 20]

    fabricated: List[str] = []
    for sentence in quantitative:
        if not _fuzzy_in_chunks(sentence, source_chunks, fuzzy_threshold):
            fabricated.append(sentence)
            logger.debug("Potentially fabricated sentence: %s", sentence[:120])

    total = len(quantitative)
    count = len(fabricated)
    rate = count / total if total > 0 else 0.0

    logger.info(
        "Fabrication check: %d/%d quantitative claims ungrounded (rate=%.2f)",
        count, total, rate,
    )

    return {
        "fabrication_rate": round(rate, 4),
        "fabrication_count": count,
        "total_quantitative_claims": total,
        "fabricated_sentences": fabricated,
    }


def check_citation_accuracy(
    citations: List[str],
    source_chunks: List[dict],
    fuzzy_threshold: int = 85,
) -> dict:
    """Evaluate how accurately cited passages match source documents.

    Args:
        citations: List of citation strings (text excerpts claimed by the model).
        source_chunks: List of source chunk dicts with a ``text`` key.
        fuzzy_threshold: Minimum rapidfuzz partial_ratio score (0–100) for
            a citation to be considered correct.

    Returns:
        Dictionary with keys:
            - ``citation_accuracy`` (float): fraction of citations that match
              a source chunk.
            - ``correct_citations`` (int)
            - ``total_citations`` (int)
    """
    if not citations:
        logger.info("No citations provided; citation accuracy is undefined (1.0).")
        return {
            "citation_accuracy": 1.0,
            "correct_citations": 0,
            "total_citations": 0,
        }

    correct = 0
    for citation in citations:
        if _fuzzy_in_chunks(str(citation), source_chunks, fuzzy_threshold):
            correct += 1
        else:
            logger.debug("Citation not found in source chunks: %s", str(citation)[:120])

    total = len(citations)
    accuracy = correct / total if total > 0 else 0.0

    logger.info(
        "Citation accuracy: %d/%d correct (%.2f)", correct, total, accuracy
    )

    return {
        "citation_accuracy": round(accuracy, 4),
        "correct_citations": correct,
        "total_citations": total,
    }


def score_response(
    output_text: str,
    citations: List[str],
    source_chunks: List[dict],
    fuzzy_threshold: int = 85,
) -> dict:
    """Compute a combined grounding score for a model response.

    Grounding score = (1 - fabrication_rate) * citation_accuracy

    Args:
        output_text: Full text of the LLM-generated response.
        citations: List of citation strings claimed by the model.
        source_chunks: List of source chunk dicts with a ``text`` key.
        fuzzy_threshold: Fuzzy matching threshold passed to both sub-checks.

    Returns:
        Merged dictionary containing all keys from check_fabrication and
        check_citation_accuracy, plus:
            - ``grounding_score`` (float): combined quality score in [0, 1].
    """
    fab_result = check_fabrication(output_text, source_chunks, fuzzy_threshold)
    cit_result = check_citation_accuracy(citations, source_chunks, fuzzy_threshold)

    grounding_score = (1.0 - fab_result["fabrication_rate"]) * cit_result["citation_accuracy"]

    result = {
        **fab_result,
        **cit_result,
        "grounding_score": round(grounding_score, 4),
    }
    logger.info("Grounding score: %.4f", grounding_score)
    return result
