"""
Prompt templates and output parsing for the RAG-based credit-risk agent.

Exports:
    build_rag_prompt(company_name, ticker, retrieved_chunks) -> str
    parse_rag_output(raw_output) -> dict
"""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values returned when parsing fails
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT: Dict[str, Any] = {
    "risk_score": 50,
    "risk_level": "medium",
    "key_signals": [],
    "citations": [],
    "cited_chunk_ids": [],
    "reasoning": "Could not parse model output.",
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are an expert credit risk analyst with 20 years of experience evaluating corporate \
10-K filings for institutional lenders. You are rigorous and evidence-based. \
You ONLY draw conclusions from the passages provided to you — never from external knowledge.\
"""

_ANALYSIS_TASK = """\
Using ONLY the passages provided below, analyse the credit risk of {company_name} ({ticker}).

For each claim or signal you identify:
  - You MUST cite the chunk_id of the passage that supports it.
  - Use verbatim quotes where possible.

Focus on:
1. **Debt levels** — total debt, leverage ratios, debt maturity schedule
2. **Liquidity concerns** — cash position, current ratio, working capital, cash burn
3. **Revenue trends** — growth, decline, concentration, cyclicality
4. **Going concern language** — any auditor or management doubt about continuing operations
5. **Covenant violations** — disclosed breaches or waiver requests
6. **Risk factor severity** — material risk factors that could impair debt repayment

After your analysis, output ONLY the following JSON block (no extra commentary after it):

```json
{{
  "risk_score": <integer 0-100, where 0=no risk, 100=certain default>,
  "risk_level": "<low|medium|high>",
  "key_signals": ["<concise signal 1>", "<concise signal 2>", ...],
  "citations": ["<exact verbatim quote from a passage>", ...],
  "cited_chunk_ids": ["<chunk_id of cited passage>", ...],
  "reasoning": "<2-3 sentence explanation of your overall assessment>"
}}
```
"""


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered passages for inclusion in the prompt."""
    lines: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        metadata = chunk.get("metadata", {})
        section = metadata.get("section", "unknown") if isinstance(metadata, dict) else "unknown"
        text = chunk.get("text", "").strip()
        lines.append(f"[{chunk_id}] (section: {section})\n{text}")
    return "\n\n".join(lines)


def build_rag_prompt(
    company_name: str,
    ticker: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """Build a Mistral 7B Instruct-format RAG prompt.

    The prompt strictly instructs the model to use only the provided passages
    and to cite chunk IDs for every claim.

    Args:
        company_name: Full company name.
        ticker: Stock ticker symbol.
        retrieved_chunks: List of chunk dicts from FAISSStore.query(), each
            containing at minimum ``chunk_id`` and ``text`` keys.

    Returns:
        A fully formatted ``[INST] ... [/INST]`` prompt string.
    """
    task = _ANALYSIS_TASK.format(company_name=company_name, ticker=ticker)

    if retrieved_chunks:
        passages_text = _format_chunks(retrieved_chunks)
        passages_block = f"--- RETRIEVED PASSAGES ---\n{passages_text}"
    else:
        passages_block = "--- RETRIEVED PASSAGES ---\n[No passages were retrieved for this company.]"

    user_content = f"{task}\n\n{passages_block}"

    prompt = (
        f"<s>[INST] <<SYS>>\n{_SYSTEM_INSTRUCTION}\n<</SYS>>\n\n"
        f"{user_content} [/INST]"
    )
    return prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_rag_output(raw_output: str) -> Dict[str, Any]:
    """Extract and validate the JSON block from RAG model output.

    Identical strategy to baseline parsing, but also extracts
    ``cited_chunk_ids``.

    Args:
        raw_output: The full decoded string returned by the model.

    Returns:
        Dict with keys: risk_score (int), risk_level (str), key_signals (list),
        citations (list), cited_chunk_ids (list), reasoning (str).
    """
    result = dict(_DEFAULT_OUTPUT)

    if not raw_output:
        logger.warning("parse_rag_output received empty string.")
        return result

    json_str: str = ""

    # Strategy 1: fenced code block
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1)

    # Strategy 2: first { to last }
    if not json_str:
        brace_match = re.search(r"(\{.*\})", raw_output, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(1)

    if not json_str:
        json_str = raw_output.strip()

    try:
        parsed: Dict[str, Any] = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "RAG JSON parsing failed: %s. Raw output (first 200 chars): %s",
            exc,
            raw_output[:200],
        )
        return result

    score = parsed.get("risk_score", result["risk_score"])
    if isinstance(score, (int, float)):
        result["risk_score"] = max(0, min(100, int(score)))

    level = parsed.get("risk_level", result["risk_level"])
    if isinstance(level, str) and level.lower() in {"low", "medium", "high"}:
        result["risk_level"] = level.lower()
    else:
        s = result["risk_score"]
        result["risk_level"] = "low" if s < 35 else ("high" if s >= 65 else "medium")

    signals = parsed.get("key_signals", [])
    if isinstance(signals, list):
        result["key_signals"] = [str(s) for s in signals]

    citations = parsed.get("citations", [])
    if isinstance(citations, list):
        result["citations"] = [str(c) for c in citations]

    chunk_ids = parsed.get("cited_chunk_ids", [])
    if isinstance(chunk_ids, list):
        result["cited_chunk_ids"] = [str(c) for c in chunk_ids]

    reasoning = parsed.get("reasoning", "")
    if isinstance(reasoning, str) and reasoning:
        result["reasoning"] = reasoning

    return result
