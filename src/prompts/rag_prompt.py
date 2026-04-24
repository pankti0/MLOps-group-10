"""
Prompt templates and output parsing for the RAG-based credit-risk agent.
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
# Thresholds aligned with dataset
# ---------------------------------------------------------------------------

def _score_to_risk_level(score: int) -> str:
    if score < 35:
        return "low"
    elif score < 70:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a highly disciplined credit risk analyst.

STRICT RULES:
- You MUST use ONLY the provided passages.
- You MUST NOT use prior knowledge.
- You MUST output ONLY valid JSON.
- You MUST NOT repeat passages.
- You MUST extract only relevant risk signals.
"""

_ANALYSIS_TASK = """\
Analyse the credit risk of {company_name} ({ticker}) using ONLY the passages below.

IMPORTANT:
- Only extract signals related to THIS company.
- Ignore irrelevant or generic text.
- Be concise and evidence-driven.
- DO NOT default to "medium" unless there is NO signal at all.
- Make a BEST estimate from available evidence.

CRITICAL SCORING RULE (STRICT):

If the company shows ANY of the following:
- substantial indebtedness
- liquidity constraints
- inability to fund operations for the next 12 months
- negative cash flow
- financial distress or losses

You MUST assign a HIGH risk score (70–100).

These are SEVERE credit risk indicators and MUST NOT be classified as medium.

If MULTIPLE strong risk signals are present:
→ risk_score MUST be between 75–90.

Only use MEDIUM risk (35–69) when:
- signals are moderate
- and do NOT threaten solvency

DO NOT be conservative when strong risk signals are present.

SCORING GUIDELINES:
- Low risk (0–34): strong balance sheet, low debt, strong liquidity
- Medium risk (35–69): moderate risk, stable
- High risk (70–100): distress, liquidity issues, high leverage

Focus ONLY on:
1. Debt and leverage
2. Liquidity and cash position
3. Revenue decline or volatility
4. Going concern warnings
5. Covenant breaches
6. Severe risk factors impacting repayment

IMPORTANT SIGNAL HANDLING:
- Do NOT rely on isolated numerical values unless they clearly indicate financial risk.
- Prefer interpreting the financial meaning (e.g., high leverage, refinancing pressure, inability to meet obligations).

DATA TYPES:
- risk_score MUST be an integer (0-100)
- risk_level MUST be one of: low, medium, high
- key_signals MUST be a list of strings
- citations MUST be a list of strings
- cited_chunk_ids MUST be a list of strings

REASONING REQUIREMENT:
reasoning MUST clearly explain:
- what risk signals were found
- why they increase or decrease risk
- how they affect the final risk_score

JSON FORMAT:
{{
  "risk_score": <integer 0-100>,
  "risk_level": "<low|medium|high>",
  "key_signals": ["..."],
  "citations": ["..."],
  "cited_chunk_ids": ["..."],
  "reasoning": "..."
}}
"""


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        metadata = chunk.get("metadata", {})
        section = metadata.get("section", "unknown") if isinstance(metadata, dict) else "unknown"
        text = chunk.get("text", "").strip()

        text = text[:600]

        lines.append(f"[{chunk_id}] (section: {section})\n{text}")

    return "\n\n".join(lines)


def build_rag_prompt(
    company_name: str,
    ticker: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:

    task = _ANALYSIS_TASK.format(company_name=company_name, ticker=ticker)

    passages_text = _format_chunks(retrieved_chunks) if retrieved_chunks else "[No relevant passages retrieved]"

    prompt = f"""<s>[INST] <<SYS>>
{_SYSTEM_INSTRUCTION}
<</SYS>>

{task}

PASSAGES:
{passages_text}

=====================
END OF PASSAGES
=====================

FINAL INSTRUCTION:

Return EXACTLY ONE JSON object.

STRICT:
- No explanation
- No extra text
- No markdown
- No repetition

Your response MUST:
- start with "{{"
- end with "}}"
- be valid JSON

Return ONLY JSON.
"""

    return prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_rag_output(raw_output: str) -> Dict[str, Any]:
    result = dict(_DEFAULT_OUTPUT)

    if not raw_output:
        logger.warning("parse_rag_output received empty string.")
        return result

    json_match = re.search(r"\{.*?\}", raw_output, re.DOTALL)

    if not json_match:
        logger.warning("No JSON found. Raw output: %s", raw_output[:200])
        return result

    json_str = json_match.group(0)

    try:
        parsed: Dict[str, Any] = json.loads(json_str)
    except Exception as exc:
        logger.warning(
            "RAG JSON parsing failed: %s. Raw output: %s",
            exc,
            raw_output[:200],
        )
        return result

    # ------------------------
    # Extract score
    # ------------------------
    score = parsed.get("risk_score", result["risk_score"])
    if isinstance(score, (int, float)):
        score = max(0, min(100, int(score)))
    else:
        score = result["risk_score"]

    result["risk_score"] = score

    # Align with dataset thresholds
    result["risk_level"] = _score_to_risk_level(score)

    # ------------------------
    # Other fields
    # ------------------------
    result["key_signals"] = [str(x) for x in parsed.get("key_signals", [])]
    result["citations"] = [str(x) for x in parsed.get("citations", [])]
    result["cited_chunk_ids"] = [str(x) for x in parsed.get("cited_chunk_ids", [])]

    reasoning = parsed.get("reasoning", "")
    if isinstance(reasoning, str) and reasoning:
        result["reasoning"] = reasoning

    if score == 50:
        logger.warning("Weak prediction (score=50) — likely low signal.")

    return result