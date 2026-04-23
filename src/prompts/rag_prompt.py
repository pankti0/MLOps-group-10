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
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a highly disciplined credit risk analyst.

STRICT RULES:
- You MUST use ONLY the provided passages.
- You MUST NOT use prior knowledge.
- You MUST output ONLY valid JSON.
- You MUST NOT continue or repeat the passages.
- You MUST extract only relevant risk signals.
"""

_ANALYSIS_TASK = """\
Analyse the credit risk of {company_name} ({ticker}) using ONLY the passages below.

IMPORTANT:
- Only extract signals related to THIS company.
- Ignore irrelevant or generic text.
- Be concise and evidence-driven.

Focus ONLY on:
1. Debt and leverage
2. Liquidity and cash position
3. Revenue decline or volatility
4. Going concern warnings
5. Covenant breaches
6. Severe risk factors impacting repayment

DATA TYPES:
- risk_score MUST be an integer (0-100)
- risk_level MUST be one of: low, medium, high
- key_signals MUST be a list of strings
- citations MUST be a list of strings
- cited_chunk_ids MUST be a list of strings
- reasoning MUST be a string

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

        # truncate long chunks
        text = text[:800]  # 🔥 reduced to improve model focus

        lines.append(f"[{chunk_id}] (section: {section})\n{text}")

    return "\n\n".join(lines)


def build_rag_prompt(
    company_name: str,
    ticker: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:

    task = _ANALYSIS_TASK.format(company_name=company_name, ticker=ticker)

    if retrieved_chunks:
        passages_text = _format_chunks(retrieved_chunks)
    else:
        passages_text = "[No relevant passages retrieved]"

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

You MUST return EXACTLY ONE JSON object.

DO NOT:
- add explanations
- add text before or after
- repeat passages
- include markdown

If you do not follow this format, the output will be discarded.

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

    # 🔥 Non-greedy JSON extraction (important fix)
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

    # ------------------------------------------------------------------
    # Safe extraction + validation
    # ------------------------------------------------------------------

    score = parsed.get("risk_score", result["risk_score"])
    if isinstance(score, (int, float)):
        result["risk_score"] = max(0, min(100, int(score)))

    level = parsed.get("risk_level", result["risk_level"])
    if isinstance(level, str) and level.lower() in {"low", "medium", "high"}:
        result["risk_level"] = level.lower()
    else:
        s = result["risk_score"]
        result["risk_level"] = "low" if s < 35 else ("high" if s >= 65 else "medium")

    result["key_signals"] = [str(x) for x in parsed.get("key_signals", [])]
    result["citations"] = [str(x) for x in parsed.get("citations", [])]
    result["cited_chunk_ids"] = [str(x) for x in parsed.get("cited_chunk_ids", [])]

    reasoning = parsed.get("reasoning", "")
    if isinstance(reasoning, str) and reasoning:
        result["reasoning"] = reasoning

    return result