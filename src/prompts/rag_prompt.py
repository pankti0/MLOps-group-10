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
- You MUST NOT output anything except valid JSON.
- You MUST NOT summarize entire passages.
- You MUST extract only relevant risk signals.

If you cannot find sufficient evidence, return a neutral (medium risk) assessment.
"""

_ANALYSIS_TASK = """\
Analyse the credit risk of {company_name} ({ticker}) using ONLY the passages below.

IMPORTANT:
- Only extract signals related to THIS company.
- Ignore any irrelevant or generic text.
- Be concise and evidence-driven.

Focus ONLY on:
1. Debt and leverage
2. Liquidity and cash position
3. Revenue decline or volatility
4. Going concern warnings
5. Covenant breaches
6. Severe risk factors impacting repayment

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON
- No explanations before or after
- No markdown
- No text outside JSON

JSON FORMAT:
{{
  "risk_score": <0-100>,
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

        # 🔥 truncate long chunks (prevents model rambling)
        text = text[:1200]

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

REMEMBER:
- Output ONLY JSON
- Do NOT repeat passages
- Do NOT explain outside JSON

[/INST]"""

    return prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_rag_output(raw_output: str) -> Dict[str, Any]:
    result = dict(_DEFAULT_OUTPUT)

    if not raw_output:
        logger.warning("parse_rag_output received empty string.")
        return result

    # 🔥 HARD JSON extraction (fixes your parsing issue)
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)

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

    # Validate fields
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