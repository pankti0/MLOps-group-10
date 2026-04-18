"""
Prompt templates and output parsing for the baseline (non-RAG) credit-risk agent.

Exports:
    build_baseline_prompt(company_name, ticker, item_1a, item_7, max_chars) -> str
    parse_model_output(raw_output) -> dict
"""

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values returned when parsing fails
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT: Dict[str, Any] = {
    "risk_score": 50,
    "risk_level": "medium",
    "key_signals": [],
    "citations": [],
    "reasoning": "Could not parse model output.",
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are an expert credit risk analyst with 20 years of experience evaluating corporate \
10-K filings for institutional lenders. Your analysis is thorough, evidence-based, and \
strictly grounded in the provided text. You never speculate beyond what the filing states.\
"""

_ANALYSIS_TASK = """\
Analyse the following excerpts from {company_name} ({ticker})'s 10-K filing and assess \
the company's credit risk. Focus on:

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
  "citations": ["<exact verbatim quote from the filing>", ...],
  "reasoning": "<2-3 sentence explanation of your overall assessment>"
}}
```
"""

_SECTION_TEMPLATE = """\
--- ITEM 1A: RISK FACTORS ---
{item_1a}

--- ITEM 7: MANAGEMENT'S DISCUSSION AND ANALYSIS ---
{item_7}
"""


def build_baseline_prompt(
    company_name: str,
    ticker: str,
    item_1a: str,
    item_7: str,
    max_chars: int = 8000,
) -> str:
    """Build a Mistral 7B Instruct-format prompt for baseline credit-risk analysis.

    The combined text of item_1a and item_7 is truncated to ``max_chars``
    characters (split evenly between sections) to avoid exceeding the model's
    context window.

    Args:
        company_name: Full company name, e.g. "Apple Inc.".
        ticker: Stock ticker symbol, e.g. "AAPL".
        item_1a: Text of Item 1A (Risk Factors) from the 10-K.
        item_7: Text of Item 7 (MD&A) from the 10-K.
        max_chars: Maximum combined characters allowed for the filing excerpts.

    Returns:
        A fully formatted ``[INST] ... [/INST]`` prompt string.
    """
    half = max_chars // 2
    item_1a_trunc = item_1a[:half].strip()
    item_7_trunc = item_7[:half].strip()

    sections_text = _SECTION_TEMPLATE.format(
        item_1a=item_1a_trunc or "[Not available]",
        item_7=item_7_trunc or "[Not available]",
    )

    task = _ANALYSIS_TASK.format(company_name=company_name, ticker=ticker)
    user_content = f"{task}\n\n{sections_text}"

    prompt = (
        f"<s>[INST] <<SYS>>\n{_SYSTEM_INSTRUCTION}\n<</SYS>>\n\n"
        f"{user_content} [/INST]"
    )
    return prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_model_output(raw_output: str) -> Dict[str, Any]:
    """Extract and validate the JSON block from model output.

    Tries multiple strategies to locate the JSON object in the raw text
    (fenced code block, bare brace-delimited block, full string parse).
    Returns default values for any missing or invalid fields.

    Args:
        raw_output: The full decoded string returned by the model.

    Returns:
        Dict with keys: risk_score (int), risk_level (str), key_signals (list),
        citations (list), reasoning (str).
    """
    result = dict(_DEFAULT_OUTPUT)

    if not raw_output:
        logger.warning("parse_model_output received empty string.")
        return result

    json_str: str = ""

    # Strategy 1: fenced code block ```json ... ```
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1)

    # Strategy 2: bare JSON object (first { to last })
    if not json_str:
        brace_match = re.search(r"(\{.*\})", raw_output, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(1)

    # Strategy 3: try the whole string
    if not json_str:
        json_str = raw_output.strip()

    try:
        parsed: Dict[str, Any] = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("JSON parsing failed: %s. Raw output (first 200 chars): %s", exc, raw_output[:200])
        return result

    # Merge parsed values with defaults, validating types
    score = parsed.get("risk_score", result["risk_score"])
    if isinstance(score, (int, float)):
        result["risk_score"] = max(0, min(100, int(score)))

    level = parsed.get("risk_level", result["risk_level"])
    if isinstance(level, str) and level.lower() in {"low", "medium", "high"}:
        result["risk_level"] = level.lower()
    elif isinstance(result["risk_score"], int):
        # Derive from score if level is missing / invalid
        s = result["risk_score"]
        result["risk_level"] = "low" if s < 35 else ("high" if s >= 65 else "medium")

    signals = parsed.get("key_signals", [])
    if isinstance(signals, list):
        result["key_signals"] = [str(s) for s in signals]

    citations = parsed.get("citations", [])
    if isinstance(citations, list):
        result["citations"] = [str(c) for c in citations]

    reasoning = parsed.get("reasoning", "")
    if isinstance(reasoning, str) and reasoning:
        result["reasoning"] = reasoning

    return result
