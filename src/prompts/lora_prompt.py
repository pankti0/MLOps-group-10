"""
Prompt templates for LoRA fine-tuning of Mistral-7B for credit risk analysis.

Uses Mistral instruct format:  [INST] ... [/INST]

Exports:
    build_lora_prompt(company_name, ticker, item_1a, item_7) -> str
    build_training_example(company_name, ticker, item_1a, item_7,
                           ideal_output) -> dict
"""

from __future__ import annotations

import json
from typing import Any, Dict

# ---------------------------------------------------------------------------
# System instruction shared between inference and training
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """You are a credit risk analyst specializing in SEC 10-K filings. \
Your task is to assess the credit risk of a company based on excerpts from its annual report.

Analyze the provided Risk Factors (Item 1A) and Management Discussion & Analysis (Item 7) \
sections and return a structured JSON assessment.

Your response MUST be valid JSON with exactly these fields:
{
  "predicted_score": <float 0-100, where 100 = maximum credit risk>,
  "risk_level": <"low" | "medium" | "high">,
  "key_signals": [<list of up to 5 specific risk signals found in the text>],
  "citations": [<list of up to 3 direct quotes from the source text supporting your assessment>],
  "rationale": "<one paragraph explaining the overall credit risk assessment>"
}

Scoring guide:
  0–33  → low risk    (strong financials, stable business, low debt burden)
  34–66 → medium risk (mixed signals, moderate leverage, some concerns)
  67–100 → high risk  (distressed financials, going concern issues, high leverage)"""


def _truncate(text: str, max_chars: int = 4000) -> str:
    """Truncate text to max_chars, appending an ellipsis if truncated.

    Args:
        text: Input text string.
        max_chars: Maximum character length.

    Returns:
        Possibly truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


def build_lora_prompt(
    company_name: str,
    ticker: str,
    item_1a: str,
    item_7: str,
) -> str:
    """Build a Mistral instruct-format prompt for credit risk analysis.

    Args:
        company_name: Full company name (e.g. "Apple Inc.").
        ticker: Stock ticker symbol (e.g. "AAPL").
        item_1a: Text of Item 1A (Risk Factors) from the 10-K.
        item_7: Text of Item 7 (MD&A) from the 10-K.

    Returns:
        Complete prompt string in Mistral [INST] ... [/INST] format.
    """
    user_content = f"""{_SYSTEM_INSTRUCTION}

---
Company: {company_name} ({ticker})

### Item 1A — Risk Factors
{_truncate(item_1a)}

### Item 7 — Management's Discussion and Analysis
{_truncate(item_7)}
---

Based on the above 10-K excerpts, provide your structured JSON credit risk assessment for {company_name} ({ticker})."""

    prompt = f"<s>[INST] {user_content} [/INST]"
    return prompt


def build_training_example(
    company_name: str,
    ticker: str,
    item_1a: str,
    item_7: str,
    ideal_output: Dict[str, Any],
) -> dict:
    """Build a training example for SFTTrainer from a company's 10-K data.

    The returned dict has a single ``"text"`` key containing the full
    instruction-response string in Mistral format, as expected by trl's
    SFTTrainer with ``dataset_text_field="text"``.

    Args:
        company_name: Full company name.
        ticker: Stock ticker symbol.
        item_1a: Text of Item 1A from the 10-K.
        item_7: Text of Item 7 from the 10-K.
        ideal_output: Dictionary matching the JSON schema defined in
            ``_SYSTEM_INSTRUCTION``.  Must include at minimum:
            ``predicted_score``, ``risk_level``, ``key_signals``,
            ``citations``, ``rationale``.

    Returns:
        ``{"text": "<full instruction-response string>"}``
    """
    prompt_part = build_lora_prompt(
        company_name=company_name,
        ticker=ticker,
        item_1a=item_1a,
        item_7=item_7,
    )

    # Ensure the ideal output is valid JSON
    response_json = json.dumps(ideal_output, indent=2, ensure_ascii=False)

    # Mistral format: [INST] ... [/INST] response </s>
    # prompt_part already ends with [/INST]
    full_text = f"{prompt_part}\n{response_json}</s>"

    return {"text": full_text}
