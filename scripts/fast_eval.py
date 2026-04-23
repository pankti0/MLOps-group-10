import json
import re

def parse_rag_output(raw_output: str) -> dict:
    """
    Robust parser for RAG outputs.

    Handles:
    - JSON outputs
    - Plain text with "Risk Score: X"
    - Fallback safely
    """

    if not raw_output or len(raw_output.strip()) == 0:
        return _fallback("Empty model output")

    raw_output = raw_output.strip()

    # -----------------------------
    # 1. Try JSON parse
    # -----------------------------
    try:
        data = json.loads(raw_output)
        return {
            "risk_score": _normalize_score(data.get("risk_score")),
            "risk_level": data.get("risk_level", "medium"),
            "key_signals": data.get("key_signals", []),
            "citations": data.get("citations", []),
        }
    except Exception:
        pass

    # -----------------------------
    # 2. Extract score via regex
    # -----------------------------
    score_match = re.search(
        r"risk[_\s-]*score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        raw_output,
        re.IGNORECASE
    )

    if score_match:
        score = _normalize_score(score_match.group(1))
    else:
        # fallback: any number
        generic_match = re.search(r"([0-9]{1,3})", raw_output)
        score = _normalize_score(generic_match.group(1)) if generic_match else 50

    # -----------------------------
    # 3. Extract risk level
    # -----------------------------
    text = raw_output.lower()

    if "high" in text:
        level = "high"
    elif "low" in text:
        level = "low"
    else:
        level = "medium"

    return {
        "risk_score": score,
        "risk_level": level,
        "key_signals": [],
        "citations": [],
    }


def _normalize_score(value):
    try:
        val = float(value)
        if val <= 1:
            val *= 100
        return round(min(max(val, 0), 100), 2)
    except:
        return 50


def _fallback(reason: str):
    return {
        "risk_score": 50,
        "risk_level": "medium",
        "key_signals": [],
        "citations": [],
        "reasoning": reason,
    }