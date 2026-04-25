"""
Risk score utilities shared across model and evaluation code.

This module centralizes score parsing and score-to-label mapping so every
approach uses the same policy and avoids silent contract drift.
"""

from __future__ import annotations

from typing import Any, Dict


DEFAULT_DEFAULT_THRESHOLD = 70.0


def parse_predicted_score(payload: Dict[str, Any], default: float = 50.0) -> float:
    """Parse a canonical predicted score from model payload.

    Supports backward compatibility with older keys (risk_score).
    """
    if not isinstance(payload, dict):
        return float(default)

    raw = payload.get("predicted_score", payload.get("risk_score", default))
    try:
        score = float(raw)
    except (TypeError, ValueError):
        score = float(default)
    return max(0.0, min(100.0, score))


def score_to_label(score: float, threshold: float = DEFAULT_DEFAULT_THRESHOLD) -> int:
    """Convert a score to binary default label."""
    return 1 if float(score) >= float(threshold) else 0


def score_to_risk_level(score: float) -> str:
    """Convert a score to low/medium/high risk level."""
    if score < 35:
        return "low"
    if score < 70:
        return "medium"
    return "high"
