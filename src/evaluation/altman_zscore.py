"""
Altman Z-Score computation for credit risk assessment.

Implements the original Altman Z-Score model for public companies:
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets

Risk zones:
    Z < 1.81   → Distress (high risk)
    1.81–2.99  → Grey zone (medium risk)
    Z > 2.99   → Safe (low risk)

Risk score mapping (sigmoid-like, 0–100 where higher = more risky):
    score = 100 * (1 / (1 + exp(Z - 1.81)))

Exports:
    compute_zscore(ticker: str) -> dict
    compute_all_zscores() -> pd.DataFrame
    save_zscore_predictions(output_path: str) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded approximate financial data (public domain, FY 2023/2024)
# ---------------------------------------------------------------------------

COMPANY_FINANCIALS: Dict[str, Dict[str, float]] = {
    "AAPL": {
        "working_capital": 4.5e9,
        "retained_earnings": -214e9,
        "ebit": 114e9,
        "market_cap": 3300e9,
        "total_liabilities": 290e9,
        "total_assets": 352e9,
        "revenue": 391e9,
    },
    "MSFT": {
        "working_capital": 40e9,
        "retained_earnings": 118e9,
        "ebit": 109e9,
        "market_cap": 3100e9,
        "total_liabilities": 225e9,
        "total_assets": 512e9,
        "revenue": 245e9,
    },
    "JNJ": {
        "working_capital": 15e9,
        "retained_earnings": 85e9,
        "ebit": 15e9,
        "market_cap": 380e9,
        "total_liabilities": 94e9,
        "total_assets": 163e9,
        "revenue": 88e9,
    },
    "DAL": {
        "working_capital": -2e9,
        "retained_earnings": -12e9,
        "ebit": 5e9,
        "market_cap": 26e9,
        "total_liabilities": 64e9,
        "total_assets": 73e9,
        "revenue": 58e9,
    },
    "T": {
        "working_capital": -10e9,
        "retained_earnings": -16e9,
        "ebit": 17e9,
        "market_cap": 145e9,
        "total_liabilities": 253e9,
        "total_assets": 402e9,
        "revenue": 123e9,
    },
    "VZ": {
        "working_capital": -9e9,
        "retained_earnings": 9e9,
        "ebit": 19e9,
        "market_cap": 165e9,
        "total_liabilities": 277e9,
        "total_assets": 380e9,
        "revenue": 134e9,
    },
    "F": {
        "working_capital": 15e9,
        "retained_earnings": 22e9,
        "ebit": 6e9,
        "market_cap": 43e9,
        "total_liabilities": 244e9,
        "total_assets": 274e9,
        "revenue": 185e9,
    },
    "AAL": {
        "working_capital": -6e9,
        "retained_earnings": -14e9,
        "ebit": 2e9,
        "market_cap": 8e9,
        "total_liabilities": 53e9,
        "total_assets": 63e9,
        "revenue": 53e9,
    },
    "GME": {
        "working_capital": 1e9,
        "retained_earnings": -3e9,
        "ebit": -0.1e9,
        "market_cap": 9e9,
        "total_liabilities": 1.5e9,
        "total_assets": 4e9,
        "revenue": 5e9,
    },
    "AMC": {
        "working_capital": -0.3e9,
        "retained_earnings": -3e9,
        "ebit": -0.2e9,
        "market_cap": 1.5e9,
        "total_liabilities": 9e9,
        "total_assets": 7e9,
        "revenue": 4.8e9,
    },
}

# Human-readable company names keyed by ticker
_COMPANY_NAMES: Dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JNJ": "Johnson & Johnson",
    "DAL": "Delta Air Lines",
    "T": "AT&T",
    "VZ": "Verizon",
    "F": "Ford Motor Company",
    "AAL": "American Airlines",
    "GME": "GameStop",
    "AMC": "AMC Entertainment",
}


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator, returning 0.0 if denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        Result of division, or 0.0 for zero denominator.
    """
    if denominator == 0.0:
        logger.warning("Division by zero encountered; returning 0.0.")
        return 0.0
    return numerator / denominator


def _risk_zone(z_score: float) -> str:
    """Map a Z-Score to its qualitative risk zone label.

    Args:
        z_score: Computed Altman Z-Score.

    Returns:
        One of 'distress', 'grey', or 'safe'.
    """
    if z_score < 1.81:
        return "distress"
    if z_score <= 2.99:
        return "grey"
    return "safe"


def _risk_score_from_z(z_score: float) -> float:
    """Convert Z-Score to a 0–100 risk score (higher = more risky).

    Uses a sigmoid-like mapping centred on the distress threshold (1.81):
        score = 100 * (1 / (1 + exp(Z - 1.81)))

    Args:
        z_score: Computed Altman Z-Score.

    Returns:
        Risk score in [0, 100].
    """
    return 100.0 * (1.0 / (1.0 + math.exp(z_score - 1.81)))


def compute_zscore(ticker: str) -> dict:
    """Compute the Altman Z-Score for a single company.

    Args:
        ticker: Stock ticker symbol (must be a key in COMPANY_FINANCIALS).

    Returns:
        Dictionary with keys:
            - ``ticker`` (str)
            - ``company_name`` (str)
            - ``z_score`` (float)
            - ``risk_zone`` (str): 'distress', 'grey', or 'safe'
            - ``predicted_score`` (float): 0–100 risk score
            - ``predicted_label`` (int): 1 if distress/grey, 0 if safe
            - ``x1`` through ``x5`` (float): individual ratio components

    Raises:
        KeyError: If ticker is not found in COMPANY_FINANCIALS.
    """
    ticker = ticker.upper()
    if ticker not in COMPANY_FINANCIALS:
        raise KeyError(
            f"Ticker '{ticker}' not found in COMPANY_FINANCIALS. "
            f"Available: {sorted(COMPANY_FINANCIALS)}"
        )

    fin = COMPANY_FINANCIALS[ticker]
    ta = fin["total_assets"]
    tl = fin["total_liabilities"]

    x1 = _safe_divide(fin["working_capital"], ta)
    x2 = _safe_divide(fin["retained_earnings"], ta)
    x3 = _safe_divide(fin["ebit"], ta)
    x4 = _safe_divide(fin["market_cap"], tl)
    x5 = _safe_divide(fin["revenue"], ta)

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    zone = _risk_zone(z)
    score = _risk_score_from_z(z)
    label = 0 if zone == "safe" else 1

    logger.info(
        "Z-Score for %s: Z=%.3f, zone=%s, score=%.1f, label=%d",
        ticker, z, zone, score, label,
    )

    return {
        "ticker": ticker,
        "company_name": _COMPANY_NAMES.get(ticker, ticker),
        "z_score": round(z, 4),
        "risk_zone": zone,
        "predicted_score": round(score, 2),
        "predicted_label": label,
        "x1": round(x1, 4),
        "x2": round(x2, 4),
        "x3": round(x3, 4),
        "x4": round(x4, 4),
        "x5": round(x5, 4),
    }


def compute_all_zscores() -> pd.DataFrame:
    """Compute Altman Z-Scores for all hardcoded companies.

    Returns:
        DataFrame with one row per company and columns:
        ticker, company_name, z_score, risk_zone, predicted_score,
        predicted_label, x1–x5.
    """
    records = []
    for ticker in COMPANY_FINANCIALS:
        try:
            records.append(compute_zscore(ticker))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to compute Z-Score for %s: %s", ticker, exc)

    df = pd.DataFrame(records)
    logger.info("Computed Z-Scores for %d companies.", len(df))
    return df


def save_zscore_predictions(output_path: str) -> pd.DataFrame:
    """Compute Z-Scores and save predictions CSV compatible with the project schema.

    The saved file contains columns required by the evaluation pipeline:
    ticker, company_name, predicted_score, predicted_label, risk_level,
    key_signals, citations, raw_output, approach.

    Args:
        output_path: Absolute or relative path for the output CSV file.
            Parent directories are created if they do not exist.

    Returns:
        The predictions DataFrame (also saved to disk).
    """
    df = compute_all_zscores()

    # Map to common predictions schema
    def _risk_level(zone: str) -> str:
        mapping = {"distress": "high", "grey": "medium", "safe": "low"}
        return mapping.get(zone, "unknown")

    predictions = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "company_name": df["company_name"],
            "predicted_score": df["predicted_score"],
            "predicted_label": df["predicted_label"],
            "risk_level": df["risk_zone"].map(_risk_level),
            "key_signals": df.apply(
                lambda row: str(
                    [
                        f"X1(WC/TA)={row['x1']:.3f}",
                        f"X2(RE/TA)={row['x2']:.3f}",
                        f"X3(EBIT/TA)={row['x3']:.3f}",
                        f"X4(MC/TL)={row['x4']:.3f}",
                        f"X5(Rev/TA)={row['x5']:.3f}",
                    ]
                ),
                axis=1,
            ),
            "citations": [str([]) for _ in range(len(df))],
            "raw_output": df.apply(
                lambda row: (
                    f"Altman Z-Score: {row['z_score']:.4f} | Zone: {row['risk_zone']}"
                ),
                axis=1,
            ),
            "approach": "altman_zscore",
        }
    )

    out_path = output_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    predictions.to_csv(out_path, index=False)
    logger.info("Saved altman_zscore predictions to %s (%d rows)", out_path, len(predictions))
    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = compute_all_zscores()
    print(df[["ticker", "z_score", "risk_zone", "predicted_score", "predicted_label"]].to_string(index=False))
