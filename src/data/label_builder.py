"""
Label builder for the Credit Risk Detection project.

Reads company metadata from data_config.yaml and produces a CSV containing
binary risk labels and supporting columns.

Exports:
    build_labels(config_path: str, output_path: str) -> pd.DataFrame
    load_labels(labels_path: str) -> pd.DataFrame
"""

import logging
import os
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = [
    "ticker",
    "company_name",
    "pdf_path",
    "label",
    "risk_category",
    "risk_score",
]


def build_labels(
    config_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Read company entries from data_config.yaml and write a labels CSV.

    Each company in the YAML ``companies`` list is mapped to one row.  The
    ``name`` field from the YAML becomes ``company_name`` in the output.

    Args:
        config_path: Path to ``configs/data_config.yaml`` (absolute or
            relative to the current working directory).
        output_path: Destination path for the CSV file, e.g.
            ``data/labels/company_labels.csv``.  Parent directories are
            created automatically.

    Returns:
        DataFrame with columns: ticker, company_name, pdf_path, label,
        risk_category, risk_score.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        KeyError: If the YAML is missing the ``companies`` key.
        ValueError: If a required field is missing from any company entry.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if "companies" not in config:
        raise KeyError(f"'companies' key missing from config: {config_path}")

    rows = []
    for entry in config["companies"]:
        required_yaml_keys = ("ticker", "name", "pdf_path", "label", "risk_category", "risk_score")
        missing = [k for k in required_yaml_keys if k not in entry]
        if missing:
            raise ValueError(
                f"Company entry {entry.get('ticker', '<unknown>')} is missing fields: {missing}"
            )

        rows.append(
            {
                "ticker": entry["ticker"],
                "company_name": entry["name"],
                "pdf_path": entry["pdf_path"],
                "label": int(entry["label"]),
                "risk_category": entry["risk_category"],
                "risk_score": int(entry["risk_score"]),
            }
        )

    df = pd.DataFrame(rows, columns=_REQUIRED_COLUMNS)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(
        "Labels CSV written to '%s' (%d rows, %d high-risk, %d low-risk)",
        output_path,
        len(df),
        (df["label"] == 1).sum(),
        (df["label"] == 0).sum(),
    )

    return df


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load a previously built labels CSV into a DataFrame.

    Args:
        labels_path: Path to the CSV file produced by :func:`build_labels`.

    Returns:
        DataFrame with the same columns as produced by :func:`build_labels`.

    Raises:
        FileNotFoundError: If ``labels_path`` does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Labels CSV at '{labels_path}' is missing columns: {missing_cols}"
        )

    logger.info("Loaded %d label rows from '%s'", len(df), labels_path)
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_root, "configs", "data_config.yaml")
    output_path = os.path.join(repo_root, "data", "labels", "company_labels.csv")

    print(f"Building labels from: {config_path}")
    print(f"Output path        : {output_path}\n")

    df = build_labels(config_path=config_path, output_path=output_path)

    print("Labels DataFrame:")
    print(df.to_string(index=False))
    print(f"\nClass distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\nRisk categories:\n{df['risk_category'].value_counts().to_string()}")
