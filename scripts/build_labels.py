"""
Script 02 – Build the company labels CSV.

Reads configs/data_config.yaml and writes data/labels/company_labels.csv,
then prints the resulting DataFrame to stdout.

Usage:
    python scripts/build_labels.py
    python scripts/build_labels.py --config configs/data_config.yaml \
                                       --output data/labels/company_labels.csv
"""

import argparse
import logging
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data.label_builder import build_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("02_build_labels")


def main(config_path: str, output_path: str) -> None:
    """Build and display the labels DataFrame.

    Args:
        config_path: Path to data_config.yaml.
        output_path: Destination CSV path.
    """
    logger.info("Config  : %s", config_path)
    logger.info("Output  : %s", output_path)

    df = build_labels(config_path=config_path, output_path=output_path)

    print("Company Labels")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    print(f"\nTotal companies : {len(df)}")
    print(f"High-risk (1)   : {(df['label'] == 1).sum()}")
    print(f"Low-risk  (0)   : {(df['label'] == 0).sum()}")
    print("\nRisk category distribution:")
    print(df["risk_category"].value_counts().to_string())
    print("\nRisk score statistics:")
    print(df["risk_score"].describe().to_string())
    print()
    logger.info("Labels CSV saved to: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the company labels CSV.")
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "data_config.yaml"),
        help="Path to data_config.yaml (default: configs/data_config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "data", "labels", "company_labels.csv"),
        help="Destination CSV path (default: data/labels/company_labels.csv)",
    )
    args = parser.parse_args()
    main(config_path=args.config, output_path=args.output)
