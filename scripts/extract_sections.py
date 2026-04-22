"""
Script 01 – Extract sections from 10-K PDF files.

For each company defined in configs/data_config.yaml:
  1. Reads the PDF using pdf_extractor.extract_text().
  2. Splits it into sections using section_extractor.extract_sections().
  3. Saves the result as JSON to data/processed/{ticker}_sections.json.
  4. Prints a summary table of characters extracted per section.

Usage:
    python scripts/extract_sections.py
    python scripts/extract_sections.py --config configs/data_config.yaml

Failures (missing PDF, extraction errors) are reported at the end without
stopping the overall run.
"""

import argparse
import json
import logging
import os
import sys
import re
from typing import List
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data.pdf_extractor import extract_text
from src.data.section_extractor import extract_sections


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("01_extract_sections")


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# -------------------------
# 🔥 STRONG VALIDATION
# -------------------------
def is_bad_item_1a(text: str) -> bool:
    if not text:
        return True

    length = len(text)

    if length < 2000:
        return True

    # 🔥 detect truncation
    if length == 50000:
        return True

    # TOC pattern
    if "..." in text[:200]:
        return True

    # 🔥 semantic signal
    start = text[:1000].lower()
    if "risk factors" not in start:
        return True

    return False


# -------------------------
# 🔥 ROBUST RECOVERY
# -------------------------
def recover_item_1a(full_text: str) -> str:
    # 🔥 MUCH stronger header detection
    ITEM_1A = re.compile(
        r'(?:^|\n)\s*item\s*1a[\s\.\-:\n]*risk\s*factors',
        re.IGNORECASE
    )

    # 🔥 stronger boundary detection
    ITEM_BOUNDARY = re.compile(
        r'item\s*1b[\s\.\-:\n]*|item\s*2[\s\.\-:\n]*',
        re.IGNORECASE
    )

    matches_1a = list(ITEM_1A.finditer(full_text))
    matches_boundaries = list(ITEM_BOUNDARY.finditer(full_text))

    logger.info("Found %d Item 1A matches", len(matches_1a))

    best_section = ""
    best_len = 0

    for m in matches_1a:
        start = m.start()

        # find next boundary AFTER this
        end = None
        for b in matches_boundaries:
            if b.start() > start:
                end = b.start()
                break

        if end is None:
            continue

        candidate = full_text[start:end]

        # 🔥 REMOVE OVER-FILTERING (this was killing MSFT)
        if len(candidate) < 3000:
            continue

        # 🔥 IMPORTANT: allow risk factors anywhere in first 3000 chars
        start_text = candidate[:3000].lower()
        if "risk factors" not in start_text:
            continue

        if len(candidate) > best_len:
            best_section = candidate
            best_len = len(candidate)

    return best_section

def _print_summary_table(results: List[dict]) -> None:
    col_widths = {"company": 25, "ticker": 6, "item_1a": 10, "item_7": 10, "item_8": 10, "status": 10}

    header = (
        f"{'Company':<{col_widths['company']}} "
        f"{'Ticker':<{col_widths['ticker']}} "
        f"{'item_1a':>{col_widths['item_1a']}} "
        f"{'item_7':>{col_widths['item_7']}} "
        f"{'item_8':>{col_widths['item_8']}} "
        f"{'Status':<{col_widths['status']}}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for r in results:
        secs = r.get("sections", {})
        print(
            f"{r['company']:<{col_widths['company']}} "
            f"{r['ticker']:<{col_widths['ticker']}} "
            f"{len(secs.get('item_1a', '')):>{col_widths['item_1a']},} "
            f"{len(secs.get('item_7', '')):>{col_widths['item_7']},} "
            f"{len(secs.get('item_8', '')):>{col_widths['item_8']},} "
            f"{r['status']:<{col_widths['status']}}"
        )

    print(sep + "\n")


def main(config_path: str) -> None:
    logger.info("Loading config from: %s", config_path)
    config = _load_config(config_path)

    processed_dir = os.path.join(
        REPO_ROOT,
        config.get("output_paths", {}).get("processed", "data/processed")
    )
    os.makedirs(processed_dir, exist_ok=True)

    companies = config.get("companies", [])
    if not companies:
        logger.error("No companies found in config.")
        sys.exit(1)

    results = []

    for company in companies:
        ticker = company["ticker"]
        name = company["name"]
        pdf_abs_path = os.path.join(REPO_ROOT, company["pdf_path"])

        logger.info("[%s] Processing: %s", ticker, name)

        result = {"company": name, "ticker": ticker, "sections": {}, "status": "OK"}

        if not os.path.isfile(pdf_abs_path):
            result["status"] = "MISSING_PDF"
            results.append(result)
            continue

        raw_text = extract_text(pdf_abs_path)
        if not raw_text:
            result["status"] = "EMPTY_TEXT"
            results.append(result)
            continue

        sections = extract_sections(raw_text)
        full_text = sections.get("full_text", "")

        # -------------------------
        # 🔥 VALIDATION + FORCE FIX
        # -------------------------
        item_1a_text = sections.get("item_1a", "")
        length = len(item_1a_text)

        logger.info("[%s] item_1a BEFORE recovery: %d chars", ticker, length)

        bad_1a = is_bad_item_1a(item_1a_text)

        # 🔥 FORCE recovery if suspicious
        if length == 50000:
            logger.warning("[%s] item_1a = 50000 → forcing recovery", ticker)
            bad_1a = True

        if bad_1a:
            logger.warning("[%s] Recovering item_1a...", ticker)

            recovered = recover_item_1a(full_text)

            if recovered:
                sections["item_1a"] = recovered
                logger.info("[%s] SUCCESS → %d chars", ticker, len(recovered))
            else:
                logger.warning("[%s] Recovery failed → fallback", ticker)
                third = len(full_text) // 3
                sections["item_1a"] = full_text[third: 2 * third]

        result["sections"] = sections

        # save
        output_path = os.path.join(processed_dir, f"{ticker}_sections.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "ticker": ticker,
                    "company": name,
                    "sections": sections,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

        results.append(result)

    _print_summary_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "data_config.yaml"),
    )
    args = parser.parse_args()
    main(config_path=args.config)