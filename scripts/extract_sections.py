"""
Script 01 – Extract sections from 10-K PDF files.
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

    text_clean = text.strip()
    length = len(text_clean)

    if length < 1500:
        return True

    if length == 50000:
        return True

    first_200 = text_clean[:200].lower()

    # header-only detection
    if "risk factors" in first_200:
        if text_clean.count(".") < 5:
            return True

    if "..." in first_200:
        return True

    if "risk factors" not in text_clean[:1000].lower():
        return True

    return False


# -------------------------
# 🔥 RECOVERY FUNCTIONS
# -------------------------
def recover_item_1a(full_text: str) -> str:
    ITEM_1A = re.compile(
        r'(?:^|\n)\s*item\s*1a[\s\.\-:\n]*risk\s*factors',
        re.IGNORECASE
    )

    ITEM_BOUNDARY = re.compile(
        r'item\s*1b[\s\.\-:\n]*|item\s*2[\s\.\-:\n]*',
        re.IGNORECASE
    )

    matches_1a = list(ITEM_1A.finditer(full_text))
    matches_boundaries = list(ITEM_BOUNDARY.finditer(full_text))

    best_section = ""
    best_len = 0

    for m in matches_1a:
        start = m.start()

        end = None
        for b in matches_boundaries:
            if b.start() > start:
                end = b.start()
                break

        if end is None:
            continue

        candidate = full_text[start:end]

        if len(candidate) < 3000:
            continue

        if "risk factors" not in candidate[:3000].lower():
            continue

        if len(candidate) > best_len:
            best_section = candidate
            best_len = len(candidate)

    return best_section


def recover_section(full_text: str, item_label: str, next_items: list) -> str:
    item_pattern = re.compile(
        rf'(?:^|\n)\s*{item_label}[\s\.\-:\n]*',
        re.IGNORECASE
    )

    boundary_pattern = re.compile(
        "|".join([rf'{ni}[\s\.\-:\n]*' for ni in next_items]),
        re.IGNORECASE
    )

    matches = list(item_pattern.finditer(full_text))
    boundaries = list(boundary_pattern.finditer(full_text))

    best = ""
    best_len = 0

    for m in matches:
        start = m.start()

        end = None
        for b in boundaries:
            if b.start() > start:
                end = b.start()
                break

        if end is None:
            continue

        candidate = full_text[start:end]

        if len(candidate) < 5000:
            continue

        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)

    return best


# -------------------------
# PRINT TABLE
# -------------------------
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


# -------------------------
# MAIN
# -------------------------
def main(config_path: str) -> None:
    logger.info("Loading config from: %s", config_path)
    config = _load_config(config_path)

    processed_dir = os.path.join(
        REPO_ROOT,
        config.get("output_paths", {}).get("processed", "data/processed")
    )
    os.makedirs(processed_dir, exist_ok=True)

    companies = config.get("companies", [])
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
        # ITEM 1A FIX
        # -------------------------
        item_1a_text = sections.get("item_1a", "")
        length = len(item_1a_text)

        logger.info("[%s] item_1a BEFORE: %d chars", ticker, length)
        logger.info("[%s] item_1a PREVIEW: %s", ticker, item_1a_text[:200].replace("\n", " "))

        bad_1a = is_bad_item_1a(item_1a_text)

        if length < 3000:
            bad_1a = True

        if bad_1a:
            recovered = recover_item_1a(full_text)
            if recovered:
                sections["item_1a"] = recovered

        # -------------------------
        # ITEM 7 FIX
        # -------------------------
        if not sections.get("item_7") or len(sections["item_7"]) < 5000:
            recovered_7 = recover_section(full_text, "item 7", ["item 7a", "item 8"])
            if recovered_7:
                sections["item_7"] = recovered_7

        # -------------------------
        # ITEM 8 FIX
        # -------------------------
        if not sections.get("item_8") or len(sections["item_8"]) < 5000:
            recovered_8 = recover_section(full_text, "item 8", ["item 9"])
            if recovered_8:
                sections["item_8"] = recovered_8

        result["sections"] = sections

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