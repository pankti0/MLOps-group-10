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
import time
from typing import Dict, List

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
    """Load and return the YAML config."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _section_summary(sections: Dict[str, str]) -> Dict[str, int]:
    """Return a dict mapping section key → character count."""
    return {k: len(v) for k, v in sections.items() if k != "full_text"}


def _print_summary_table(results: List[dict]) -> None:
    """Print a formatted summary table to stdout."""
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

    processed_dir = os.path.join(REPO_ROOT, config.get("output_paths", {}).get("processed", "data/processed"))
    os.makedirs(processed_dir, exist_ok=True)
    logger.info("Output directory: %s", processed_dir)

    companies = config.get("companies", [])
    if not companies:
        logger.error("No companies found in config.")
        sys.exit(1)

    results = []
    failures = []

    for company in companies:
        ticker = company["ticker"]
        name = company["name"]
        pdf_rel_path = company["pdf_path"]
        pdf_abs_path = os.path.join(REPO_ROOT, pdf_rel_path)

        logger.info("[%s] Processing: %s", ticker, name)
        t_start = time.time()

        result = {"company": name, "ticker": ticker, "sections": {}, "status": "OK"}

        #  1. Extract PDF text 
        if not os.path.isfile(pdf_abs_path):
            msg = f"PDF not found: {pdf_abs_path}"
            logger.error("[%s] %s", ticker, msg)
            result["status"] = "MISSING_PDF"
            failures.append({"ticker": ticker, "company": name, "error": msg})
            results.append(result)
            continue

        raw_text = extract_text(pdf_abs_path)
        if not raw_text:
            msg = "PDF extraction returned empty text"
            logger.warning("[%s] %s", ticker, msg)
            result["status"] = "EMPTY_TEXT"
            failures.append({"ticker": ticker, "company": name, "error": msg})
            results.append(result)
            continue

        logger.info("[%s] Extracted %d characters from PDF", ticker, len(raw_text))

        #  2. Extract sections
        try:
            sections = extract_sections(raw_text)
        except Exception as exc:
            msg = f"section_extractor raised: {exc}"
            logger.error("[%s] %s", ticker, msg)
            result["status"] = "SECTION_ERROR"
            failures.append({"ticker": ticker, "company": name, "error": msg})
            results.append(result)
            continue

       
        full_text = sections.get("full_text", "")
        any_missing = False
        for key in ("item_1a", "item_7"):
            if not sections.get(key):
                logger.warning("[%s] Section '%s' could not be extracted.", ticker, key)
                any_missing = True
        if any_missing and full_text and not any(sections.get(k) for k in ("item_1a", "item_7", "item_8")):
           
            third = len(full_text) // 3
            sections["item_1a"] = full_text[third: 2 * third][:50000]
            sections["item_7"] = full_text[2 * third:][:50000]
            logger.info("[%s] No sections found — using full_text thirds as fallback.", ticker)
        result["sections"] = sections

        
        for key in ("item_1a", "item_7"):
            if not sections.get(key):
                result["status"] = "PARTIAL"

        #  3. Save to JSON
        output_path = os.path.join(processed_dir, f"{ticker}_sections.json")
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ticker": ticker,
                        "company": name,
                        "pdf_path": pdf_rel_path,
                        "label": company["label"],
                        "risk_category": company["risk_category"],
                        "risk_score": company["risk_score"],
                        "sections": sections,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
            elapsed = time.time() - t_start
            logger.info("[%s] Saved to %s (%.1fs)", ticker, output_path, elapsed)
        except OSError as exc:
            msg = f"Failed to write JSON: {exc}"
            logger.error("[%s] %s", ticker, msg)
            result["status"] = "WRITE_ERROR"
            failures.append({"ticker": ticker, "company": name, "error": msg})

        results.append(result)

    # 4. Summary 
    _print_summary_table(results)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    partial_count = sum(1 for r in results if r["status"] == "PARTIAL")
    fail_count = len(failures)

    print(f"Completed: {ok_count} OK, {partial_count} partial, {fail_count} failed "
          f"out of {len(companies)} companies.\n")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  [{f['ticker']}] {f['company']}: {f['error']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 10-K sections from PDF files.")
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "data_config.yaml"),
        help="Path to data_config.yaml (default: configs/data_config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
