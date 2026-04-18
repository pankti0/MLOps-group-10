"""
Section extraction utilities for SEC 10-K annual report text.

Extracts Item 1A (Risk Factors), Item 7 (MD&A), and Item 8 (Financial
Statements) using regex with multiple fallback patterns to handle
inconsistent 10-K formatting across filers.

Key design decisions:
- Headers often appear on separate lines in PDF-extracted text:
    "Item 1A.\nRisk Factors\n..." rather than "Item 1A. Risk Factors..."
- The Table of Contents contains the first occurrence of each header with
  no content body, so we find ALL occurrences and return the first with
  substantial content (>500 chars).
- If no occurrence has >500 chars, we return the longest one found.

Exports:
    extract_sections(text: str) -> dict
        Keys: item_1a, item_7, item_8, full_text
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum characters for a section to be considered real content (not TOC)
_MIN_CONTENT_CHARS = 500

# Sentinel: start of the *next* Item or Part header (used as stop boundary).
# Matches "Item 7." / "ITEM 7A." / "Part II" etc. at the start of a line.
_NEXT_ITEM_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:ITEM|Item)\s+\d+[A-Z]?\.?\s*$"
    r"|(?m)^[ \t]*(?:ITEM|Item)\s+\d+[A-Z]?\.\s+\S"
    r"|(?m)^[ \t]*(?:PART|Part)\s+[IVX]+\.?\s*$",
)


def _clean_section_text(text: str) -> str:
    """Remove common PDF extraction artifacts from section text."""
    lines = text.split("\n")
    # Remove lines that are purely numeric (page numbers)
    cleaned_lines = [line for line in lines if not re.match(r"^\s*\d+\s*$", line)]
    text = "\n".join(cleaned_lines)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove form-feed / control characters (keep \n \t)
    text = re.sub(r"[\x0c\x0b\x08\x07]", " ", text)
    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_after_match(text: str, match_end: int) -> str:
    """Extract content from match_end to the next item/part header."""
    remaining = text[match_end:]
    stop_match = _NEXT_ITEM_PATTERN.search(remaining)
    if stop_match:
        section_text = remaining[: stop_match.start()]
    else:
        section_text = remaining
    return _clean_section_text(section_text)


def _find_all_occurrences(text: str, patterns: List[re.Pattern]) -> List[Tuple[int, int, int]]:
    """Find ALL match positions across all patterns.

    Returns list of (pattern_index, match_start, match_end) sorted by match_start.
    """
    occurrences: List[Tuple[int, int, int]] = []
    for i, pattern in enumerate(patterns):
        for m in pattern.finditer(text):
            occurrences.append((i, m.start(), m.end()))
    occurrences.sort(key=lambda x: x[1])
    return occurrences


def _extract_section(text: str, section_key: str) -> str:
    """Find all header occurrences and return the first with substantial content.

    Strategy:
    1. Find every occurrence of the header (across all pattern variants).
    2. For each occurrence in document order, extract content until the next header.
    3. Return the first occurrence whose content exceeds _MIN_CONTENT_CHARS.
    4. If none exceed the threshold, return the longest content found.
    """
    patterns = _SECTION_CONFIGS[section_key]
    occurrences = _find_all_occurrences(text, patterns)

    if not occurrences:
        logger.warning("Section '%s' could not be found in document.", section_key)
        return ""

    best_text = ""
    best_len = 0

    for pat_idx, _start, end in occurrences:
        candidate = _extract_after_match(text, end)
        clen = len(candidate)
        if clen > _MIN_CONTENT_CHARS:
            logger.info(
                "Section '%s' extracted via pattern %d (%d chars)",
                section_key, pat_idx, clen,
            )
            return candidate
        if clen > best_len:
            best_len = clen
            best_text = candidate

    if best_text:
        logger.warning(
            "Section '%s': best content only %d chars (below %d threshold). Using it anyway.",
            section_key, best_len, _MIN_CONTENT_CHARS,
        )
    else:
        logger.warning("Section '%s' matched headers but all content bodies were empty.", section_key)

    return best_text


def _clean_section_text(text: str) -> str:
    """Remove common PDF extraction artifacts from section text."""
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not re.match(r"^\s*\d+\s*$", line)]
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\x0c\x0b\x08\x07]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Per-section pattern lists — order matters only as a tiebreaker when two
# patterns match at the same position. The "find all occurrences" strategy
# means we try every pattern across the whole document.
# ---------------------------------------------------------------------------

_ITEM_1A_PATTERNS = [
    # "Item 1A. Risk Factors" on one line (standard SEC format, with period)
    re.compile(r"(?m)^[ \t]*Item\s+1A[\.\s]\s*Risk\s+Factors\s*$", re.IGNORECASE),
    # "Item 1A" (no period) followed by "Risk Factors" on the next line
    re.compile(r"(?m)^[ \t]*Item\s+1A\s*\n[ \t]*Risk\s+Factors\s*$", re.IGNORECASE),
    # "ITEM 1A." on its own line followed by "Risk Factors" on the next
    re.compile(r"(?m)^[ \t]*ITEM\s+1A\.?\s*\n\s*Risk\s+Factors\s*$", re.IGNORECASE),
    # "Item 1A." alone on a line (header split from title in TOC-style PDFs)
    re.compile(r"(?m)^[ \t]*Item\s+1A\.?\s*$", re.IGNORECASE),
    # Standalone "RISK FACTORS" heading
    re.compile(r"(?m)^[ \t]*RISK\s+FACTORS\s*$"),
]

_ITEM_7_PATTERNS = [
    re.compile(
        r"(?m)^[ \t]*Item\s+7[\.\s]\s*Management[\u2019\u2018's]+\s+Discussion\s+and\s+Analysis.*$",
        re.IGNORECASE,
    ),
    # "Item 7" (no period) then MD&A title on next line
    re.compile(
        r"(?m)^[ \t]*Item\s+7\s*\n[ \t]*Management[\u2019\u2018's]+\s+Discussion",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?m)^[ \t]*ITEM\s+7\.?\s*\n\s*Management[\u2019\u2018's]+\s+Discussion",
        re.IGNORECASE,
    ),
    re.compile(r"(?m)^[ \t]*Item\s+7\.?\s*$", re.IGNORECASE),
    re.compile(
        r"(?m)^[ \t]*MANAGEMENT[\u2019\u2018's]+\s+DISCUSSION\s+AND\s+ANALYSIS\s*$",
        re.IGNORECASE,
    ),
    # Fallback: "Management's Discussion" anywhere as a standalone heading
    re.compile(r"(?m)^[ \t]*Management.s Discussion and Analysis\s*$", re.IGNORECASE),
]

_ITEM_8_PATTERNS = [
    re.compile(
        r"(?m)^[ \t]*Item\s+8[\.\s]\s*Financial\s+Statements.*$",
        re.IGNORECASE,
    ),
    # "Item 8" (no period) then Financial Statements on next line
    re.compile(
        r"(?m)^[ \t]*Item\s+8\s*\n[ \t]*Financial\s+Statements",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?m)^[ \t]*ITEM\s+8\.?\s*\n\s*Financial\s+Statements",
        re.IGNORECASE,
    ),
    re.compile(r"(?m)^[ \t]*Item\s+8\.?\s*$", re.IGNORECASE),
    re.compile(r"(?m)^[ \t]*FINANCIAL\s+STATEMENTS\s*$"),
]

_SECTION_CONFIGS: Dict[str, List[re.Pattern]] = {
    "item_1a": _ITEM_1A_PATTERNS,
    "item_7": _ITEM_7_PATTERNS,
    "item_8": _ITEM_8_PATTERNS,
}


def extract_sections(text: str) -> Dict[str, str]:
    """Extract key 10-K sections from full document text.

    Attempts multiple regex patterns per section to handle varied 10-K
    formatting. Sections that cannot be found are returned as empty strings
    with a warning logged.

    Args:
        text: Full text of the 10-K document (as returned by pdf_extractor).

    Returns:
        Dictionary with keys:
            - ``item_1a``: Risk Factors section text.
            - ``item_7``:  MD&A section text.
            - ``item_8``:  Financial Statements section text.
            - ``full_text``: The original full document text (unchanged).
    """
    if not text:
        logger.warning("extract_sections received empty text.")
        return {"item_1a": "", "item_7": "", "item_8": "", "full_text": ""}

    sections: Dict[str, str] = {"full_text": text}

    for key in ("item_1a", "item_7", "item_8"):
        sections[key] = _extract_section(text, key)

    extracted = {k: v for k, v in sections.items() if k != "full_text" and v}
    logger.info(
        "Section extraction complete. Found: %s",
        list(extracted.keys()) if extracted else "none",
    )

    return sections


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo_root)

    from src.data.pdf_extractor import extract_text

    import glob

    pattern = os.path.join(repo_root, "10-k forms", "*.pdf")
    pdf_files = sorted(glob.glob(pattern))

    if not pdf_files:
        print(f"No PDF files found at: {pattern}")
        sys.exit(1)

    first_pdf = pdf_files[0]
    print(f"Testing section extraction on: {first_pdf}\n")

    text = extract_text(first_pdf)
    sections = extract_sections(text)

    for key, content in sections.items():
        if key == "full_text":
            print(f"full_text: {len(content)} characters")
        else:
            print(f"{key}: {len(content)} characters")
            if content:
                print(f"  Preview: {content[:200].strip()!r}\n")
            else:
                print("  (not found)\n")
