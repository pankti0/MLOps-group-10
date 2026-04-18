"""
PDF text extraction utilities using PyMuPDF (fitz).

Exports:
    extract_text(pdf_path: str) -> str
    extract_text_by_page(pdf_path: str) -> List[str]
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file as a single string.

    Pages are joined with newline characters. If extraction fails,
    an empty string is returned and a warning is logged.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Extracted text as a single string, or empty string on failure.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")
        return ""

    if not os.path.isfile(pdf_path):
        logger.warning("PDF file not found: %s", pdf_path)
        return ""

    try:
        pages = extract_text_by_page(pdf_path)
        full_text = "\n".join(pages)
        logger.info(
            "Extracted %d characters from %d pages in '%s'",
            len(full_text),
            len(pages),
            pdf_path,
        )
        return full_text
    except Exception as exc:
        logger.warning("Failed to extract text from '%s': %s", pdf_path, exc)
        return ""


def extract_text_by_page(pdf_path: str) -> List[str]:
    """Extract text from a PDF file, returning one string per page.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of strings, one per page. Returns empty list on failure.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")
        return []

    if not os.path.isfile(pdf_path):
        logger.warning("PDF file not found: %s", pdf_path)
        return []

    pages: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")  # type: ignore[arg-type]
            pages.append(text)
        doc.close()
        logger.info("Extracted %d pages from '%s'", len(pages), pdf_path)
    except Exception as exc:
        logger.warning("Failed to extract pages from '%s': %s", pdf_path, exc)

    return pages


if __name__ == "__main__":
    import glob
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pattern = os.path.join(repo_root, "10-k forms", "*.pdf")
    pdf_files = sorted(glob.glob(pattern))

    if not pdf_files:
        print(f"No PDF files found matching: {pattern}")
        sys.exit(1)

    first_pdf = pdf_files[0]
    print(f"Testing extraction on: {first_pdf}")

    text = extract_text(first_pdf)
    print(f"Total characters extracted: {len(text)}")
    print(f"First 500 characters:\n{text[:500]}")

    pages = extract_text_by_page(first_pdf)
    print(f"\nTotal pages: {len(pages)}")
    if pages:
        print(f"Page 1 preview ({len(pages[0])} chars):\n{pages[0][:300]}")
