"""
Text preprocessing and chunking utilities for 10-K documents.

Exports:
    chunk_text(text, chunk_size, overlap, metadata) -> List[dict]
    clean_text(text) -> str
"""

import logging
import re
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Approximate characters per token (GPT-style BPE tokenisation).
_CHARS_PER_TOKEN: int = 4

# Default minimum chunk length in characters before a chunk is accepted.
_MIN_CHUNK_CHARS: int = 100


def clean_text(text: str) -> str:
    """Clean raw PDF-extracted text for downstream processing.

    Operations performed:
    - Remove isolated page-number lines (lines containing only digits).
    - Remove common 10-K header/footer boilerplate (e.g. "Table of Contents").
    - Collapse runs of blank lines to a single blank line.
    - Remove form-feed and other non-printable control characters.
    - Collapse intra-line whitespace runs to a single space.
    - Strip leading and trailing whitespace.

    Args:
        text: Raw text string (e.g. from pdf_extractor.extract_text).

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove form-feed / vertical-tab / bell characters
    text = re.sub(r"[\x0c\x0b\x08\x07]", " ", text)

    # Remove lines that are solely page numbers
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip pure digit lines (page numbers) and common boilerplate
        if re.match(r"^\d+$", stripped):
            continue
        if re.match(r"^(Table\s+of\s+Contents|INDEX|index)$", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs within a line → single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _token_approx(text: str) -> int:
    """Return an approximate token count using a fixed chars-per-token ratio.

    Args:
        text: Input string.

    Returns:
        Estimated token count (integer).
    """
    return max(1, len(text) // _CHARS_PER_TOKEN)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: Optional[Dict] = None,
) -> List[Dict]:
    """Split text into overlapping chunks based on approximate token count.

    The chunking algorithm works character-by-character using the
    ``_CHARS_PER_TOKEN`` constant to convert token counts to character
    counts. Chunks shorter than 100 characters are silently dropped.

    Chunk IDs are built from metadata fields ``ticker`` and ``section``
    when present (format: ``{ticker}_{section}_{index}``). Otherwise a
    UUID4-based ID is used.

    Args:
        text: Input text to chunk (should already be cleaned).
        chunk_size: Target chunk size in tokens (default 512).
        overlap: Overlap between consecutive chunks in tokens (default 64).
        metadata: Optional dict attached to every chunk. Expected keys
            (if present): ``ticker`` (str), ``section`` (str).
            Additional keys are preserved as-is.

    Returns:
        List of chunk dicts, each with keys:
            - ``text`` (str): Chunk text.
            - ``chunk_id`` (str): Unique identifier for the chunk.
            - ``metadata`` (dict): Copy of input metadata (never None).
            - ``char_start`` (int): Start character index in original text.
            - ``char_end`` (int): End character index in original text.

    Raises:
        ValueError: If ``chunk_size`` <= 0 or ``overlap`` >= ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    if not text:
        logger.warning("chunk_text received empty text; returning empty list.")
        return []

    meta = dict(metadata) if metadata else {}
    ticker = meta.get("ticker", "")
    section = meta.get("section", "")

    # Convert token counts to character counts
    chunk_chars = chunk_size * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN
    step_chars = chunk_chars - overlap_chars

    chunks: List[Dict] = []
    start = 0
    total_len = len(text)
    index = 0

    while start < total_len:
        end = min(start + chunk_chars, total_len)
        chunk_text_str = text[start:end]

        if len(chunk_text_str) >= _MIN_CHUNK_CHARS:
            # Build chunk_id
            if ticker and section:
                chunk_id = f"{ticker}_{section}_{index}"
            elif ticker:
                chunk_id = f"{ticker}_{index}"
            else:
                chunk_id = f"{uuid.uuid4().hex}_{index}"

            chunks.append(
                {
                    "text": chunk_text_str,
                    "chunk_id": chunk_id,
                    "metadata": dict(meta),
                    "char_start": start,
                    "char_end": end,
                }
            )
            index += 1
        else:
            logger.debug(
                "Skipping short chunk at char %d-%d (%d chars < min %d)",
                start,
                end,
                len(chunk_text_str),
                _MIN_CHUNK_CHARS,
            )

        if end == total_len:
            break
        start += step_chars

    logger.info(
        "chunk_text: produced %d chunks from %d chars (chunk_size=%d tokens, overlap=%d tokens)",
        len(chunks),
        total_len,
        chunk_size,
        overlap,
    )
    return chunks


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo_root)

    sample_text = (
        "This is a sample 10-K text for testing purposes. " * 200
    )

    cleaned = clean_text(sample_text)
    print(f"clean_text output length: {len(cleaned)} chars")

    meta = {"ticker": "AAPL", "section": "item_1a", "company": "Apple"}
    chunks = chunk_text(cleaned, chunk_size=512, overlap=64, metadata=meta)

    print(f"Number of chunks: {len(chunks)}")
    if chunks:
        c = chunks[0]
        print(f"First chunk_id : {c['chunk_id']}")
        print(f"First chunk chars: {c['char_start']} – {c['char_end']}")
        print(f"First chunk text preview: {c['text'][:120]!r}")
