"""
build_embeddings.py — Build and save a FAISS index from processed 10-K sections.

Usage:
    python scripts/build_embeddings.py

The script:
  1. Loads all processed sections from data/processed/{ticker}_sections.json
  2. Chunks each section with preprocessor.chunk_text (chunk_size=512, overlap=64)
  3. Tags each chunk with metadata: {ticker, company_name, section, chunk_index}
  4. Builds a FAISS index via FAISSStore
  5. Saves the index to data/embeddings/faiss.index and data/embeddings/chunk_metadata.json
  6. Prints total chunks embedded and index size
"""

import json
import logging
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_embeddings")


# Setting Paths

LABELS_CSV = os.path.join(_REPO_ROOT, "data", "labels", "company_labels.csv")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "data", "embeddings")
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss.index")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.json")

# Sections to embed 
SECTIONS_TO_EMBED = ["item_1a", "item_7", "item_8"]

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text_safe(text: str, ticker: str, company_name: str, section: str) -> list:
    """Chunk text using the data pipeline's preprocessor if available.

    Falls back to a simple character-based chunker if the preprocessor
    module is not present (e.g. running in isolation).

    Args:
        text: Raw section text.
        ticker: Company ticker symbol.
        company_name: Full company name.
        section: Section name (e.g. "item_1a").

    Returns:
        List of chunk dicts with keys: text, chunk_id, metadata, char_start, char_end.
    """
    metadata = {
        "ticker": ticker,
        "company_name": company_name,
        "section": section,
    }

    # Try to use the project preprocessor
    try:
        from src.data.preprocessor import chunk_text  # type: ignore

        chunks = chunk_text(
            text=text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            metadata=metadata,
        )
        # Inject chunk_index into metadata for convenience
        for i, c in enumerate(chunks):
            if isinstance(c.get("metadata"), dict):
                c["metadata"]["chunk_index"] = i
            else:
                c["metadata"] = {**metadata, "chunk_index": i}
        return chunks
    except (ImportError, AttributeError):
        pass

    # Fallback: simple character-based chunker
    logger.warning(
        "preprocessor.chunk_text not available; using built-in chunker for %s/%s.",
        ticker,
        section,
    )
    return _simple_chunk(text, ticker, company_name, section, metadata)


def _simple_chunk(
    text: str, ticker: str, company_name: str, section: str, metadata: dict
) -> list:
    """Character-based text chunker used as a fallback."""
    chunks = []
    start = 0
    chunk_index = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunk_id = f"{ticker}_{section}_{chunk_index}"
            chunks.append(
                {
                    "text": chunk_text_str,
                    "chunk_id": chunk_id,
                    "metadata": {**metadata, "chunk_index": chunk_index},
                    "char_start": start,
                    "char_end": end,
                }
            )
            chunk_index += 1

        start += step
        if start >= len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: build and save FAISS index from all processed 10-K sections."""
    import pandas as pd

    from src.embeddings.embedder import Embedder
    from src.embeddings.faiss_store import FAISSStore

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Load company list
    if not os.path.isfile(LABELS_CSV):
        logger.error("Labels CSV not found: %s", LABELS_CSV)
        sys.exit(1)

    companies_df = pd.read_csv(LABELS_CSV)
    logger.info("Loaded %d companies from '%s'.", len(companies_df), LABELS_CSV)

    # Collect all chunks
    all_chunks = []
    missing_files = []

    for _, row in companies_df.iterrows():
        ticker = str(row["ticker"])
        company_name = str(row.get("company_name", ticker))
        sections_path = os.path.join(PROCESSED_DIR, f"{ticker}_sections.json")

        if not os.path.isfile(sections_path):
            logger.warning("Sections file not found for %s: %s", ticker, sections_path)
            missing_files.append(ticker)
            continue

        try:
            with open(sections_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Sections are nested under a "sections" key
            sections = data.get("sections", data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load sections for %s: %s", ticker, exc)
            missing_files.append(ticker)
            continue

        company_chunks = 0
        for section in SECTIONS_TO_EMBED:
            text = sections.get(section, "") or ""
            if not text.strip():
                logger.debug("No text for %s/%s, skipping.", ticker, section)
                continue

            chunks = chunk_text_safe(
                text=text,
                ticker=ticker,
                company_name=company_name,
                section=section,
            )
            all_chunks.extend(chunks)
            company_chunks += len(chunks)
            logger.info("  %s / %-8s -> %d chunks", ticker, section, len(chunks))

        logger.info("%s: %d total chunks.", ticker, company_chunks)

    if not all_chunks:
        logger.error("No chunks produced. Verify that processed sections exist in '%s'.", PROCESSED_DIR)
        sys.exit(1)

    logger.info("Total chunks to embed: %d", len(all_chunks))

    if missing_files:
        logger.warning("Skipped tickers (no sections file): %s", ", ".join(missing_files))

    # Build embedder and FAISS store
    embedder = Embedder()
    store = FAISSStore(dimension=embedder.get_dimension())
    store.build_index(all_chunks, embedder)


    store.save(INDEX_PATH, METADATA_PATH)


    print(f"  Total chunks embedded : {len(all_chunks)}")
    print(f"  FAISS index size      : {store._index.ntotal} vectors")  # type: ignore[union-attr]
    print(f"  Embedding dimension   : {embedder.get_dimension()}")
    print(f"  Index saved to        : {INDEX_PATH}")
    print(f"  Metadata saved to     : {METADATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
