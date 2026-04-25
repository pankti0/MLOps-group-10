"""
build_embeddings.py — Build and save a FAISS index from processed 10-K sections.

Usage:
    python scripts/build_embeddings.py
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


# Paths
LABELS_CSV = os.path.join(_REPO_ROOT, "data", "labels", "company_labels.csv")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "data", "embeddings")
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss.index")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.json")

SECTIONS_TO_EMBED = ["item_1a", "item_7", "item_8"]

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _select_embedding_device() -> str:
    """Pick embedding device with CUDA preference and safe fallback.

    Order:
    1. EMBED_DEVICE env var (if set)
    2. CUDA if available
    3. CPU fallback
    """
    env_device = os.getenv("EMBED_DEVICE", "").strip()
    if env_device:
        logger.info("Using EMBED_DEVICE override: %s", env_device)
        return env_device

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA is available; using GPU for embeddings.")
            return "cuda"
    except ImportError:
        logger.warning("torch not installed; falling back to CPU for embeddings.")

    logger.info("CUDA not available; using CPU for embeddings.")
    return "cpu"


# -----------------------------------------------------------
# Chunking
# -----------------------------------------------------------

def chunk_text_safe(text: str, ticker: str, company_name: str, section: str) -> list:
    """Chunk text WITHOUT injecting identity into embeddings."""

    metadata = {
        "ticker": ticker,
        "company_name": company_name,
        "section": section,
    }

    try:
        from src.data.preprocessor import chunk_text  # type: ignore

        chunks = chunk_text(
            text=text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            metadata=metadata,
        )

        # ✅ Only add metadata, DO NOT modify text
        for i, c in enumerate(chunks):
            if isinstance(c.get("metadata"), dict):
                c["metadata"]["chunk_index"] = i
            else:
                c["metadata"] = {**metadata, "chunk_index": i}

        return chunks

    except (ImportError, AttributeError):
        pass

    logger.warning(
        "preprocessor.chunk_text not available; using built-in chunker for %s/%s.",
        ticker,
        section,
    )

    return _simple_chunk(text, ticker, company_name, section, metadata)


def _simple_chunk(
    text: str, ticker: str, company_name: str, section: str, metadata: dict
) -> list:
    """Character-based chunker WITHOUT identity injection."""

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
                    "text": chunk_text_str,  # ✅ CLEAN TEXT ONLY
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


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main() -> None:
    """Build and save FAISS index."""
    import pandas as pd

    from src.embeddings.embedder import Embedder
    from src.embeddings.faiss_store import FAISSStore

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    if not os.path.isfile(LABELS_CSV):
        logger.error("Labels CSV not found: %s", LABELS_CSV)
        sys.exit(1)

    companies_df = pd.read_csv(LABELS_CSV)
    logger.info("Loaded %d companies.", len(companies_df))

    all_chunks = []
    missing_files = []

    for _, row in companies_df.iterrows():
        ticker = str(row["ticker"])
        company_name = str(row.get("company_name", ticker))
        sections_path = os.path.join(PROCESSED_DIR, f"{ticker}_sections.json")

        if not os.path.isfile(sections_path):
            logger.warning("Missing sections for %s", ticker)
            missing_files.append(ticker)
            continue

        try:
            with open(sections_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sections = data.get("sections", data)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", ticker, exc)
            missing_files.append(ticker)
            continue

        company_chunks = 0

        for section in SECTIONS_TO_EMBED:
            text = sections.get(section, "") or ""
            if not text.strip():
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
        logger.error("No chunks produced.")
        sys.exit(1)

    logger.info("Total chunks: %d", len(all_chunks))

    # Build embeddings (prefer GPU when available)
    device = _select_embedding_device()
    embedder = Embedder(device=device)
    store = FAISSStore(dimension=embedder.get_dimension())

    store.build_index(all_chunks, embedder)
    store.save(INDEX_PATH, METADATA_PATH)

    print("=" * 60)
    print(f"Embedding device      : {device}")
    print(f"Total chunks embedded : {len(all_chunks)}")
    print(f"FAISS index size      : {store._index.ntotal}")
    print(f"Embedding dimension   : {embedder.get_dimension()}")
    print(f"Index saved to        : {INDEX_PATH}")
    print(f"Metadata saved to     : {METADATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()