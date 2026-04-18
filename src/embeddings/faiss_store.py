"""
FAISS-based vector store for chunk retrieval.

Exports:
    FAISSStore — builds, persists, and queries a FAISS flat inner-product index.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.embeddings.embedder import Embedder

logger = logging.getLogger(__name__)


class FAISSStore:
    """Stores text chunk embeddings in a FAISS IndexFlatIP for fast retrieval.

    Vectors are L2-normalised before insertion so that inner-product (IP)
    search equals cosine similarity.

    Args:
        dimension: Dimensionality of the embedding vectors (default 768 for
            BAAI/bge-base-en-v1.5).
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self._index: Optional[Any] = None  # faiss.IndexFlatIP
        self._chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict[str, Any]], embedder: Embedder) -> None:
        """Embed all chunks and build the FAISS index.

        Each chunk dict must contain at minimum a ``text`` key. All other
        fields are preserved in the metadata store and returned during
        queries.

        Args:
            chunks: List of chunk dicts produced by preprocessor.chunk_text.
            embedder: Embedder instance used to produce embeddings.
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        if not chunks:
            logger.warning("build_index() called with empty chunk list.")
            self._index = faiss.IndexFlatIP(self.dimension)
            self._chunks = []
            return

        texts = [c["text"] for c in chunks]
        logger.info("Building FAISS index from %d chunks.", len(chunks))

        embeddings = embedder.embed(texts)

        # Normalise to unit length (idempotent if embedder already normalised)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)  # type: ignore[arg-type]

        self._index = index
        self._chunks = list(chunks)

        logger.info(
            "FAISS index built: %d vectors, dimension=%d.", index.ntotal, self.dimension
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, metadata_path: str) -> None:
        """Persist the FAISS index and chunk metadata to disk.

        Args:
            index_path: File path for the FAISS index (e.g. ``faiss.index``).
            metadata_path: File path for the JSON metadata file.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss-cpu is not installed.") from exc

        if self._index is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        faiss.write_index(self._index, index_path)
        logger.info("FAISS index saved to '%s'.", index_path)

        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(self._chunks, fh, ensure_ascii=False, indent=2)
        logger.info("Chunk metadata (%d entries) saved to '%s'.", len(self._chunks), metadata_path)

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load a previously saved FAISS index and chunk metadata from disk.

        Args:
            index_path: Path to the saved FAISS index file.
            metadata_path: Path to the JSON metadata file.
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss-cpu is not installed.") from exc

        self._index = faiss.read_index(index_path)
        self.dimension = self._index.d
        logger.info(
            "FAISS index loaded from '%s': %d vectors, dimension=%d.",
            index_path,
            self._index.ntotal,
            self.dimension,
        )

        with open(metadata_path, "r", encoding="utf-8") as fh:
            self._chunks = json.load(fh)
        logger.info(
            "Chunk metadata loaded from '%s': %d entries.", metadata_path, len(self._chunks)
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        embedder: Embedder,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return the top-k most relevant chunks for a query.

        Args:
            query_text: Natural language query string.
            embedder: Embedder instance (must match the one used to build the index).
            k: Maximum number of results to return.

        Returns:
            List of chunk dicts, each augmented with a ``score`` (float) key,
            sorted from highest to lowest score.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if self._index is None:
            raise RuntimeError("Index is not ready. Call build_index() or load() first.")

        if not query_text:
            logger.warning("query() called with empty query string.")
            return []

        query_vec = embedder.embed_query(query_text)
        # Normalise
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        actual_k = min(k, self._index.ntotal)
        if actual_k == 0:
            logger.warning("Index is empty; returning no results.")
            return []

        scores, indices = self._index.search(query_vec, actual_k)  # type: ignore[arg-type]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        logger.debug("Query returned %d results (k=%d).", len(results), k)
        return results
