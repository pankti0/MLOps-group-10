"""
Text embedding utilities using sentence-transformers.

Exports:
    Embedder — wraps a sentence-transformer model to produce dense embeddings.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# BGE models expect a query instruction prefix during retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_BATCH_SIZE = 64


class Embedder:
    """Embed text using a sentence-transformer model.

    Supports BGE-family models (BAAI/bge-*) with automatic query-prefix
    injection and L2 normalisation so that inner-product search equals
    cosine similarity.

    Args:
        model_name: HuggingFace model identifier (default BAAI/bge-base-en-v1.5).
        device: Torch device string, e.g. "cpu", "cuda", "mps".
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._is_bge = "bge" in model_name.lower()

        logger.info("Loading embedding model '%s' on device '%s'", model_name, device)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc

        logger.info("Embedding model loaded (dim=%d).", self.get_dimension())

    def get_dimension(self) -> int:
        """Return the embedding dimension of the underlying model."""
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed(self, texts: List[str], batch_size: int = _BGE_BATCH_SIZE) -> np.ndarray:
        """Embed a list of texts into dense vectors.

        Vectors are L2-normalised when the underlying model is a BGE model so
        that inner-product search is equivalent to cosine similarity.

        Args:
            texts: List of strings to embed. Empty list returns a zero-row
                array with shape (0, dimension).
            batch_size: Number of texts to encode per forward pass.

        Returns:
            Float32 numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            dim = self.get_dimension()
            logger.warning("embed() called with empty list; returning (0, %d) array.", dim)
            return np.zeros((0, dim), dtype=np.float32)

        logger.info("Embedding %d texts (batch_size=%d).", len(texts), batch_size)

        try:
            from tqdm import tqdm as _tqdm  # noqa: F401

            show_progress = len(texts) > batch_size
        except ImportError:
            show_progress = False

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self._is_bge,
            convert_to_numpy=True,
        )

        logger.info("Embedding complete; shape=%s.", embeddings.shape)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        BGE models prepend a task instruction prefix before encoding the query
        to improve retrieval relevance.

        Args:
            query: The query string.

        Returns:
            Float32 numpy array of shape (embedding_dim,).
        """
        if not query:
            logger.warning("embed_query() called with empty string.")
            return np.zeros(self.get_dimension(), dtype=np.float32)

        if self._is_bge:
            query = _BGE_QUERY_PREFIX + query

        vector: np.ndarray = self._model.encode(
            [query],
            normalize_embeddings=self._is_bge,
            convert_to_numpy=True,
        )
        return vector[0].astype(np.float32)
