"""
Embedding engine using sentence-transformers (all-MiniLM-L6-v2).
Provides a lazy-loaded singleton so the model is only loaded once.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from rag.config import EMBEDDING_MODEL_NAME


class EmbeddingEngine:
    """Wraps sentence-transformers for efficient embedding generation."""

    _instance = None

    def __new__(cls):
        """Singleton pattern — reuse the same model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            print(f"[EmbeddingEngine] Loading model: {EMBEDDING_MODEL_NAME}...")
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("[EmbeddingEngine] Model loaded successfully.")
        return self._model

    def encode(self, texts, batch_size=64, show_progress=False):
        """
        Encode a list of texts into normalized embedding vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Show a progress bar during encoding.

        Returns:
            numpy.ndarray of shape (len(texts), EMBEDDING_DIMENSION).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query):
        """
        Encode a single query string.

        Args:
            query: The query string.

        Returns:
            numpy.ndarray of shape (1, EMBEDDING_DIMENSION).
        """
        return self.encode([query])
