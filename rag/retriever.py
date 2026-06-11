"""
Hybrid retriever combining semantic search (FAISS) with BM25 keyword matching.
Uses Reciprocal Rank Fusion (RRF) to merge results from both strategies.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from rag.embeddings import EmbeddingEngine
from rag.vectorstore import VectorStore
from rag.config import RETRIEVAL_TOP_K, SEMANTIC_WEIGHT, BM25_WEIGHT


class HybridRetriever:
    """
    Retrieves the most relevant document chunks for a query
    using a combination of semantic similarity and keyword matching.
    """

    def __init__(self, vector_store: VectorStore, embedding_engine: EmbeddingEngine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._tfidf_chunks = None

    def _build_tfidf_index(self, chunks):
        """Build or rebuild the TF-IDF index over all chunks."""
        if not chunks:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            self._tfidf_chunks = None
            return

        texts = [c.get("text", "") for c in chunks]
        self._tfidf_chunks = chunks
        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),   # Unigrams + bigrams for better keyword matching
            sublinear_tf=True
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    def _bm25_search(self, query, top_k):
        """
        Perform TF-IDF based keyword search (BM25 approximation).

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of (chunk_dict, score) tuples, sorted by score descending.
        """
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return []

        query_vec = self._tfidf_vectorizer.transform([query])
        scores = sklearn_cosine(query_vec, self._tfidf_matrix).flatten()

        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = dict(self._tfidf_chunks[idx])
                results.append((chunk, float(scores[idx])))

        return results

    def _semantic_search(self, query, top_k):
        """
        Perform semantic vector search via FAISS.

        Args:
            query: The search query string.
            top_k: Number of top results.

        Returns:
            List of (chunk_dict, score) tuples, sorted by score descending.
        """
        query_embedding = self.embedding_engine.encode_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return [(r, r.get("score", 0.0)) for r in results]

    def retrieve(self, query, top_k=RETRIEVAL_TOP_K, all_chunks=None):
        """
        Hybrid retrieval using Reciprocal Rank Fusion.

        Combines rankings from:
        1. Semantic search (FAISS embeddings)
        2. BM25 keyword search (TF-IDF)

        Args:
            query: The user's query string.
            top_k: Number of final results to return.
            all_chunks: Optional list of all chunks (for BM25). If None, BM25 is skipped.

        Returns:
            List of chunk dicts with a "fusion_score" field, sorted by relevance.
        """
        # Semantic search
        semantic_results = self._semantic_search(query, top_k=top_k * 2)

        # BM25 search (if chunks available)
        bm25_results = []
        if all_chunks:
            # Rebuild TF-IDF if needed (chunk corpus may have changed)
            if self._tfidf_chunks is None or len(self._tfidf_chunks) != len(all_chunks):
                self._build_tfidf_index(all_chunks)
            bm25_results = self._bm25_search(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion (RRF)
        # Score = sum( 1 / (k + rank) ) across both result lists
        rrf_k = 60  # Standard RRF constant
        fusion_scores = {}  # chunk_id -> { score, chunk_data }

        for rank, (chunk, score) in enumerate(semantic_results):
            cid = chunk.get("id", str(rank))
            rrf_score = SEMANTIC_WEIGHT / (rrf_k + rank + 1)
            if cid in fusion_scores:
                fusion_scores[cid]["score"] += rrf_score
            else:
                fusion_scores[cid] = {"score": rrf_score, "chunk": chunk}

        for rank, (chunk, score) in enumerate(bm25_results):
            cid = chunk.get("id", f"bm25_{rank}")
            rrf_score = BM25_WEIGHT / (rrf_k + rank + 1)
            if cid in fusion_scores:
                fusion_scores[cid]["score"] += rrf_score
            else:
                fusion_scores[cid] = {"score": rrf_score, "chunk": chunk}

        # Sort by fusion score
        ranked = sorted(fusion_scores.values(), key=lambda x: x["score"], reverse=True)

        # Return top_k results with fusion score attached
        results = []
        for item in ranked[:top_k]:
            chunk = dict(item["chunk"])
            chunk["fusion_score"] = item["score"]
            results.append(chunk)

        return results
