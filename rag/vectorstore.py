"""
Vector store management using FAISS.
Handles creating, loading, searching, and persisting the vector index.
Metadata is stored in a parallel JSON file alongside the FAISS binary index.
"""

import os
import json
import numpy as np
import faiss

from rag.config import VECTORDB_DIR, EMBEDDING_DIMENSION


FAISS_INDEX_PATH = os.path.join(VECTORDB_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTORDB_DIR, "metadata.json")


class VectorStore:
    """Manages a FAISS index with associated chunk metadata."""

    def __init__(self):
        self.index = None
        self.metadata = []  # List of dicts, parallel to FAISS index rows
        self._load_or_create()

    def _load_or_create(self):
        """Load an existing index from disk, or create a new one."""
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            print("[VectorStore] Loading existing index from disk...")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"[VectorStore] Loaded {self.index.ntotal} vectors.")
        else:
            print("[VectorStore] Creating new FAISS index...")
            # IndexFlatIP = Inner Product (equivalent to cosine sim on normalized vectors)
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.metadata = []
            print("[VectorStore] New empty index created.")

    def add(self, embeddings, chunk_metadata_list):
        """
        Add embeddings and their associated metadata to the index.

        Args:
            embeddings: numpy.ndarray of shape (N, EMBEDDING_DIMENSION).
            chunk_metadata_list: List of N metadata dicts (id, text, source, page, etc.)
        """
        if len(embeddings) == 0:
            return

        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(chunk_metadata_list)
        print(f"[VectorStore] Added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding, top_k=10):
        """
        Search for the top_k most similar vectors.

        Args:
            query_embedding: numpy.ndarray of shape (1, EMBEDDING_DIMENSION).
            top_k: Number of results to return.

        Returns:
            List of dicts: [{**metadata, "score": float}, ...]
        """
        if self.index.ntotal == 0:
            return []

        # Clamp top_k to the number of vectors available
        top_k = min(top_k, self.index.ntotal)

        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self):
        """Persist the index and metadata to disk."""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"[VectorStore] Saved {self.index.ntotal} vectors to disk.")

    def delete_by_source(self, source_filename):
        """
        Remove all vectors associated with a specific source document.
        Since FAISS IndexFlatIP doesn't support deletion, we rebuild the index.

        Args:
            source_filename: The source filename to remove.
        """
        # Find indices to keep
        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if m.get("source") != source_filename
        ]

        if len(keep_indices) == len(self.metadata):
            print(f"[VectorStore] No vectors found for source: {source_filename}")
            return

        removed = len(self.metadata) - len(keep_indices)

        if len(keep_indices) == 0:
            # All vectors removed — reset
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.metadata = []
        else:
            # Reconstruct vectors for the indices we want to keep
            old_vectors = np.array(
                [self.index.reconstruct(i) for i in keep_indices],
                dtype=np.float32
            )
            old_metadata = [self.metadata[i] for i in keep_indices]

            # Create a fresh index and re-add
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.index.add(old_vectors)
            self.metadata = old_metadata

        self.save()
        print(f"[VectorStore] Removed {removed} vectors for: {source_filename}. "
              f"Remaining: {self.index.ntotal}")

    @property
    def total_vectors(self):
        """Return the total number of vectors in the index."""
        return self.index.ntotal

    def get_all_sources(self):
        """Return a set of all unique source filenames in the store."""
        return list(set(m.get("source", "unknown") for m in self.metadata))
