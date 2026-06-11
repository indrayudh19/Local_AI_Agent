"""
Configuration constants for the RAG pipeline.
All paths, model names, and tuning parameters are centralized here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
VECTORDB_DIR = os.path.join(DATA_DIR, "vectordb")

# Ensure all data directories exist
for d in [UPLOADS_DIR, CHUNKS_DIR, VECTORDB_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Output dimension for all-MiniLM-L6-v2

# ──────────────────────────────────────────────
# Document Processing
# ──────────────────────────────────────────────
CHUNK_SIZE = 512          # Characters per chunk
CHUNK_OVERLAP = 100       # Overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50     # Discard chunks shorter than this

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
RETRIEVAL_TOP_K = 10      # Number of chunks to retrieve
BM25_WEIGHT = 0.3         # Weight for BM25 keyword score in hybrid fusion
SEMANTIC_WEIGHT = 0.7     # Weight for semantic similarity in hybrid fusion

# ──────────────────────────────────────────────
# NLP Agent
# ──────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"
ANSWER_TOP_N = 3          # Number of top-scored sentences in the final answer
CONFIDENCE_HIGH = 0.55    # Threshold for high confidence
CONFIDENCE_MEDIUM = 0.35  # Threshold for medium confidence
DEDUP_THRESHOLD = 0.90    # Cosine similarity threshold for deduplication
