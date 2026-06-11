"""
RAG Pipeline orchestrator.
Ties together ingestion, embedding, vector storage, retrieval, and the NLP agent
into a single easy-to-use interface.
"""

import os

from rag.config import UPLOADS_DIR
from rag.embeddings import EmbeddingEngine
from rag.ingest import ingest_pdf, get_all_chunks, delete_chunks_for_file
from rag.vectorstore import VectorStore
from rag.retriever import HybridRetriever
from rag.nlp_agent import NLPAgent
from rag.config import RETRIEVAL_TOP_K


class RAGPipeline:
    """
    Main orchestrator for the RAG chatbot system.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest("path/to/document.pdf")
        result = pipeline.query("What is the main topic?")
        print(result["answer"])
    """

    def __init__(self):
        print("[RAGPipeline] Initializing components...")
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store, self.embedding_engine)
        self.agent = NLPAgent(self.embedding_engine)
        print("[RAGPipeline] Ready.")

    def ingest(self, filepath):
        """
        Ingest a PDF document into the system.

        Steps:
        1. Extract text and chunk the PDF
        2. Generate embeddings for each chunk
        3. Store embeddings + metadata in the FAISS vector store

        Args:
            filepath: Absolute path to the PDF file.

        Returns:
            dict with ingestion statistics.
        """
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}

        filename = os.path.basename(filepath)

        # Check if already ingested — remove old data first
        existing_sources = self.vector_store.get_all_sources()
        if filename in existing_sources:
            print(f"[RAGPipeline] Re-ingesting {filename}. Removing old data...")
            self.vector_store.delete_by_source(filename)
            delete_chunks_for_file(filename)

        # Step 1: Extract and chunk
        chunks = ingest_pdf(filepath)
        if not chunks:
            return {"error": f"No text could be extracted from {filename}"}

        # Step 2: Generate embeddings
        chunk_texts = [c["text"] for c in chunks]
        print(f"[RAGPipeline] Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = self.embedding_engine.encode(chunk_texts, show_progress=True)

        # Step 3: Store in vector database
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save()

        return {
            "filename": filename,
            "chunks": len(chunks),
            "status": "success"
        }

    def query(self, question):
        """
        Answer a question using the RAG pipeline.

        Steps:
        1. Retrieve relevant chunks via hybrid search
        2. Pass chunks to the NLP agent for extractive answer synthesis

        Args:
            question: The user's question string.

        Returns:
            dict with "answer", "sources", "confidence", "confidence_label"
        """
        if self.vector_store.total_vectors == 0:
            return {
                "answer": "No documents have been uploaded yet. "
                          "Please upload a PDF to get started!",
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "none"
            }

        # Step 1: Retrieve
        all_chunks = get_all_chunks()
        retrieved = self.retriever.retrieve(
            question,
            top_k=RETRIEVAL_TOP_K,
            all_chunks=all_chunks
        )

        # Step 2: Reason and answer
        result = self.agent.answer(question, retrieved)
        return result

    def list_documents(self):
        """
        List all ingested documents.

        Returns:
            List of source filenames.
        """
        return self.vector_store.get_all_sources()

    def delete_document(self, filename):
        """
        Remove a document from the system entirely.

        Args:
            filename: The document filename to remove.

        Returns:
            dict with status.
        """
        # Remove from vector store
        self.vector_store.delete_by_source(filename)

        # Remove chunk JSON
        delete_chunks_for_file(filename)

        # Remove the PDF itself
        pdf_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[RAGPipeline] Deleted PDF: {pdf_path}")

        return {"filename": filename, "status": "deleted"}

    def get_stats(self):
        """
        Return system statistics.

        Returns:
            dict with vector count, document count, etc.
        """
        sources = self.vector_store.get_all_sources()
        return {
            "total_vectors": self.vector_store.total_vectors,
            "total_documents": len(sources),
            "documents": sources
        }
