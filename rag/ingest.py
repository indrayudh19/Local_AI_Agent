"""
Document ingestion module.
Handles PDF text extraction, cleaning, and chunking.
"""

import os
import re
import json
import hashlib
from datetime import datetime

import fitz  # PyMuPDF

from rag.config import (
    UPLOADS_DIR, CHUNKS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
)


def extract_text_from_pdf(filepath):
    """
    Extract text from a PDF file, page by page.

    Args:
        filepath: Absolute path to the PDF file.

    Returns:
        List of dicts: [{"page": int, "text": str}, ...]
    """
    pages = []
    doc = fitz.open(filepath)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text
            })
    doc.close()
    return pages


def clean_text(text):
    """
    Clean extracted PDF text.
    - Normalize whitespace
    - Remove excessive newlines
    - Strip non-printable characters
    """
    # Remove non-printable characters (keep newlines and tabs temporarily)
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    # Collapse multiple spaces (but preserve single newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def chunk_text(text, source_file, page_num, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        source_file: Original filename for metadata.
        page_num: Page number for metadata.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunk dicts with text, metadata.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Try to break at a sentence boundary (., !, ?)
        if end < text_length:
            # Look for the last sentence-ending punctuation within the chunk
            last_period = text.rfind('.', start + chunk_size // 2, end)
            last_question = text.rfind('?', start + chunk_size // 2, end)
            last_exclaim = text.rfind('!', start + chunk_size // 2, end)
            best_break = max(last_period, last_question, last_exclaim)

            if best_break > start:
                end = best_break + 1  # Include the punctuation

        chunk_text_content = text[start:end].strip()

        if len(chunk_text_content) >= MIN_CHUNK_LENGTH:
            chunk_id = hashlib.md5(
                f"{source_file}:{page_num}:{start}".encode()
            ).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk_text_content,
                "source": source_file,
                "page": page_num,
                "start_char": start,
                "end_char": end
            })

        start = end - overlap if end < text_length else text_length

    return chunks


def ingest_pdf(filepath):
    """
    Full ingestion pipeline for a single PDF:
    1. Extract text page-by-page
    2. Clean the text
    3. Chunk into overlapping segments
    4. Save chunks as JSON

    Args:
        filepath: Path to the PDF file.

    Returns:
        List of all chunk dicts from this document.
    """
    filename = os.path.basename(filepath)
    print(f"[Ingest] Processing: {filename}")

    # Extract text
    pages = extract_text_from_pdf(filepath)
    if not pages:
        print(f"[Ingest] Warning: No text extracted from {filename}")
        return []

    # Process each page
    all_chunks = []
    for page_data in pages:
        cleaned = clean_text(page_data["text"])
        if cleaned:
            page_chunks = chunk_text(
                cleaned,
                source_file=filename,
                page_num=page_data["page"]
            )
            all_chunks.extend(page_chunks)

    print(f"[Ingest] Extracted {len(all_chunks)} chunks from {len(pages)} pages.")

    # Save chunks to JSON
    chunks_file = os.path.join(CHUNKS_DIR, f"{os.path.splitext(filename)[0]}.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            "source": filename,
            "ingested_at": datetime.now().isoformat(),
            "total_chunks": len(all_chunks),
            "chunks": all_chunks
        }, f, indent=2, ensure_ascii=False)

    print(f"[Ingest] Saved chunks to: {chunks_file}")
    return all_chunks


def get_all_chunks():
    """
    Load all previously ingested chunks from the chunks directory.

    Returns:
        List of all chunk dicts across all documents.
    """
    all_chunks = []
    for filename in os.listdir(CHUNKS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(CHUNKS_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_chunks.extend(data.get("chunks", []))
    return all_chunks


def delete_chunks_for_file(source_filename):
    """
    Delete the chunk JSON file for a given source document.

    Args:
        source_filename: The original PDF filename.
    """
    base = os.path.splitext(source_filename)[0]
    chunks_file = os.path.join(CHUNKS_DIR, f"{base}.json")
    if os.path.exists(chunks_file):
        os.remove(chunks_file)
        print(f"[Ingest] Deleted chunks for: {source_filename}")
