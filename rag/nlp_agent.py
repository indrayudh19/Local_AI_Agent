"""
NLP-based reasoning agent — the "brain" of the RAG system.
Performs extractive answer synthesis using multi-signal scoring.
No LLM is used. All reasoning is done via classical NLP techniques:
  - Query type classification
  - Named Entity Recognition (spaCy)
  - Sentence-level semantic similarity
  - Keyword overlap scoring
  - Entity-type matching
  - Confidence estimation
"""

import re
import spacy
import numpy as np
from collections import Counter

from rag.embeddings import EmbeddingEngine
from rag.config import (
    SPACY_MODEL, ANSWER_TOP_N,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, DEDUP_THRESHOLD
)


class NLPAgent:
    """
    An NLP-based reasoning agent that extracts answers from retrieved document
    chunks without using any large language model.
    """

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self._nlp = None

        # Question type → expected spaCy entity labels
        self.EXPECTED_ENTITY_MAP = {
            "who":    ["PERSON", "ORG", "NORP"],
            "where":  ["GPE", "LOC", "FAC"],
            "when":   ["DATE", "TIME", "EVENT"],
            "how_many": ["CARDINAL", "QUANTITY", "PERCENT", "MONEY"],
            "what":   [],  # Too broad to constrain
            "why":    [],
            "how":    [],
            "definition": [],
            "yes_no": [],
        }

    @property
    def nlp(self):
        """Lazy-load the spaCy model."""
        if self._nlp is None:
            print(f"[NLPAgent] Loading spaCy model: {SPACY_MODEL}...")
            self._nlp = spacy.load(SPACY_MODEL)
            print("[NLPAgent] spaCy model loaded.")
        return self._nlp

    # ─────────────────────────────────────────────
    # 1. Query Analysis
    # ─────────────────────────────────────────────

    def analyze_query(self, query):
        """
        Analyze the user's query to determine:
        - Question type (who, what, when, where, why, how, yes_no, definition)
        - Key entities and noun phrases
        - Expected answer entity types
        - Important keywords

        Returns:
            dict with analysis results.
        """
        doc = self.nlp(query)
        query_lower = query.lower().strip()

        # Classify question type
        q_type = self._classify_question(query_lower)

        # Extract entities
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]

        # Extract noun phrases (key concepts)
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Extract important keywords (nouns, proper nouns, adjectives)
        keywords = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN", "ADJ")
            and not token.is_stop
            and len(token.text) > 2
        ]

        # Expected answer entity types
        expected_entities = self.EXPECTED_ENTITY_MAP.get(q_type, [])

        return {
            "original": query,
            "type": q_type,
            "entities": entities,
            "noun_phrases": noun_phrases,
            "keywords": keywords,
            "expected_entity_types": expected_entities
        }

    def _classify_question(self, query_lower):
        """Classify the question type based on keyword patterns."""
        # Remove leading articles and common prefixes
        q = re.sub(r'^(can you |please |could you |tell me )', '', query_lower)

        if re.match(r'^(who|whom)\b', q):
            return "who"
        elif re.match(r'^where\b', q):
            return "where"
        elif re.match(r'^when\b', q):
            return "when"
        elif re.match(r'^(how many|how much)\b', q):
            return "how_many"
        elif re.match(r'^how\b', q):
            return "how"
        elif re.match(r'^why\b', q):
            return "why"
        elif re.match(r'^(what is|what are|what does|define|explain)\b', q):
            return "definition"
        elif re.match(r'^what\b', q):
            return "what"
        elif re.match(r'^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b', q):
            return "yes_no"
        else:
            return "what"  # Default to "what" for statements/commands

    # ─────────────────────────────────────────────
    # 2. Multi-Signal Sentence Scoring
    # ─────────────────────────────────────────────

    def score_sentences(self, query_analysis, retrieved_chunks):
        """
        Score individual sentences from retrieved chunks using multiple signals.

        Signals:
        1. Semantic similarity (embedding cosine sim between query and sentence)
        2. Keyword overlap (what fraction of query keywords appear in the sentence)
        3. Entity match (does the sentence contain the expected entity type)
        4. Position bias (slight boost for earlier sentences in a chunk)

        Args:
            query_analysis: Output of analyze_query().
            retrieved_chunks: List of chunk dicts from the retriever.

        Returns:
            List of scored sentence dicts, sorted by total score descending.
        """
        if not retrieved_chunks:
            return []

        # Split chunks into sentences
        candidate_sentences = []
        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "")
            doc = self.nlp(chunk_text)
            sentences = list(doc.sents)

            for sent_idx, sent in enumerate(sentences):
                sent_text = sent.text.strip()
                if len(sent_text) < 15:  # Skip very short fragments
                    continue

                candidate_sentences.append({
                    "text": sent_text,
                    "source": chunk.get("source", "unknown"),
                    "page": chunk.get("page", 0),
                    "chunk_score": chunk.get("fusion_score", 0.0),
                    "position_in_chunk": sent_idx / max(len(sentences), 1),
                    "entities": [
                        {"text": ent.text, "label": ent.label_}
                        for ent in sent.ents
                    ]
                })

        if not candidate_sentences:
            return []

        # ── Signal 1: Semantic similarity ──
        query_embedding = self.embedding_engine.encode_query(query_analysis["original"])
        sent_texts = [s["text"] for s in candidate_sentences]
        sent_embeddings = self.embedding_engine.encode(sent_texts)

        semantic_scores = np.dot(sent_embeddings, query_embedding.T).flatten()

        # ── Signal 2: Keyword overlap ──
        query_keywords = set(query_analysis["keywords"])
        keyword_scores = []
        for sent in candidate_sentences:
            sent_lower = sent["text"].lower()
            if query_keywords:
                overlap = sum(1 for kw in query_keywords if kw in sent_lower)
                keyword_scores.append(overlap / len(query_keywords))
            else:
                keyword_scores.append(0.0)

        # ── Signal 3: Entity type match ──
        expected_types = set(query_analysis["expected_entity_types"])
        entity_scores = []
        for sent in candidate_sentences:
            if expected_types:
                matches = sum(
                    1 for ent in sent["entities"]
                    if ent["label"] in expected_types
                )
                entity_scores.append(min(matches * 0.3, 1.0))
            else:
                entity_scores.append(0.0)

        # ── Signal 4: Position bias ──
        position_scores = [
            max(0.0, 1.0 - sent["position_in_chunk"]) * 0.1
            for sent in candidate_sentences
        ]

        # ── Combine signals ──
        # Weights: semantic=0.55, keyword=0.20, entity=0.15, position=0.10
        for i, sent in enumerate(candidate_sentences):
            total_score = (
                0.55 * float(semantic_scores[i]) +
                0.20 * keyword_scores[i] +
                0.15 * entity_scores[i] +
                0.10 * position_scores[i]
            )
            sent["semantic_score"] = float(semantic_scores[i])
            sent["keyword_score"] = keyword_scores[i]
            sent["entity_score"] = entity_scores[i]
            sent["total_score"] = total_score

        # Sort by total score
        candidate_sentences.sort(key=lambda x: x["total_score"], reverse=True)
        return candidate_sentences

    # ─────────────────────────────────────────────
    # 3. Deduplication
    # ─────────────────────────────────────────────

    def deduplicate(self, scored_sentences, threshold=DEDUP_THRESHOLD):
        """
        Remove near-duplicate sentences using embedding cosine similarity.

        Args:
            scored_sentences: List of scored sentence dicts (sorted by score).
            threshold: Cosine similarity threshold above which a sentence is considered duplicate.

        Returns:
            Filtered list with duplicates removed.
        """
        if len(scored_sentences) <= 1:
            return scored_sentences

        texts = [s["text"] for s in scored_sentences]
        embeddings = self.embedding_engine.encode(texts)

        unique = [scored_sentences[0]]
        unique_embeddings = [embeddings[0]]

        for i in range(1, len(scored_sentences)):
            is_dup = False
            for ue in unique_embeddings:
                sim = float(np.dot(embeddings[i], ue))
                if sim > threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(scored_sentences[i])
                unique_embeddings.append(embeddings[i])

        return unique

    # ─────────────────────────────────────────────
    # 4. Answer Synthesis
    # ─────────────────────────────────────────────

    def synthesize_answer(self, query_analysis, retrieved_chunks):
        """
        Full answer synthesis pipeline:
        1. Score all candidate sentences
        2. Deduplicate
        3. Select top-N
        4. Format answer with source citations
        5. Estimate confidence

        Args:
            query_analysis: Output of analyze_query().
            retrieved_chunks: List of chunk dicts from the retriever.

        Returns:
            dict with "answer", "sources", "confidence", "confidence_label"
        """
        # No documents?
        if not retrieved_chunks:
            return {
                "answer": "I don't have any documents to search through. "
                          "Please upload a PDF document first.",
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "none"
            }

        # Score sentences
        scored = self.score_sentences(query_analysis, retrieved_chunks)

        if not scored:
            return {
                "answer": "I couldn't find any relevant information in the "
                          "uploaded documents for your query.",
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "none"
            }

        # Deduplicate
        unique_scored = self.deduplicate(scored)

        # Select top-N
        top_sentences = unique_scored[:ANSWER_TOP_N]

        # Determine confidence
        best_score = top_sentences[0]["total_score"]
        if best_score >= CONFIDENCE_HIGH:
            confidence_label = "high"
        elif best_score >= CONFIDENCE_MEDIUM:
            confidence_label = "medium"
        else:
            confidence_label = "low"

        # Gather unique sources
        sources = []
        seen_sources = set()
        for sent in top_sentences:
            source_key = f"{sent['source']}:p{sent['page']}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "file": sent["source"],
                    "page": sent["page"]
                })

        # Format the answer
        if confidence_label == "low" and best_score < 0.2:
            answer_text = (
                "I couldn't find a confident answer in the uploaded documents. "
                "Here's the closest match I found:\n\n"
            )
        else:
            answer_text = ""

        # Assemble answer from top sentences
        answer_parts = []
        for sent in top_sentences:
            answer_parts.append(sent["text"])

        answer_text += " ".join(answer_parts)

        # Add source citations
        if sources:
            citation_parts = []
            for src in sources:
                citation_parts.append(f"📄 {src['file']} (page {src['page']})")
            answer_text += "\n\n" + " | ".join(citation_parts)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": round(best_score, 3),
            "confidence_label": confidence_label
        }

    def answer(self, query, retrieved_chunks):
        """
        Main entry point: analyze query → synthesize answer.

        Args:
            query: The user's question string.
            retrieved_chunks: List of chunk dicts from the retriever.

        Returns:
            dict with answer, sources, confidence.
        """
        query_analysis = self.analyze_query(query)
        return self.synthesize_answer(query_analysis, retrieved_chunks)
