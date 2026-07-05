"""
Lightweight, dependency-minimal RAG (Retrieval-Augmented Generation) pipeline.

Implements the standard RAG stages end-to-end:
    ingestion -> chunking -> embedding (TF-IDF) -> retrieval -> generation

Uses scikit-learn's TF-IDF vectorizer + cosine similarity as the retriever
so the whole pipeline runs offline with no external API keys or GPU needed.
Swap `embed()` for a sentence-transformers or OpenAI/Anthropic embedding
call to upgrade retrieval quality in production.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_text(doc_id: str, text: str, chunk_size: int = 400, overlap: int = 50) -> List[Chunk]:
    """Split a document into overlapping word-based chunks."""
    words = text.split()
    chunks: List[Chunk] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(Chunk(doc_id=doc_id, text=" ".join(chunk_words)))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


class RAGPipeline:
    """Ingests documents, retrieves relevant chunks, and generates an answer."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None

    def ingest(self, documents: dict) -> None:
        """documents: {doc_id: raw_text}"""
        for doc_id, text in documents.items():
            self.chunks.extend(chunk_text(doc_id, text, self.chunk_size, self.overlap))
        self._reindex()

    def _reindex(self) -> None:
        if not self.chunks:
            return
        corpus = [c.text for c in self.chunks]
        self._matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Chunk]:
        if self._matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]

    @staticmethod
    def _extractive_answer(query: str, chunks: List[Chunk]) -> str:
        """Very small extractive generator: pulls the sentence most relevant
        to the query out of the retrieved chunks. Replace with a real LLM
        call (OpenAI/Anthropic/local model) for abstractive generation."""
        if not chunks:
            return "I couldn't find anything relevant in the knowledge base."

        query_terms = set(re.findall(r"\w+", query.lower()))
        best_sentence, best_score = "", -1
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                terms = set(re.findall(r"\w+", sentence.lower()))
                overlap_score = len(query_terms & terms)
                if overlap_score > best_score:
                    best_score, best_sentence = overlap_score, sentence
        return best_sentence.strip() or chunks[0].text[:200]

    def answer(self, query: str, top_k: int = 3) -> dict:
        retrieved = self.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "answer": self._extractive_answer(query, retrieved),
            "sources": [{"doc_id": c.doc_id, "text": c.text[:200]} for c in retrieved],
        }
