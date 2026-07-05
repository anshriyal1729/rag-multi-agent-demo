import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.sample_docs import SAMPLE_DOCS
from src.agents import MultiAgentOrchestrator
from src.rag_pipeline import RAGPipeline, chunk_text


def build_pipeline():
    pipeline = RAGPipeline(chunk_size=60, overlap=10)
    pipeline.ingest(SAMPLE_DOCS)
    return pipeline


def test_chunking_splits_long_text():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text("doc1", text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(c.doc_id == "doc1" for c in chunks)


def test_retrieve_returns_relevant_chunks():
    pipeline = build_pipeline()
    results = pipeline.retrieve("vector database embeddings", top_k=2)
    assert len(results) > 0
    assert any("vector" in c.text.lower() for c in results)


def test_answer_includes_sources():
    pipeline = build_pipeline()
    result = pipeline.answer("What is RAG?")
    assert result["answer"]
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0


def test_out_of_scope_query_returns_no_sources():
    pipeline = build_pipeline()
    result = pipeline.answer("What is the airspeed velocity of an unladen swallow?")
    assert result["sources"] == [] or len(result["sources"]) == 0


def test_orchestrator_low_confidence_path():
    # Uses vocabulary with zero overlap with the sample knowledge base.
    # (Pure TF-IDF retrieval is lexical, not semantic, so any shared word
    # -- even "query" itself -- would otherwise register a false match.
    # This is a known limitation; see README "Limitations & Upgrades".)
    pipeline = build_pipeline()
    orchestrator = MultiAgentOrchestrator(pipeline, top_k=2)
    result = orchestrator.handle_query("banana smoothie recipe ingredients")
    assert "don't have enough information" in result["answer"]
    assert len(result["trace"]) == 3


def test_orchestrator_confident_path():
    pipeline = build_pipeline()
    orchestrator = MultiAgentOrchestrator(pipeline, top_k=2)
    result = orchestrator.handle_query("How does a multi-agent system work?")
    assert "don't have enough information" not in result["answer"]
    assert len(result["trace"]) == 3
