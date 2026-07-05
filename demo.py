"""Run this file to see the RAG + multi-agent pipeline answer a few sample queries.

Usage:
    python demo.py
"""
from data.sample_docs import SAMPLE_DOCS
from src.agents import MultiAgentOrchestrator
from src.rag_pipeline import RAGPipeline

QUERIES = [
    "What is RAG and how does retrieval work?",
    "How do vector databases help with retrieval?",
    "How does a multi-agent system coordinate different tasks?",
    "What does a self-healing AIOps pipeline do?",
    "What is the capital of France?",  # out-of-scope query to show low-confidence handling
]


def main() -> None:
    pipeline = RAGPipeline(chunk_size=60, overlap=10)
    pipeline.ingest(SAMPLE_DOCS)

    orchestrator = MultiAgentOrchestrator(pipeline, top_k=2)

    for query in QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        result = orchestrator.handle_query(query)
        print(f"ANSWER: {result['answer']}")
        print("TRACE:")
        for line in result["trace"]:
            print(f"  - {line}")
        if result["sources"]:
            print("SOURCES:")
            for src in result["sources"]:
                print(f"  - [{src['doc_id']}] {src['text'][:100]}...")
    print("=" * 80)


if __name__ == "__main__":
    main()
