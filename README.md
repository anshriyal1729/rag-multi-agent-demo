# RAG + Multi-Agent Demo

A small, dependency-light implementation of a Retrieval-Augmented Generation
(RAG) pipeline coordinated by a simple multi-agent orchestrator — built as a
portfolio project to showcase the core mechanics behind production GenAI
systems (RAG ingestion/retrieval, agent hand-offs, confidence gating).

This is a personal/demo project, not production code from any employer —
it's intentionally framework-free so the full pipeline is readable end to
end in ~150 lines, but the same tool-calling / hand-off pattern maps
directly onto LangChain/LangGraph nodes if you want to port it.

## What it does

1. **Ingestion** — takes raw documents (`data/sample_docs.py`) and splits
   them into overlapping chunks (`chunk_text`).
2. **Embedding & Retrieval** — chunks are vectorized with TF-IDF and ranked
   by cosine similarity against the query (`RAGPipeline.retrieve`).
3. **Multi-agent orchestration** — three agents cooperate on each query:
   - `RetrieverAgent` — fetches relevant chunks
   - `ReasoningAgent` — decides if there's enough context to answer confidently
   - `ResponderAgent` — drafts the final answer, or a graceful "I don't know"
4. **Generation** — a lightweight extractive generator picks the most
   relevant sentence from the retrieved chunks (swap in a real LLM call for
   abstractive answers — see below).

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
```

Example output:
```
QUERY: What is RAG and how does retrieval work?
ANSWER: Retrieval-Augmented Generation, or RAG, combines a retriever and a generator.
TRACE:
  - [RetrieverAgent] retrieve: Retrieved 2 chunk(s) for query: '...'
  - [ReasoningAgent] assess_confidence: confident=True based on 2 retrieved chunk(s)
  - [ResponderAgent] respond: Generated final answer (confident=True)
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## Project layout

```
rag-multi-agent-demo/
├── src/
│   ├── rag_pipeline.py   # chunking, TF-IDF retrieval, extractive answer
│   └── agents.py         # RetrieverAgent, ReasoningAgent, ResponderAgent
├── data/
│   └── sample_docs.py    # tiny built-in knowledge base
├── tests/
│   └── test_rag_pipeline.py
└── demo.py               # run this
```

## Limitations & upgrade paths

- **Lexical, not semantic, retrieval.** TF-IDF matches on shared words, so a
  query sharing a common word with the corpus (even a word like "query"
  itself) can register a false match. Swap `TfidfVectorizer` for a real
  sentence embedding model (e.g. `sentence-transformers`, or an
  OpenAI/Anthropic/Vertex embedding endpoint) plus a vector store (FAISS,
  Pinecone, Chroma) for semantic retrieval and hybrid search.
- **Extractive, not abstractive, generation.** The demo generator just picks
  the most relevant sentence rather than writing a new one. Replace
  `_extractive_answer` with a call to an LLM (passing the retrieved chunks
  as context) for real generation.
- **No persistence.** The knowledge base is rebuilt in memory every run.
  For production, persist chunks + embeddings to a vector DB.

## License

MIT — see [LICENSE](LICENSE).
