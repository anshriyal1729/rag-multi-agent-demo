"""A tiny sample knowledge base so the demo runs with zero setup."""

SAMPLE_DOCS = {
    "rag_overview": (
        "Retrieval-Augmented Generation, or RAG, combines a retriever and a generator. "
        "The retriever searches a knowledge base for chunks of text relevant to a user's "
        "query, typically using vector embeddings and cosine similarity. The generator, "
        "usually a large language model, then uses the retrieved chunks as context to "
        "produce a grounded, up-to-date answer instead of relying only on parameters "
        "learned during training."
    ),
    "vector_databases": (
        "Vector databases such as FAISS, Pinecone, Chroma, and OpenSearch store dense "
        "embeddings of text chunks and support approximate nearest neighbor search. This "
        "lets a RAG system retrieve semantically similar chunks even when the query does "
        "not share exact keywords with the source document. Hybrid search combines this "
        "vector similarity with traditional keyword-based (BM25) search for better recall."
    ),
    "multi_agent_systems": (
        "A multi-agent system splits a task across several specialized agents that each "
        "handle one responsibility, such as retrieval, reasoning, or response generation. "
        "Agents communicate through a shared state or message passing, and an orchestrator "
        "coordinates hand-offs between them. This modular design makes it easier to debug "
        "failures, since each agent's contribution can be inspected independently."
    ),
    "self_healing_ops": (
        "In AIOps, a self-healing pipeline detects anomalies in system alerts, runs an "
        "automated root-cause analysis, and recommends or triggers remediation actions "
        "such as an Ansible playbook or a configuration rollback. This reduces manual "
        "intervention and mean time to resolution for common operational incidents."
    ),
}
