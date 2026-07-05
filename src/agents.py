"""
Minimal multi-agent orchestration on top of rag_pipeline.RAGPipeline.

Three specialized agents cooperate on each query, mirroring a common
production pattern (retrieval agent -> reasoning agent -> response agent):

    RetrieverAgent  - fetches relevant context from the knowledge base
    ReasoningAgent  - decides whether the retrieved context is sufficient
    ResponderAgent  - drafts the final response, flagging low-confidence answers

This is intentionally framework-free (no LangChain/LangGraph) so it's easy
to read end-to-end, but the same tool-calling / hand-off pattern maps
directly onto LangGraph nodes or a LangChain AgentExecutor if you want to
port it later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .rag_pipeline import Chunk, RAGPipeline


@dataclass
class AgentTrace:
    agent: str
    action: str
    detail: str


class RetrieverAgent:
    def __init__(self, pipeline: RAGPipeline, top_k: int = 3):
        self.pipeline = pipeline
        self.top_k = top_k

    def run(self, query: str) -> tuple[List[Chunk], AgentTrace]:
        chunks = self.pipeline.retrieve(query, top_k=self.top_k)
        trace = AgentTrace(
            agent="RetrieverAgent",
            action="retrieve",
            detail=f"Retrieved {len(chunks)} chunk(s) for query: {query!r}",
        )
        return chunks, trace


class ReasoningAgent:
    """Decides if retrieved context is strong enough to answer confidently."""

    def __init__(self, min_chunks: int = 1):
        self.min_chunks = min_chunks

    def run(self, query: str, chunks: List[Chunk]) -> tuple[bool, AgentTrace]:
        confident = len(chunks) >= self.min_chunks
        trace = AgentTrace(
            agent="ReasoningAgent",
            action="assess_confidence",
            detail=f"confident={confident} based on {len(chunks)} retrieved chunk(s)",
        )
        return confident, trace


class ResponderAgent:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(self, query: str, chunks: List[Chunk], confident: bool) -> tuple[dict, AgentTrace]:
        if not confident:
            result = {
                "query": query,
                "answer": "I don't have enough information in the knowledge base to answer confidently.",
                "sources": [],
            }
        else:
            result = self.pipeline.answer(query)
        trace = AgentTrace(
            agent="ResponderAgent",
            action="respond",
            detail=f"Generated final answer (confident={confident})",
        )
        return result, trace


class MultiAgentOrchestrator:
    """Coordinates Retriever -> Reasoning -> Responder for a single query."""

    def __init__(self, pipeline: RAGPipeline, top_k: int = 3):
        self.pipeline = pipeline
        self.retriever = RetrieverAgent(pipeline, top_k=top_k)
        self.reasoner = ReasoningAgent()
        self.responder = ResponderAgent(pipeline)

    def handle_query(self, query: str) -> dict:
        traces: List[AgentTrace] = []

        chunks, t1 = self.retriever.run(query)
        traces.append(t1)

        confident, t2 = self.reasoner.run(query, chunks)
        traces.append(t2)

        result, t3 = self.responder.run(query, chunks, confident)
        traces.append(t3)

        result["trace"] = [f"[{t.agent}] {t.action}: {t.detail}" for t in traces]
        return result
