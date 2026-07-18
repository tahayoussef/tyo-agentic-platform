"""Embedding model factory.

We use NVIDIA's retrieval-tuned embeddings. A key, non-obvious detail lives here: these
models are **asymmetric** — a document ("passage") and a search query are embedded
differently. LangChain's ``NVIDIAEmbeddings`` handles that automatically by calling the
model with ``input_type="passage"`` inside ``embed_documents`` and ``input_type="query"``
inside ``embed_query``. Getting that split right is one of the biggest levers on retrieval
accuracy, so we deliberately rely on those two methods rather than embedding by hand.
"""

from __future__ import annotations

from typing import Any

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_qdrant import FastEmbedSparse

from rag_agent.config import Settings


def build_embeddings(settings: Settings) -> NVIDIAEmbeddings:
    """Instantiate the NVIDIA (dense) embedding model from settings."""
    return NVIDIAEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        truncate=settings.embedding_truncate,
    )


def build_sparse_embeddings(settings: Settings) -> Any:
    """Build the sparse (keyword/BM25) embedding used for the hybrid-search's lexical half.

    Runs locally via FastEmbed; the model is downloaded on first use. Returns a
    ``FastEmbedSparse`` — typed ``Any`` because langchain-qdrant ships no stubs.
    """
    return FastEmbedSparse(model_name=settings.sparse_embedding_model)


def build_reranker(settings: Settings) -> NVIDIARerank:
    """Build the cross-encoder reranker for two-stage retrieval (fetch many, re-score, cut)."""
    return NVIDIARerank(
        model=settings.reranker_model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        top_n=settings.top_k,
    )
