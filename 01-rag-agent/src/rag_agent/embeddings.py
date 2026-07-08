"""Embedding model factory.

We use NVIDIA's retrieval-tuned embeddings. A key, non-obvious detail lives here: these
models are **asymmetric** — a document ("passage") and a search query are embedded
differently. LangChain's ``NVIDIAEmbeddings`` handles that automatically by calling the
model with ``input_type="passage"`` inside ``embed_documents`` and ``input_type="query"``
inside ``embed_query``. Getting that split right is one of the biggest levers on retrieval
accuracy, so we deliberately rely on those two methods rather than embedding by hand.
"""

from __future__ import annotations

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from rag_agent.config import Settings


def build_embeddings(settings: Settings) -> NVIDIAEmbeddings:
    """Instantiate the NVIDIA embedding model from settings."""
    return NVIDIAEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        truncate=settings.embedding_truncate,
    )
