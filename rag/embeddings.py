"""
SmartStock AI Analyzer — RAG Embeddings Wrapper
Uses sentence-transformers for local embeddings (no API cost).
"""

from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings

from utils.logger import log_agent

# Default model — small, fast, free
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings(model_name: str = _DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """Get or create a singleton embedding model."""
    global _embeddings_instance
    if _embeddings_instance is None:
        log_agent("RAG", f"Loading embedding model: {model_name}")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log_agent("RAG", "Embedding model loaded ✓")
    return _embeddings_instance
