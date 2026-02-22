"""
SmartStock AI Analyzer — RAG Vector Store
FAISS index backed by HuggingFaceEmbeddings (all-MiniLM-L6-v2).
Caches index to disk with TTL-based freshness check.
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.config import settings
from utils.logger import log_agent


# ──────────────────────────────────────────────
# Singleton embeddings (free, local)
# ──────────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create singleton HuggingFaceEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        log_agent("VectorStore", "임베딩 모델 로딩 중 (all-MiniLM-L6-v2)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        log_agent("VectorStore", "임베딩 모델 로딩 완료")
    return _embeddings


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    """Return the cache directory path for a ticker's FAISS index."""
    today = datetime.now().strftime("%Y%m%d")
    name = f"{ticker.upper()}_{today}"
    return settings.vectordb_dir / name


def _is_cache_fresh(cache_dir: Path) -> bool:
    """Check if cached index is within CACHE_TTL."""
    index_file = cache_dir / "index.faiss"
    if not index_file.exists():
        return False
    age_seconds = time.time() - index_file.stat().st_mtime
    ttl_seconds = settings.cache_ttl
    return age_seconds < ttl_seconds


# ──────────────────────────────────────────────
# Build / Load
# ──────────────────────────────────────────────

def build_vectorstore(
    ticker: str,
    documents: list[Document],
    force_rebuild: bool = False,
) -> FAISS | None:
    """
    Build or load a FAISS vectorstore for a ticker.

    1. If cache is fresh and force_rebuild=False → load from disk
    2. Otherwise → build from documents and save to disk
    3. If documents is empty → return None

    Returns FAISS instance or None if no documents.
    """
    cache_dir = _cache_path(ticker)

    # Try loading from cache
    if not force_rebuild and _is_cache_fresh(cache_dir):
        log_agent("VectorStore", f"캐시된 인덱스 로딩: {cache_dir}")
        try:
            embeddings = get_embeddings()
            vs = FAISS.load_local(
                str(cache_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            log_agent("VectorStore", "캐시 인덱스 로딩 완료")
            return vs
        except Exception as e:
            log_agent("VectorStore", f"[yellow]캐시 로딩 실패, 재구축: {e}[/yellow]")

    # Build from documents
    if not documents:
        log_agent("VectorStore", "[yellow]문서 없음 — 벡터스토어 생성 건너뜀[/yellow]")
        return None

    log_agent("VectorStore", f"FAISS 인덱스 구축 중 ({len(documents)} 문서)...")
    embeddings = get_embeddings()

    try:
        vs = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        log_agent("VectorStore", f"[red]인덱스 구축 실패: {e}[/red]")
        return None

    # Save to disk
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(cache_dir))
        log_agent("VectorStore", f"인덱스 저장 완료: {cache_dir}")
    except Exception as e:
        log_agent("VectorStore", f"[yellow]인덱스 저장 실패: {e}[/yellow]")

    return vs


# ──────────────────────────────────────────────
# Retrieve
# ──────────────────────────────────────────────

def retrieve(
    vs: FAISS | None,
    query: str,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k similar chunks from the vectorstore.

    Returns list of dicts: {content, source, url, timestamp}
    Returns empty list if vs is None or retrieval fails.
    """
    if vs is None:
        log_agent("VectorStore", "[yellow]벡터스토어 없음 — 검색 건너뜀[/yellow]")
        return []

    try:
        docs = vs.similarity_search(query, k=k)
    except Exception as e:
        log_agent("VectorStore", f"[red]검색 실패: {e}[/red]")
        return []

    results: list[dict] = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "url": doc.metadata.get("url", ""),
            "timestamp": doc.metadata.get("timestamp", ""),
        })

    log_agent("VectorStore", f"검색 결과: {len(results)}건")
    return results
