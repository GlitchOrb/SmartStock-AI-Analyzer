"""
SmartStock AI Analyzer — RAG Loader
Fetches news from Google News RSS, parses with feedparser,
deduplicates by title similarity, returns LangChain Documents.
"""

from __future__ import annotations

import difflib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import log_agent


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

_BASE_RSS_URL = (
    "https://news.google.com/rss/search?"
    "q={query}+stock&hl=ko&gl=KR&ceid=KR:ko"
)

_SIMILARITY_THRESHOLD = 0.80


# ──────────────────────────────────────────────
# RSS Fetch
# ──────────────────────────────────────────────

def load_rss_news(ticker: str, max_articles: int = 30) -> tuple[list[Document], list[str]]:
    """
    Fetch news from Google News RSS for a given ticker.

    Returns:
        (documents, warnings) — list of LangChain Document objects
        and any warning messages.
    """
    warnings: list[str] = []
    query = ticker.upper()
    url = _BASE_RSS_URL.format(query=query)

    log_agent("RAGLoader", f"뉴스 수집 중: {url}")

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        log_agent("RAGLoader", f"[red]RSS 파싱 실패: {e}[/red]")
        warnings.append(f"RSS 파싱 실패: {e}")
        return [], warnings

    if not feed.entries:
        log_agent("RAGLoader", f"[yellow]{ticker}: 뉴스 기사 0건[/yellow]")
        warnings.append(f"{ticker}: 뉴스 기사를 찾을 수 없습니다.")
        return [], warnings

    raw_docs: list[dict] = []
    for entry in feed.entries[:max_articles]:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", entry.get("description", "")).strip()
        link = entry.get("link", "")
        published = entry.get("published", "")

        # Parse source from title (Google News format: "Title - Source")
        source = ""
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            source = parts[-1].strip()

        # Parse timestamp
        timestamp = ""
        if published:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(published)
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                timestamp = published

        if title:
            raw_docs.append({
                "title": title,
                "summary": summary,
                "link": link,
                "source": source,
                "timestamp": timestamp,
            })

    log_agent("RAGLoader", f"수집된 원본 기사: {len(raw_docs)}건")

    # Deduplicate
    deduped = clean_and_deduplicate(raw_docs)
    log_agent("RAGLoader", f"중복 제거 후: {len(deduped)}건")

    if not deduped:
        warnings.append(f"{ticker}: 중복 제거 후 뉴스 기사 0건")
        return [], warnings

    # Convert to LangChain Documents
    documents: list[Document] = []
    for doc in deduped:
        content = f"{doc['title']}\n\n{doc['summary']}" if doc["summary"] else doc["title"]
        documents.append(Document(
            page_content=content,
            metadata={
                "source": doc["source"],
                "url": doc["link"],
                "timestamp": doc["timestamp"],
                "title": doc["title"],
            },
        ))

    return documents, warnings


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────

def clean_and_deduplicate(docs: list[dict]) -> list[dict]:
    """
    Remove items with >80% title similarity (difflib SequenceMatcher).
    Keeps the first occurrence.
    """
    if not docs:
        return []

    kept: list[dict] = [docs[0]]

    for candidate in docs[1:]:
        is_duplicate = False
        for existing in kept:
            ratio = difflib.SequenceMatcher(
                None,
                candidate["title"].lower(),
                existing["title"].lower(),
            ).ratio()
            if ratio > _SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)

    return kept
