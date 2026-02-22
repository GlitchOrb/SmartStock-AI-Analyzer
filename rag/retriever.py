"""
SmartStock AI Analyzer — LangChain Retriever backed by FAISS
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from rag.vectorstore import load_vectorstore
from utils.logger import log_agent


class SmartStockRetriever(BaseRetriever):
    """Retriever that queries the FAISS index for relevant past analyses."""

    index_name: str = Field(default="smartstock")
    top_k: int = Field(default=4)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        vectorstore = load_vectorstore(self.index_name)
        if vectorstore is None:
            log_agent("RAG", "No index found — returning empty results")
            return []

        results = vectorstore.similarity_search(query, k=self.top_k)
        log_agent("RAG", f"Retrieved {len(results)} relevant documents")
        return results
