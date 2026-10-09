"""Retriever module for rag_knowledge."""

import logging
from typing import Any, Dict, List, Optional

from ..storage.qdrant_storage import (
    QdrantKnowledgeStore,
    get_global_qdrant_store,
)

logger = logging.getLogger("rag.retriever")


class KnowledgeRetriever:
    """Vector-backed knowledge retriever for RAG queries using Qdrant."""

    def __init__(self, store: Optional[QdrantKnowledgeStore] = None) -> None:
        self.store: QdrantKnowledgeStore = store or get_global_qdrant_store()

    @property
    def documents(self) -> List[Dict[str, Any]]:
        """Lists active knowledge documents from Qdrant."""
        return self.store.list_documents()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant documents for the given query using Qdrant vector search.

        Args:
            query: The user query string (e.g. 'What is the schedule for ComputeX?').
            top_k: Maximum number of relevant documents to return (defaults to RAG_TOP_K or 5).
            min_score: Minimum relevance score required to be considered a match.

        Returns:
            List of matching document dicts with 'id', 'title', 'summary', 'content', and 'score'.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        import os
        effective_top_k = top_k if top_k is not None else int(os.getenv("RAG_TOP_K", "5"))
        effective_min_score = min_score if min_score is not None else float(os.getenv("RAG_MIN_SCORE", "0.40"))

        try:
            return self.store.search_text(clean_query, top_k=effective_top_k, min_score=effective_min_score)
        except Exception as e:
            logger.error(f"Error retrieving knowledge from Qdrant for '{clean_query}': {e}", exc_info=True)
            return []

    def build_context_string(self, query: str, top_k: Optional[int] = None) -> str:
        """Convenience method to retrieve and format context into a clean text block."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        context_parts = []
        for r in results:
            context_parts.append(f"[{r.get('title')}]\n{r.get('content')}")
        return "\n\n".join(context_parts)
