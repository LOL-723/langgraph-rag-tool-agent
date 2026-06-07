from typing import Any

from core.config import settings
from llm.rag_service import rag_service


def retrieve_document_context(
    query: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    retrieved, retrieval_mode, rag_query_str = rag_service.retrieve_with_mode(
        query,
        settings.RAG_RETRIEVE_TOP_K,
        document_id=document_id,
    )
    reranked = rag_service.rerank(
        query,
        retrieved,
        settings.RAG_RERANK_TOP_K,
    )

    return {
        "retrieved_docs": [source.model_dump() for source in reranked],
        "rag_retrieval_mode": retrieval_mode,
        "rag_query_str": rag_query_str,
        "document_id": document_id,
    }


def retrieve_uploaded_document_tool(
    query: str | None = None,
    document_id: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    if not query or not query.strip():
        return {
            "error": "query cannot be empty",
            "retrieved_docs": [],
            "rag_retrieval_mode": None,
            "rag_query_str": {},
            "document_id": document_id,
        }

    return retrieve_document_context(
        query=query.strip(),
        document_id=document_id,
    )
