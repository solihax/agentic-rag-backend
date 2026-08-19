"""Retriever: wraps the vector store for top-k similarity search."""

from langchain_core.documents import Document as LCDocument

from backend.agent.retrievers.vector_store import get_vector_store
from backend.config.settings import get_settings


def retrieve(query: str, k: int | None = None) -> list[LCDocument]:
    """
    Retrieves the top-k most similar chunks for a query.

    Args:
        query: The user's question or search text.
        k: Number of results. Defaults to Settings.retrieval_top_k (4).

    Returns:
        List of LangChain Documents, each with metadata containing
        filename, source, and chunk_id for citation purposes.
    """
    settings = get_settings()
    top_k = k if k is not None else settings.retrieval_top_k

    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=top_k)
