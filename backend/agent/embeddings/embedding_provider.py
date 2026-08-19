"""
Embedding provider factory - routes all embedding calls through the
mentor-issued OpenAI-compatible proxy.

Why this file exists:
    Both document ingestion (embedding chunks before storing in Qdrant)
    and query-time retrieval (embedding the user's question) need the
    EXACT same embedding model and dimension - otherwise vector similarity
    search silently returns garbage. Centralizing this in one factory
    guarantees ingestion and retrieval never drift apart.
"""

from langchain_openai import OpenAIEmbeddings

from backend.config.settings import get_settings


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Returns a configured OpenAIEmbeddings instance pointed at the Gemini
    proxy's embedding model.

    Raises:
        ValueError: if GEMINI_API_KEY is missing, so the error surfaces
                    immediately at call time with a clear message.
    """
    settings = get_settings()

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before generating embeddings."
        )

    return OpenAIEmbeddings(
        base_url=settings.gemini_proxy_openai_base_url,
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
    )
