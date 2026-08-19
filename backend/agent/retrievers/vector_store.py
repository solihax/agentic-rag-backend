"""Qdrant vector store: creates the collection and stores embedded chunks."""

import logging

from langchain_core.documents import Document as LCDocument
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from backend.agent.embeddings.embedding_provider import get_embedding_model
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 3072  # confirmed via smoke_test_llm.py output


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_mode == "local":
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection_exists(client: QdrantClient) -> None:
    settings = get_settings()
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", settings.qdrant_collection_name)


def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    client = get_qdrant_client()
    ensure_collection_exists(client)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
        embedding=get_embedding_model(),
    )


def store_chunks(chunks: list[LCDocument]) -> list[str]:
    """Embeds and stores chunks in Qdrant. Returns the list of point IDs."""
    vector_store = get_vector_store()
    ids = vector_store.add_documents(chunks)
    logger.info("Stored %s chunks in Qdrant", len(ids))
    return ids


def list_source_filenames() -> list[str]:
    """Returns the distinct filenames of all documents stored in Qdrant."""
    settings = get_settings()
    client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name not in collections:
        return []

    filenames: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", payload)
            filename = metadata.get("filename")
            if filename:
                filenames.add(filename)
        if offset is None:
            break

    return sorted(filenames)
