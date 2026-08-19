"""Web search tool (Tavily). Swap providers by editing only this file."""

import logging

from langchain_core.documents import Document
from tavily import TavilyClient

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> list[Document]:
    settings = get_settings()

    if settings.web_search_provider != "tavily" or not settings.tavily_api_key:
        logger.warning("Web search not configured; returning no results.")
        return []

    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        response = client.search(query=query, max_results=max_results)
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return []

    docs = []
    for result in response.get("results", []):
        docs.append(
            Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("url", "web_search"),
                    "filename": result.get("title", "Web Result"),
                    "source_type": "web",
                },
            )
        )
    return docs
