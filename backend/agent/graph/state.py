"""Shared state passed between all LangGraph nodes."""

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    generation: str
    documents: list[Document]
    generation_retries: int
    web_search_retries: int
