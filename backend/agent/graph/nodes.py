"""LangGraph nodes and conditional routing functions."""

import logging

from backend.agent.chains.generation_chain import generate_answer
from backend.agent.chains.graders import (
    grade_answer_usefulness,
    grade_document,
    grade_hallucination,
)
from backend.agent.graph.state import GraphState
from backend.agent.retrievers.retriever import retrieve
from backend.agent.retrievers.web_search_tool import web_search
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def retrieve_node(state: GraphState) -> GraphState:
    documents = retrieve(state["question"])
    return {**state, "documents": documents}


def grade_documents_node(state: GraphState) -> GraphState:
    question = state["question"]
    filtered = [
        doc for doc in state["documents"] if grade_document(question, doc.page_content)
    ]
    return {**state, "documents": filtered}


def web_search_node(state: GraphState) -> GraphState:
    results = web_search(state["question"])
    documents = state.get("documents", []) + results
    retries = state.get("web_search_retries", 0) + 1
    return {**state, "documents": documents, "web_search_retries": retries}


def generate_node(state: GraphState) -> GraphState:
    generation = generate_answer(state["question"], state["documents"])
    retries = state.get("generation_retries", 0) + 1
    return {**state, "generation": generation, "generation_retries": retries}


def decide_to_generate(state: GraphState) -> str:
    settings = get_settings()
    if state["documents"]:
        return "generate"
    if state.get("web_search_retries", 0) >= settings.max_web_search_retries:
        return "generate"
    return "web_search"


def decide_after_generate(state: GraphState) -> str:
    settings = get_settings()
    documents_text = "\n\n".join(doc.page_content for doc in state["documents"])
    generation = state["generation"]

    grounded = grade_hallucination(documents_text, generation) if documents_text else True
    if not grounded:
        if state.get("generation_retries", 0) >= settings.max_generation_retries:
            return "end"
        return "generate"

    useful = grade_answer_usefulness(state["question"], generation)
    if not useful:
        if state.get("web_search_retries", 0) >= settings.max_web_search_retries:
            return "end"
        return "web_search"

    return "end"
