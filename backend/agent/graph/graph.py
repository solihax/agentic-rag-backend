"""Builds and compiles the agentic RAG LangGraph workflow."""

from langgraph.graph import END, StateGraph

from backend.agent.graph.nodes import (
    decide_after_generate,
    decide_to_generate,
    generate_node,
    grade_documents_node,
    retrieve_node,
    web_search_node,
)
from backend.agent.graph.state import GraphState


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "generate",
        decide_after_generate,
        {"generate": "generate", "web_search": "web_search", "end": END},
    )

    return workflow.compile()
