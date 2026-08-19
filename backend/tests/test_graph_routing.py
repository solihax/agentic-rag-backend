"""
Unit tests for graph routing logic. These test the decision functions
directly with fake state, without calling the real LLM - keeps tests
fast and free, and isolates routing logic from grading logic.
"""

from unittest.mock import patch

from langchain_core.documents import Document

from backend.agent.graph.nodes import decide_after_generate, decide_to_generate


def test_decide_to_generate_with_documents_goes_to_generate():
    state = {"documents": [Document(page_content="relevant text")]}
    assert decide_to_generate(state) == "generate"


def test_decide_to_generate_no_documents_triggers_web_search():
    state = {"documents": [], "web_search_retries": 0}
    assert decide_to_generate(state) == "web_search"


def test_decide_to_generate_no_documents_at_retry_limit_gives_up():
    # max_web_search_retries default is 1
    state = {"documents": [], "web_search_retries": 1}
    assert decide_to_generate(state) == "generate"


@patch("backend.agent.graph.nodes.grade_hallucination", return_value=True)
@patch("backend.agent.graph.nodes.grade_answer_usefulness", return_value=True)
def test_decide_after_generate_grounded_and_useful_ends(mock_useful, mock_grounded):
    state = {
        "documents": [Document(page_content="fact")],
        "generation": "a grounded, useful answer",
        "question": "test question",
        "generation_retries": 0,
        "web_search_retries": 0,
    }
    assert decide_after_generate(state) == "end"


@patch("backend.agent.graph.nodes.grade_hallucination", return_value=False)
def test_decide_after_generate_hallucinated_retries_generation(mock_grounded):
    state = {
        "documents": [Document(page_content="fact")],
        "generation": "a hallucinated answer",
        "question": "test question",
        "generation_retries": 0,
        "web_search_retries": 0,
    }
    assert decide_after_generate(state) == "generate"


@patch("backend.agent.graph.nodes.grade_hallucination", return_value=False)
def test_decide_after_generate_hallucinated_at_retry_limit_ends(mock_grounded):
    state = {
        "documents": [Document(page_content="fact")],
        "generation": "a hallucinated answer",
        "question": "test question",
        "generation_retries": 2,  # max_generation_retries default is 2
        "web_search_retries": 0,
    }
    assert decide_after_generate(state) == "end"
