"""API tests using FastAPI's TestClient. /chat is mocked to avoid live LLM calls."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sources_endpoint_returns_list():
    response = client.get("/sources")
    assert response.status_code == 200
    assert "sources" in response.json()
    assert isinstance(response.json()["sources"], list)


def test_chat_endpoint_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 400


@patch("backend.api.main._graph")
def test_chat_endpoint_returns_answer_and_sources(mock_graph):
    mock_graph.invoke.return_value = {
        "generation": "This is a test answer.",
        "documents": [],
    }
    response = client.post("/chat", json={"question": "What is this about?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test answer."
    assert data["sources"] == []


def test_ingest_endpoint_rejects_non_pdf():
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
