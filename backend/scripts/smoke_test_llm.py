"""
Manual smoke test for the Gemini proxy connection.

Run this directly (not with pytest) to confirm your .env key and the
mentor's proxy are working BEFORE we build the ingestion/retrieval/graph
code on top of it. This is a fast, cheap sanity check, not a formal test.

Usage (from the project root, with your venv active):
    python -m backend.scripts.smoke_test_llm
"""

from backend.agent.llm_provider import get_chat_llm
from backend.agent.embeddings.embedding_provider import get_embedding_model


def main() -> None:
    print("Testing chat model (flash_lite)...")
    llm = get_chat_llm(tier="flash_lite")
    response = llm.invoke("Reply with exactly one word: OK")
    print(f"Chat response: {response.content!r}")

    print("\nTesting embedding model...")
    embeddings = get_embedding_model()
    vector = embeddings.embed_query("hello world")
    print(f"Embedding vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

    print("\nAll checks passed. Proxy connection is working.")


if __name__ == "__main__":
    main()
