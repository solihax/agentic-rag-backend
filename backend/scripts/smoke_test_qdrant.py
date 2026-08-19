"""
Usage:
    python -m backend.scripts.smoke_test_qdrant "sample_test.pdf"
"""

import sys

from backend.agent.ingestion.pdf_extractor import extract_pdf
from backend.agent.ingestion.chunker import chunk_document
from backend.agent.retrievers.vector_store import store_chunks, get_vector_store


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.smoke_test_qdrant <path_to_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]
    doc = extract_pdf(file_path)
    chunks = chunk_document(doc)

    print(f"Storing {len(chunks)} chunk(s) in Qdrant...")
    ids = store_chunks(chunks)
    print(f"Stored with IDs: {ids}")

    print("\nTesting retrieval with a sample query...")
    vector_store = get_vector_store()
    results = vector_store.similarity_search("sample test document", k=2)
    for r in results:
        print(f"\nMatch: {r.page_content[:150]!r}")
        print(f"Metadata: {r.metadata}")


if __name__ == "__main__":
    main()