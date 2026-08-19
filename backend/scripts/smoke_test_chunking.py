"""
Usage:
    python -m backend.scripts.smoke_test_chunking "sample_test.pdf"
"""

import sys

from backend.agent.ingestion.pdf_extractor import extract_pdf
from backend.agent.ingestion.chunker import chunk_document


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.smoke_test_chunking <path_to_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]
    doc = extract_pdf(file_path)
    chunks = chunk_document(doc)

    print(f"Document: {doc.filename}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\n--- {chunk.metadata['chunk_id']} ---")
        print(f"Length: {len(chunk.page_content)} chars")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content preview: {chunk.page_content[:200]!r}")


if __name__ == "__main__":
    main()