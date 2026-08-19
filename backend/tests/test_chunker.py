"""Unit tests for the chunking module."""

from backend.agent.ingestion.chunker import chunk_document
from backend.models.document import ExtractedDocument, PageContent


def test_chunk_document_produces_chunks_with_metadata():
    doc = ExtractedDocument(
        filename="test.pdf",
        source_type="pdf",
        pages=[PageContent(page_number=1, text="Hello world. " * 20, images=[])],
    )

    chunks = chunk_document(doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata["filename"] == "test.pdf"
        assert chunk.metadata["source_type"] == "pdf"
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content) > 0


def test_chunk_document_empty_page_still_includes_page_header():
    # full_text_with_captions() always adds a "--- Page N ---" header,
    # so even an empty page produces one small chunk, not zero.
    doc = ExtractedDocument(
        filename="empty.pdf",
        source_type="pdf",
        pages=[PageContent(page_number=1, text="", images=[])],
    )

    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert "Page 1" in chunks[0].page_content
