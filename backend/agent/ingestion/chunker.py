"""
Chunking: splits extracted document text into overlapping chunks ready
for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from backend.config.settings import get_settings
from backend.models.document import ExtractedDocument


def chunk_document(document: ExtractedDocument) -> list[LCDocument]:
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    full_text = document.full_text_with_captions()
    raw_chunks = splitter.split_text(full_text)

    lc_documents: list[LCDocument] = []
    for i, chunk_text in enumerate(raw_chunks):
        lc_documents.append(
            LCDocument(
                page_content=chunk_text,
                metadata={
                    "filename": document.filename,
                    "source": document.filename,
                    "source_type": document.source_type,
                    "chunk_id": f"{document.filename}_chunk{i}",
                },
            )
        )

    return lc_documents