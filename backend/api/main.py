"""FastAPI backend: /chat, /ingest, /health, /sources."""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.agent.graph.graph import build_graph
from backend.agent.ingestion.chunker import chunk_document
from backend.agent.ingestion.image_captioner import caption_all_images
from backend.agent.ingestion.pdf_extractor import extract_pdf
from backend.agent.retrievers.vector_store import list_source_filenames, store_chunks
from backend.models.api import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    SourceInfo,
    SourcesResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Agentic RAG Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Built once at startup, reused across requests (LangGraph graphs are
# stateless/thread-safe to invoke repeatedly - rebuilding per request
# would waste time recompiling the graph structure every call).
_graph = build_graph()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = _graph.invoke(
        {
            "question": request.question,
            "generation": "",
            "documents": [],
            "generation_retries": 0,
            "web_search_retries": 0,
        }
    )

    sources = [
        SourceInfo(
            filename=doc.metadata.get("filename", "unknown"),
            source_type=doc.metadata.get("source_type", "unknown"),
        )
        for doc in result["documents"]
    ]

    return ChatResponse(answer=result["generation"], sources=sources)


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported currently")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        doc = extract_pdf(tmp_path)
        doc.filename = file.filename  # keep original name, not the temp path

        images_count = sum(len(page.images) for page in doc.pages)
        doc = caption_all_images(doc)

        chunks = chunk_document(doc)
        ids = store_chunks(chunks)

        return IngestResponse(
            filename=file.filename,
            chunks_stored=len(ids),
            images_captioned=images_count,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/sources", response_model=SourcesResponse)
def sources() -> SourcesResponse:
    return SourcesResponse(sources=list_source_filenames())
