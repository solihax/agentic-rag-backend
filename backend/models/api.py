"""Pydantic models for FastAPI request/response bodies."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    filename: str
    source_type: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    images_captioned: int


class HealthResponse(BaseModel):
    status: str


class SourcesResponse(BaseModel):
    sources: list[str]
