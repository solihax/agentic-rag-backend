"""Data models for content extracted from documents during ingestion."""

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedImage(BaseModel):
    """A single image extracted from a PDF page, before captioning."""

    image_id: str = Field(..., description="Unique id, e.g. 'page3_img1'")
    page_number: int
    image_bytes: bytes = Field(..., repr=False)
    image_format: str = Field(..., description="e.g. 'png', 'jpeg'")
    caption: str | None = Field(default=None, description="Filled in by the vision captioning step")


class PageContent(BaseModel):
    """Extracted content for a single page of a document."""

    page_number: int
    text: str
    images: list[ExtractedImage] = Field(default_factory=list)


class ExtractedDocument(BaseModel):
    """Full extraction result for one ingested file."""

    filename: str
    source_type: Literal["pdf", "txt", "md"]
    pages: list[PageContent]

    def full_text_with_captions(self) -> str:
        parts = []
        for page in self.pages:
            page_text = page.text
            for img in page.images:
                if img.caption:
                    page_text += f"\n[Image on page {page.page_number}: {img.caption}]"
            parts.append(f"--- Page {page.page_number} ---\n{page_text}")
        return "\n\n".join(parts)