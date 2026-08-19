"""PDF ingestion: extracts text and images per page using PyMuPDF (fitz)."""

import logging

import fitz  # PyMuPDF

from backend.models.document import ExtractedDocument, ExtractedImage, PageContent

logger = logging.getLogger(__name__)


def extract_pdf(file_path: str) -> ExtractedDocument:
    doc = fitz.open(file_path)
    pages: list[PageContent] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        text = page.get_text()

        images: list[ExtractedImage] = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                logger.warning(
                    "Failed to extract image xref=%s on page %s: %s",
                    xref, page_number, e,
                )
                continue

            images.append(
                ExtractedImage(
                    image_id=f"page{page_number}_img{img_index + 1}",
                    page_number=page_number,
                    image_bytes=base_image["image"],
                    image_format=base_image["ext"],
                )
            )

        pages.append(PageContent(page_number=page_number, text=text, images=images))

    doc.close()

    return ExtractedDocument(
        filename=file_path.split("/")[-1].split("\\")[-1],
        source_type="pdf",
        pages=pages,
    )