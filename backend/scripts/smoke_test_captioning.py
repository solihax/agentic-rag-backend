"""
Usage:
    python -m backend.scripts.smoke_test_captioning "sample_test.pdf"
"""

import sys

from backend.agent.ingestion.pdf_extractor import extract_pdf
from backend.agent.ingestion.image_captioner import caption_all_images


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.smoke_test_captioning <path_to_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]
    doc = extract_pdf(file_path)

    total_images = sum(len(page.images) for page in doc.pages)
    print(f"Found {total_images} image(s) across {len(doc.pages)} page(s)")

    if total_images == 0:
        print("No images to caption. Try a PDF with embedded images.")
        return

    print("Captioning images (this calls the vision model)...")
    doc = caption_all_images(doc)

    for page in doc.pages:
        for image in page.images:
            print(f"\n{image.image_id} (page {page.page_number}):")
            print(f"  Caption: {image.caption}")

    print("\n--- Combined text preview (what gets chunked later) ---")
    print(doc.full_text_with_captions()[:500])


if __name__ == "__main__":
    main()