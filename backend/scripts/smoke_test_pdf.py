"""
Usage:
    python -m backend.scripts.smoke_test_pdf "path\\to\\your\\file.pdf"
"""

import sys

from backend.agent.ingestion.pdf_extractor import extract_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.smoke_test_pdf <path_to_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]
    doc = extract_pdf(file_path)

    print(f"Filename: {doc.filename}")
    print(f"Pages: {len(doc.pages)}")

    for page in doc.pages:
        print(f"\n--- Page {page.page_number} ---")
        print(f"Text length: {len(page.text)} chars")
        print(f"Images found: {len(page.images)}")
        if page.text.strip():
            print(f"Preview: {page.text[:150]!r}")


if __name__ == "__main__":
    main()