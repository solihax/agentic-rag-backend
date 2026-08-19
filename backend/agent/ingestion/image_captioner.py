"""
Image captioning using the Gemini vision model, routed through the same
OpenAI-compatible proxy as chat and embeddings.
"""

import base64
import logging

from backend.agent.llm_provider import get_chat_llm
from backend.models.document import ExtractedDocument

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe this image in 2-3 sentences, factually and specifically. "
    "Mention any text, charts, diagrams, numbers, or labels visible in "
    "the image, since this description will be used for document search."
)


def caption_image(image_bytes: bytes, image_format: str) -> str:
    llm = get_chat_llm(tier="flash_lite")

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/{image_format};base64,{b64_image}"

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

    response = llm.invoke([message])
    return response.content.strip()


def caption_all_images(document: ExtractedDocument) -> ExtractedDocument:
    total_images = sum(len(page.images) for page in document.pages)
    captioned_count = 0

    for page in document.pages:
        for image in page.images:
            try:
                image.caption = caption_image(image.image_bytes, image.image_format)
                captioned_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to caption image %s: %s", image.image_id, e
                )

    logger.info("Captioned %s/%s images", captioned_count, total_images)
    return document