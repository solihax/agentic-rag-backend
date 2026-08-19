"""
LLM provider factory - routes all Gemini chat calls through the
mentor-issued OpenAI-compatible proxy.

Why this file exists:
    Multiple graph nodes (grader, generator, hallucination checker,
    supervisor) each need an LLM client. Without this factory, each node
    would hardcode its own `ChatOpenAI(base_url=..., api_key=..., model=...)`
    call - meaning a proxy URL change or model rename would require editing
    every node file individually. Centralizing it here means that change
    happens in exactly one place (`backend/config/settings.py`).

Model routing (per project spec):
    - "flash_lite": cheap/fast model for high-volume calls
      (document relevance grading, hallucination grading, answer grading).
    - "flash": stronger model for supervisor/critic-level reasoning
      and final answer generation.

Never print or log the API key. LangChain's ChatOpenAI does not print
it either, but we're explicit about this here as a reminder for anyone
extending this file.
"""

from typing import Literal

from langchain_openai import ChatOpenAI

from backend.config.settings import get_settings

ModelTier = Literal["flash_lite", "flash"]


def get_chat_llm(tier: ModelTier = "flash_lite", temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a configured ChatOpenAI instance pointed at the Gemini proxy.

    Args:
        tier: "flash_lite" for high-volume/cheap calls (graders), or
              "flash" for supervisor/critic/final-generation calls.
        temperature: Sampling temperature. Defaults to 0.0 because graders
                     and structured-output nodes need deterministic,
                     repeatable YES/NO-style answers. Override this for the
                     final generation node if you want more natural prose
                     (e.g. temperature=0.3).

    Raises:
        ValueError: if GEMINI_API_KEY is missing from the environment,
                    so failures happen at call time with a clear message
                    instead of a confusing 401 from the proxy.
    """
    settings = get_settings()

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before calling any LLM."
        )

    model_name = (
        settings.gemini_model_flash_lite
        if tier == "flash_lite"
        else settings.gemini_model_flash
    )

    return ChatOpenAI(
        base_url=settings.gemini_proxy_openai_base_url,
        api_key=settings.gemini_api_key,
        model=model_name,
        temperature=temperature,
    )
