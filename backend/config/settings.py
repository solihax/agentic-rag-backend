"""
Central application configuration.

Why this file exists:
    Every other module (ingestion, embeddings, retriever, graph nodes, API)
    needs shared settings such as API keys, model names, chunk sizes, and
    retrieval parameters. Instead of reading `os.getenv(...)` in a dozen
    different files (error-prone, no validation, silent typos), we define
    ONE typed settings object here. If a required variable is missing or
    malformed, the app fails fast at startup with a clear error instead of
    failing mysteriously three layers deep during a user request.

Design notes:
    - Uses `pydantic-settings` (the standalone package; in Pydantic v2,
      BaseSettings moved out of pydantic core into pydantic-settings).
    - Values are loaded from a `.env` file (see `.env.example`) and/or
      real environment variables. Real env vars always take precedence
      over `.env` file values, which is what you want in Docker/HF Spaces.
    - `Settings` is instantiated once as a singleton (`get_settings()`)
      and reused everywhere via dependency injection, so we don't reread
      or revalidate the .env file on every function call.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # LLM Provider
    # ------------------------------------------------------------------
    # "gemini" is implemented first. "openai" is a placeholder for the
    # provider-abstraction step later -- the interface will support it
    # without changing any node/chain code.
    llm_provider: Literal["gemini", "openai"] = "gemini"

    gemini_api_key: str = Field(
        default="",
        description="API key for the Gemini proxy (mentor-issued or your own). Required if llm_provider=gemini.",
    )
    openai_api_key: str = Field(
        default="",
        description="Reserved for future direct-OpenAI provider support.",
    )

    # Mentor-issued LiteLLM proxy. OpenAI-compatible endpoint is used for
    # chat + embeddings (via langchain-openai). The native google-genai SDK
    # endpoint is kept separate in case any code path needs raw Gemini calls.
    gemini_proxy_openai_base_url: str = Field(
        default="https://saidazam-litellm-proxy.hf.space/v1",
        description="OpenAI-compatible base URL for the Gemini proxy (used by ChatOpenAI/OpenAIEmbeddings).",
    )
    gemini_proxy_native_base_url: str = Field(
        default="https://saidazam-litellm-proxy.hf.space/gemini",
        description="Native google-genai SDK base URL for the Gemini proxy.",
    )

    # Model routing per mentor's instructions:
    #   - flash-lite: high-volume / cheap calls (grading, retrieval helpers)
    #   - flash: supervisor / critic / final generation (higher quality)
    #   - embedding: all embedding calls
    gemini_model_flash_lite: str = "gemini-flash-lite"
    gemini_model_flash: str = "gemini-flash"
    gemini_embedding_model: str = "gemini-embedding"

    # ------------------------------------------------------------------
    # Web Search Provider
    # ------------------------------------------------------------------
    web_search_provider: Literal["tavily", "none"] = "tavily"
    tavily_api_key: str = ""

    # ------------------------------------------------------------------
    # Vector Store (Qdrant)
    # ------------------------------------------------------------------
    qdrant_mode: Literal["local", "cloud"] = "local"
    qdrant_path: str = "./qdrant_data"  # used when qdrant_mode == "local"
    qdrant_url: str = ""                # used when qdrant_mode == "cloud"
    qdrant_api_key: str = ""            # used when qdrant_mode == "cloud"
    qdrant_collection_name: str = "agentic_rag_documents"

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    retrieval_top_k: int = 4

    # ------------------------------------------------------------------
    # Agent / Graph control
    # ------------------------------------------------------------------
    max_generation_retries: int = 2
    max_web_search_retries: int = 1

    # ------------------------------------------------------------------
    # API / App
    # ------------------------------------------------------------------
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_chunk_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be smaller than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("retrieval_top_k", "max_generation_retries", "max_web_search_retries")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be >= 0")
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    `lru_cache` ensures the .env file is parsed and validated only once
    per process, not on every call. FastAPI routes and graph nodes should
    always fetch settings via this function (dependency injection),
    never by instantiating `Settings()` directly.
    """
    return Settings()
