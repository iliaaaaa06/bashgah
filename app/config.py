from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- API ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    admin_api_key: str = Field(..., min_length=16)
    cors_origins: str = "*"  # comma separated

    # --- LLM (any OpenAI-compatible server: llama.cpp llama-server, vLLM, ...) ---
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "local-model"
    llm_max_tokens: int = 1024
    llm_timeout: float = 180.0

    # --- Embedding ---
    # If set, embeddings come from this OpenAI-compatible server; otherwise the model is loaded in-process
    embedding_base_url: str = ""
    embedding_api_key: str = "not-needed"
    # Remote: model name on the server. Local: HF name or a local folder path
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"  # auto | cuda | mps | cpu
    embedding_batch_size: int = 16

    # --- Vector DB / storage ---
    chroma_path: Path = BASE_DIR / "data" / "chroma"
    chroma_collection: str = "persian_docs"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    max_upload_mb: int = 50

    # --- RAG ---
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    # cosine similarity (0..1); chunks below this are treated as irrelevant
    relevance_threshold: float = 0.45

    # --- Web search ---
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"  # duckduckgo | tavily
    tavily_api_key: str = ""
    web_search_max_results: int = 5
    web_search_region: str = "wt-wt"

    @field_validator("chroma_path", "upload_dir")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return v if v.is_absolute() else (BASE_DIR / v).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
