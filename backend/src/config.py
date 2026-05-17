"""Application configuration loaded from environment variables.

Sử dụng pydantic-settings để type-safe + validate khi startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

def _detect_project_root() -> Path:
    """Detect project root linh hoạt:

    - Dev: `backend/src/config.py` → project_AI/ (parent.parent.parent).
    - Container: `/app/src/config.py` → /app (parent.parent).

    Heuristic: dir nào có `data/` subdir thì chính nó.
    """
    here = Path(__file__).resolve()
    # Walk up tối đa 4 levels tìm dir có `data/`.
    for parent in (here.parent.parent.parent, here.parent.parent, here.parent):
        if (parent / "data").is_dir() or (parent / "backend").is_dir():
            return parent
    return here.parent.parent  # fallback


PROJECT_ROOT = _detect_project_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: Literal["dev", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # LLM provider
    llm_provider: Literal["openrouter", "gemini", "deepseek", "custom"] = "openrouter"
    llm_model: str = "google/gemini-2.5-flash"
    # SecretStr → repr hiện "**********" thay vì value thật. .get_secret_value() để dùng.
    gemini_api_key: SecretStr = SecretStr("")
    gemini_api_key_backup: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    deepseek_api_key: SecretStr = SecretStr("")

    # Custom OpenAI-compatible endpoint (khi llm_provider="custom")
    custom_llm_base_url: str = ""
    custom_llm_api_key: SecretStr = SecretStr("")
    custom_llm_model: str = ""  # fallback dùng `llm_model` nếu trống

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    # Vector store
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "legal_vn"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_llm_cache: int = 86400
    redis_ttl_query_embed: int = 604800
    redis_ttl_session: int = 3600

    # Auth
    jwt_secret: SecretStr = SecretStr("0sgs1jDlfynXbiovwr8OhK24Y9FaX-QJz1ASHsVpg-s")
    jwt_algorithm: str = "HS256"
    # NoDecode: bỏ qua JSON-decode mặc định của pydantic-settings → validator xử lý CSV.
    # list[SecretStr] nặng nề; giữ list[str] nhưng repr custom bên dưới sẽ ẩn.
    allowed_api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=list, repr=False
    )

    # Rate limit
    rate_limit_per_minute: int = 30

    # Retrieval
    top_k_dense: int = 30
    top_k_bm25: int = 30
    top_k_final: int = 10
    enable_rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # CORS
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    @field_validator("allowed_api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s) for s in v]
        return []

    @field_validator("jwt_secret", mode="after")
    @classmethod
    def _warn_default_secret(cls, v: SecretStr) -> SecretStr:
        if v.get_secret_value().startswith("change-me"):
            import warnings

            warnings.warn(
                "JWT_SECRET is using default placeholder. "
                "Set a proper random secret in production.",
                stacklevel=2,
            )
        return v

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


settings = Settings()
