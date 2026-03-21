#config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: str
    COHERE_API_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    REDIS_BROKER_URL: str
    REDIS_CACHE_URL: str
    QDRANT_URL: str
    HUGGINGFACE_API_KEY: str

    # MinIO (local S3-compatible storage — replaces Cloudflare R2)
    MINIO_ENDPOINT_URL: str = "http://minio:9000"
    MINIO_ACCESS_KEY_ID: str = "minioadmin"
    MINIO_SECRET_ACCESS_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "multirag"
    # Public URL used to construct download links for stored images.
    # In local dev this is http://localhost:9000/multirag
    MINIO_PUBLIC_URL: str = "http://localhost:9000/multirag"

    # Security
    # Similarity threshold (0-1) for semantic cache hits.
    # 0.92 = very high similarity required before serving cached answer.
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    # Rate limits (requests per minute per IP)
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_CHAT: str = "30/minute"
    # Maximum allowed upload file size in bytes (default 50 MB)
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
