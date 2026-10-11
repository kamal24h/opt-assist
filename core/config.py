from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    qdrant_collection: str = "documents"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"

    chunk_size: int = 800
    chunk_overlap: int = 120

    embedding_batch_size: int = 32
    qdrant_batch_size: int = 64

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()