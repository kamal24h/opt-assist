# config.py
import os
from typing import Optional
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # تنظیمات پایگاه داده رابطه‌ای (PostgreSQL)
    postgres_user: str = "assistant_user"
    postgres_password: str = "aRian2502aSsist2026"
    postgres_db: str = "assistant_db"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # تنظیمات Vector DB (Qdrant)
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "assistant_knowledge1"

    # تنظیمات Ollama و هوش مصنوعی
    ollama_url: str = "http://ollama:11434"
    llm_model: str = "qwen3:8b"
    # llm_model: str = "partai/dorna-llama3"
    embedding_model: str = "qllama/bge-m3:q4_k_m"

    # تنظیمات Chunker و پردازش اسناد
    chunk_size: int = 800
    chunk_overlap: int = 120
    embed_batch_size: int = 32

    # تنظیمات Cache (Redis)
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_ttl_seconds: int = 86400  # ۱ روز

    # تنظیمات تاریخچه مکالمه
    # حداکثر تعداد پیام‌هایی که به LLM داده می‌شود (۵ نوبت user+assistant = ۱۰ پیام)
    chat_history_max_messages: int = 10

    # تولید URL دیتابیس بدون نیاز به ترکیب دستی رشته‌ها
    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # اولویت‌دهی به متغیرهای محیطی سیستم و سپس فایل .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # نادیده گرفتن متغیرهای اضافه موجود در .env
    )


settings = Settings()