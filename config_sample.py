# config.py
import os
from typing import Optional
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # تنظیمات پایگاه داده رابطه‌ای (PostgreSQL)
    postgres_user: str = "POSTGRES_USER"
    postgres_password: str = "POSTGRES_PASSWORD"
    postgres_db: str = "POSTGRES_DB"
    postgres_host: str = "POSTGRES_HOST"
    postgres_port: int = "POSTGRES_PORT"

    # تنظیمات Vector DB (Qdrant)
    qdrant_url: str = "QDRANT_URL"
    qdrant_api_key: Optional[str] = "QDRANT_API_KEY"
    collection_name: str = "COLLECTION_NAME"

    # تنظیمات Ollama و هوش مصنوعی
    ollama_url: str = "OLLAMA_URL"
    llm_model: str = "LLM_MODEL"
    # llm_model: str = "partai/dorna-llama3"
    embedding_model: str = "EMBEDDING_MODEL"

    # تنظیمات Chunker و پردازش اسناد
    chunk_size: int = "CHUNK_SIZE"
    chunk_overlap: int = "CHUNK_OVERLAP"
    embed_batch_size: int = "EMBED_BATCH_SIZE"

    # تنظیمات Cache (Redis)
    redis_host: str = "REDIS_HOST"
    redis_port: int = "REDIS_PORT"
    redis_password: Optional[str] = "REDIS_PASSWORD"
    redis_ttl_seconds: int = 86400  # ۱ روز

    # تنظیمات تاریخچه مکالمه
    # حداکثر تعداد پیام‌هایی که به LLM داده می‌شود (۵ نوبت user+assistant = ۱۰ پیام)
    chat_history_max_messages: int = 10

    # تنظیمات RabbitMQ
    rabbitmq_host: str = "RABBITMQ_HOST"
    rabbitmq_port: int = "RABBITMQ_PORT"
    rabbitmq_user: str = "RABBITMQ_USER"
    rabbitmq_password: str = "RABBITMQ_PASSWORD"
    rabbitmq_vhost: str = "RABBITMQ_VHOST"
    rabbitmq_exchange: str = "RABBITMQ_EXCHANGE"
    rabbitmq_queue: str = "RABBITMQ_QUEUE"
    rabbitmq_routing_key: str = "RABBITMQ_ROUTING_KEY"
    rabbitmq_enabled: bool = "RABBITMQ_ENABLED"

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
        extra="ignore",
    )


settings = Settings()
