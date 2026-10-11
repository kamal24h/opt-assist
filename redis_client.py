# redis_client.py
import json
import logging

import redis

from config import settings

logger = logging.getLogger(__name__)

redis_pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password,
    decode_responses=True,
)


def get_redis_client() -> redis.Redis:
    return redis.Redis(connection_pool=redis_pool)


class ConversationCache:
    """
    مدیریت کش مکالمه در Redis.

    ساختار کلیدها:
      conv:{conv_id}:meta     -> JSON شامل app_id, tenant_id, user_id, status
      conv:{conv_id}:messages -> لیست JSON پیام‌ها به ترتیب زمانی (قدیم -> جدید)

    منبع حقیقت: PostgreSQL.
    Redis صرفاً به‌عنوان کش سریع استفاده می‌شود. در صورت عدم دسترسی،
    سیستم باید بدون خطا به کار خود ادامه دهد.
    """

    # ---------- Meta ----------

    @staticmethod
    def create_conversation(
        redis_client: redis.Redis,
        conv_id: str,
        app_id: str,
        tenant_id: str,
        user_id: str,
        ttl_seconds: int = settings.redis_ttl_seconds,
    ) -> None:
        key = f"conv:{conv_id}:meta"
        data = {
            "conversation_id": conv_id,
            "app_id": app_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "status": "active",
        }
        redis_client.set(key, json.dumps(data), ex=ttl_seconds)

    @staticmethod
    def get_conversation_meta(redis_client: redis.Redis, conv_id: str):
        key = f"conv:{conv_id}:meta"
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None

    # ---------- Messages ----------

    @staticmethod
    def append_message(
        redis_client: redis.Redis,
        conv_id: str,
        role: str,
        content: str,
        ttl_seconds: int = settings.redis_ttl_seconds,
    ) -> None:
        """
        افزودن یک پیام به انتهای لیست پیام‌های مکالمه.
        TTL لیست پس از هر append تمدید می‌شود.
        """
        key = f"conv:{conv_id}:messages"
        msg = {"role": role, "content": content}
        pipe = redis_client.pipeline()
        pipe.rpush(key, json.dumps(msg, ensure_ascii=False))
        pipe.expire(key, ttl_seconds)
        pipe.execute()

    @staticmethod
    def get_messages(redis_client: redis.Redis, conv_id: str):
        key = f"conv:{conv_id}:messages"
        messages = redis_client.lrange(key, 0, -1)
        return [json.loads(m) for m in messages]

    @staticmethod
    def get_recent_messages(
        redis_client: redis.Redis,
        conv_id: str,
        limit: int,
    ):
        """
        دریافت آخرین `limit` پیام مکالمه به ترتیب زمانی (قدیم -> جدید).
        اگر تعداد پیام‌ها کمتر از limit باشد، همه برگردانده می‌شوند.
        """
        key = f"conv:{conv_id}:messages"
        messages = redis_client.lrange(key, -limit, -1)
        return [json.loads(m) for m in messages]

    @staticmethod
    def set_messages(
        redis_client: redis.Redis,
        conv_id: str,
        messages: list,
        ttl_seconds: int = settings.redis_ttl_seconds,
    ) -> None:
        """
        بازنویسی کل لیست پیام‌های مکالمه (برای repopulate از Postgres).
        """
        key = f"conv:{conv_id}:messages"
        pipe = redis_client.pipeline()
        pipe.delete(key)
        if messages:
            serialized = [json.dumps(m, ensure_ascii=False) for m in messages]
            pipe.rpush(key, *serialized)
            pipe.expire(key, ttl_seconds)
        pipe.execute()

    # ---------- Cleanup ----------

    @staticmethod
    def clear_conversation(redis_client: redis.Redis, conv_id: str) -> None:
        redis_client.delete(f"conv:{conv_id}:meta")
        redis_client.delete(f"conv:{conv_id}:messages")