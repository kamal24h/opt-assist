# services/conversation_service.py
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Tuple

from sqlalchemy import desc
from sqlalchemy.orm import Session

from config import settings
from database import Conversation, Message
from message_broker import MessageBroker, build_message_payload
from redis_client import ConversationCache, get_redis_client

logger = logging.getLogger(__name__)


def _safe_redis_client():
    """
    دریافت کلاینت Redis با مدیریت خطا.
    اگر Redis در دسترس نباشد، None برمی‌گرداند تا سیستم روی Postgres ادامه دهد.
    """
    try:
        return get_redis_client()
    except Exception as exc:
        logger.warning("Redis client unavailable: %s", exc)
        return None


class ConversationService:

    # ---------- Conversation CRUD ----------

    @staticmethod
    def create_conversation(
        db: Session,
        app_id: str,
        tenant_id: str,
        user_id: str,
        title: Optional[str] = None,
    ) -> Conversation:
        conv = Conversation(
            id=uuid.uuid4(),
            app_id=app_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or "گفتگوی جدید",
            status="active",
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)

        # dual-write meta به Redis (best-effort)
        redis_client = _safe_redis_client()
        if redis_client is not None:
            try:
                ConversationCache.create_conversation(
                    redis_client,
                    str(conv.id),
                    app_id=conv.app_id,
                    tenant_id=conv.tenant_id,
                    user_id=conv.user_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to cache conversation meta %s: %s", conv.id, exc
                )

        return conv

    @staticmethod
    def get_conversation(
        db: Session,
        conv_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Conversation]:
        query = db.query(Conversation).filter(Conversation.id == conv_id)
        if tenant_id:
            query = query.filter(Conversation.tenant_id == tenant_id)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        return query.first()

    @staticmethod
    def list_conversations(
        db: Session,
        tenant_id: str,
        user_id: str,
        app_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Conversation]:
        query = db.query(Conversation).filter(
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
        )
        if app_id:
            query = query.filter(Conversation.app_id == app_id)

        return (
            query.order_by(desc(Conversation.updated_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update_conversation(
        db: Session,
        conv: Conversation,
        title: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Conversation:
        if title is not None:
            conv.title = title
        if status is not None:
            conv.status = status
            if status == "closed":
                conv.ended_at = datetime.utcnow()
        conv.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(conv)
        return conv

    # ---------- Messages ----------

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        meta_info: Optional[dict] = None,
        redis_client=None,
    ) -> Message:
        """
        ثبت پیام در PostgreSQL (منبع حقیقت)، سپس dual-write به Redis
        و انتشار رویداد به RabbitMQ.

        در صورت عدم دسترسی به Redis یا RabbitMQ، خطا پرتاب نمی‌شود و
        سیستم روی Postgres به کار خود ادامه می‌دهد.
        """
        msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            meta_info=meta_info,
        )
        db.add(msg)

        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )
        if conv:
            conv.updated_at = datetime.utcnow()
            if role == "user" and (conv.title == "گفتگوی جدید" or not conv.title):
                conv.title = content[:60] + ("..." if len(content) > 60 else "")

        db.commit()
        db.refresh(msg)

        # dual-write به Redis (best-effort)
        if redis_client is None:
            redis_client = _safe_redis_client()

        if redis_client is not None:
            try:
                ConversationCache.append_message(
                    redis_client,
                    str(conversation_id),
                    role=role,
                    content=content,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to append message to Redis for %s: %s",
                    conversation_id,
                    exc,
                )

        # انتشار رویداد پیام به RabbitMQ (best-effort)
        if settings.rabbitmq_enabled:
            try:
                payload = build_message_payload(
                    message_id=str(msg.id),
                    conversation_id=str(conversation_id),
                    tenant_id=conv.tenant_id if conv else None,
                    user_id=conv.user_id if conv else None,
                    app_id=conv.app_id if conv else None,
                    role=role,
                    content=content,
                    meta_info=meta_info,
                    created_at=msg.created_at,
                )
                MessageBroker.publish_message_event(payload)
            except Exception as exc:
                logger.warning(
                    "Failed to publish message event for %s: %s", msg.id, exc
                )

        return msg

    @staticmethod
    def get_messages(
        db: Session,
        conversation_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Message]:
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_recent_chat_history(
        db: Session,
        conversation_id: uuid.UUID,
        max_messages: int = settings.chat_history_max_messages,
        redis_client=None,
    ) -> List[Tuple[str, str]]:
        """
        دریافت آخرین `max_messages` پیام مکالمه به ترتیب زمانی (قدیم -> جدید).

        استراتژی:
          ۱. تلاش برای خواندن از Redis
          ۲. در صورت خالی/خطا، خواندن از PostgreSQL
          ۳. در صورت موفقیت از Postgres، repopulate کردن Redis (best-effort)
        """
        if redis_client is None:
            redis_client = _safe_redis_client()

        # ۱. تلاش از Redis
        if redis_client is not None:
            try:
                cached = ConversationCache.get_recent_messages(
                    redis_client, str(conversation_id), max_messages
                )
                if cached:
                    return [(m["role"], m["content"]) for m in cached]
            except Exception as exc:
                logger.warning(
                    "Redis history read failed for %s: %s", conversation_id, exc
                )

        # ۲. Fallback به Postgres
        recent = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at))
            .limit(max_messages)
            .all()
        )
        messages: List[Tuple[str, str]] = [
            (m.role, m.content) for m in reversed(recent)
        ]

        # ۳. Repopulate Redis از Postgres (best-effort)
        if redis_client is not None and messages:
            try:
                ConversationCache.set_messages(
                    redis_client,
                    str(conversation_id),
                    [{"role": r, "content": c} for r, c in messages],
                )
            except Exception as exc:
                logger.warning(
                    "Redis repopulate failed for %s: %s", conversation_id, exc
                )

        return messages