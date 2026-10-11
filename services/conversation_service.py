# services/conversation_service.py

import uuid

from datetime import datetime

from sqlalchemy.orm import Session

from database import Conversation

from redis_client import (
    get_redis_client,
    ConversationCache,
)

from services.message_service import MessageService


class ConversationService:

    @staticmethod
    def begin(
        db: Session,
        app_id: str,
        tenant_id: str,
        user_id: str,
    ) -> Conversation:

        conversation = Conversation(
            id=uuid.uuid4(),
            app_id=app_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status="active",
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        redis_client = get_redis_client()

        ConversationCache.create_conversation(
            redis_client=redis_client,
            conv_id=str(conversation.id),
            app_id=app_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        return conversation

    @staticmethod
    def get(
        db: Session,
        conversation_id: uuid.UUID,
    ):

        return (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id
            )
            .first()
        )

    @staticmethod
    def is_active(
        conversation_id: uuid.UUID,
    ) -> bool:

        redis_client = get_redis_client()

        metadata = ConversationCache.get_conversation_meta(
            redis_client,
            str(conversation_id),
        )

        return bool(
            metadata
            and metadata.get("status") == "active"
        )

    @staticmethod
    def end(
        db: Session,
        conversation_id: uuid.UUID,
    ):

        conversation = ConversationService.get(
            db,
            conversation_id,
        )

        if not conversation:
            return None, 0

        redis_client = get_redis_client()

        cached_messages = (
            ConversationCache.get_messages(
                redis_client,
                str(conversation_id),
            )
        )

        try:

            message_count = (
                MessageService.archive_redis_messages(
                    db=db,
                    conversation_id=conversation_id,
                    cached_messages=cached_messages,
                )
            )

            conversation.status = "closed"
            conversation.ended_at = datetime.utcnow()

            db.commit()

            ConversationCache.clear_conversation(
                redis_client,
                str(conversation_id),
            )

            return conversation, message_count

        except Exception:
            db.rollback()
            raise