# conversation_service.py
import uuid
from datetime import datetime
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database import Conversation, Message

class ConversationService:
    @staticmethod
    def create_conversation(db: Session, app_id: str, tenant_id: str, user_id: str, title: Optional[str] = None) -> Conversation:
        conv = Conversation(
            id=uuid.uuid4(),
            app_id=app_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or "گفتگوی جدید",
            status="active"
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    @staticmethod
    def get_conversation(db: Session, conv_id: uuid.UUID, tenant_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Conversation]:
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
        offset: int = 0
    ) -> List[Conversation]:
        query = db.query(Conversation).filter(
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id
        )
        if app_id:
            query = query.filter(Conversation.app_id == app_id)
        
        return query.order_by(desc(Conversation.updated_at)).offset(offset).limit(limit).all()

    @staticmethod
    def update_conversation(db: Session, conv: Conversation, title: Optional[str] = None, status: Optional[str] = None) -> Conversation:
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

    @staticmethod
    def add_message(
        db: Session, 
        conversation_id: uuid.UUID, 
        role: str, 
        content: str, 
        meta_info: Optional[dict] = None
    ) -> Message:
        msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            meta_info=meta_info
        )
        db.add(msg)
        
        # به‌روزرسانی updated_at مکالمه
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.updated_at = datetime.utcnow()
            # اگر عنوان پیش‌فرض است و این اولین پیام کاربر است، ساخت عنوان از روی پیام
            if role == "user" and (conv.title == "گفتگوی جدید" or not conv.title):
                conv.title = content[:60] + ("..." if len(content) > 60 else "")

        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def get_messages(db: Session, conversation_id: uuid.UUID, limit: int = 50, offset: int = 0) -> List[Message]:
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_recent_chat_history(db: Session, conversation_id: uuid.UUID, max_messages: int = 6) -> List[Tuple[str, str]]:
        """بازیابی آخرین پیام‌ها برای تزریق به Chat History زنجیره LLM"""
        recent = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at))
            .limit(max_messages)
            .all()
        )
        # برگرداندن به ترتیب زمانی صعودی
        ordered = reversed(recent)
        return [(m.role, m.content) for m in ordered]