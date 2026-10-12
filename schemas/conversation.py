# schemas/conversation.py
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    appId: str
    tenantId: str
    userId: str
    userName: Optional[str] = None
    tenantDesc: Optional[str] = None
    title: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[Literal["active", "closed", "archived"]] = None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    meta_info: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: uuid.UUID
    app_id: str
    tenant_id: str
    user_id: str
    user_name: Optional[str] = None
    tenant_desc: Optional[str] = None
    title: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    ended_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConversationDetailOut(ConversationOut):
    messages: List[MessageOut] = []


class SendMessageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    tenantId: Optional[str] = None
    userId: Optional[str] = None