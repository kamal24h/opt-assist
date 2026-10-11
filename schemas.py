# schemas.py
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
import uuid
from pydantic import BaseModel, Field

# --- Conversation Schemas ---
class ConversationCreate(BaseModel):
    appId: str
    tenantId: str
    userId: str
    title: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[Literal["active", "closed", "archived"]] = None

class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    metaInfo: Optional[Dict[str, Any]] = None
    createdAt: datetime

    class Config:
        from_attributes = True

class ConversationOut(BaseModel):
    id: uuid.UUID
    appId: str
    tenantId: str
    userId: str
    title: Optional[str] = None
    status: str
    createdAt: datetime
    updatedAt: datetime
    endedAt: Optional[datetime] = None

    class Config:
        from_attributes = True

class ConversationDetailOut(ConversationOut):
    messages: List[MessageOut] = []

class MessageCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    # جهت اعتبارسنجی مالکیت درخواست در صورت نیاز:
    userId: Optional[str] = None
    tenantId: Optional[str] = None