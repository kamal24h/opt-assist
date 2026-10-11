# schemas/conversation.py
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# class BeginConversationRequest(BaseModel):
#     app_id: str = Field(alias="appId")
#     tenant_id: str = Field(alias="tenantId")
#     user_id: str = Field(alias="userId")

#     model_config = ConfigDict(
#         populate_by_name=True
#     )
# class BeginConversationResponse(BaseModel):
#     conversation_id: uuid.UUID = Field(alias="conversationId")
#     status: str
#     app_id: str = Field(alias="appId")
#     tenant_id: str = Field(alias="tenantId")
#     user_id: str = Field(alias="userId")

#     model_config = ConfigDict(
#         populate_by_name=True
#     )
# class QueryConversationRequest(BaseModel):
#     conversation_id: Optional[uuid.UUID] = Field(
#         default=None,
#         alias="conversationId",
#     )

#     prompt: str

#     model_config = ConfigDict(
#         populate_by_name=True
#     )
# class QueryConversationResponse(BaseModel):
#     answer: str
#     conversation_id: Optional[uuid.UUID] = Field(
#         default=None,
#         alias="conversationId",
#     )

#     model_config = ConfigDict(
#         populate_by_name=True
#     )
# class EndConversationRequest(BaseModel):
#     conversation_id: uuid.UUID = Field(
#         alias="conversationId"
#     )

#     model_config = ConfigDict(
#         populate_by_name=True
#     )
# class EndConversationResponse(BaseModel):
    # status: str
    # message: str
    # total_messages_saved: int

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
    meta_info: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True
class ConversationOut(BaseModel):
    id: uuid.UUID
    app_id: str
    tenant_id: str
    user_id: str
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