# schemas/conversation.py

import uuid

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class BeginConversationRequest(BaseModel):
    app_id: str = Field(alias="appId")
    tenant_id: str = Field(alias="tenantId")
    user_id: str = Field(alias="userId")

    model_config = ConfigDict(
        populate_by_name=True
    )


class BeginConversationResponse(BaseModel):
    conversation_id: uuid.UUID = Field(alias="conversationId")
    status: str
    app_id: str = Field(alias="appId")
    tenant_id: str = Field(alias="tenantId")
    user_id: str = Field(alias="userId")

    model_config = ConfigDict(
        populate_by_name=True
    )


class QueryConversationRequest(BaseModel):
    conversation_id: Optional[uuid.UUID] = Field(
        default=None,
        alias="conversationId",
    )

    prompt: str

    model_config = ConfigDict(
        populate_by_name=True
    )


class QueryConversationResponse(BaseModel):
    answer: str
    conversation_id: Optional[uuid.UUID] = Field(
        default=None,
        alias="conversationId",
    )

    model_config = ConfigDict(
        populate_by_name=True
    )


class EndConversationRequest(BaseModel):
    conversation_id: uuid.UUID = Field(
        alias="conversationId"
    )

    model_config = ConfigDict(
        populate_by_name=True
    )


class EndConversationResponse(BaseModel):
    status: str
    message: str
    total_messages_saved: int

