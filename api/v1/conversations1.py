# api/v1/conversations.py

from fastapi import (APIRouter, Depends, HTTPException)

from sqlalchemy.orm import Session

from database import get_db

from schemas.conversation import (
    BeginConversationRequest,
    BeginConversationResponse,
    EndConversationRequest,
    EndConversationResponse,
    QueryConversationRequest,
    QueryConversationResponse,
)

from services.conversation_service import (ConversationService)

from services.assistant_service import AssistantService

assistant_service = AssistantService()

router = APIRouter()

@router.post("/query", response_model=QueryConversationResponse)
async def query_assistant(request: QueryConversationRequest):

    # حالت قدیمی:
    # {
    #   "prompt": "..."
    # }

    if request.conversation_id is None:
        answer = assistant_service.ask(request.prompt)
        return QueryConversationResponse(answer=answer, conversationId=None)

    # حالت Conversation

    if not ConversationService.is_active(request.conversation_id):

        raise HTTPException(status_code=404,
         detail=(
                "مکالمه یافت نشد یا "
                "پایان یافته است."
            ))

    try:

        answer = assistant_service.ask_conversation(
            conversation_id=request.conversation_id,
            prompt=request.prompt,
        )

        return QueryConversationResponse(
            conversationId=request.conversation_id,
            answer=answer,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.post("/beginConversation",response_model=BeginConversationResponse)
async def begin_conversation(request: BeginConversationRequest, db: Session = Depends(get_db)):

    try:

        conversation = ConversationService.begin(
            db=db,
            app_id=request.app_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
        )

        return BeginConversationResponse(
            conversationId=conversation.id,
            status="created",
            appId=conversation.app_id,
            tenantId=conversation.tenant_id,
            userId=conversation.user_id,
        )

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to begin conversation: "
                f"{str(exc)}"
            ),
        )


@router.post("/endConversation", response_model=EndConversationResponse)
async def end_conversation(request: EndConversationRequest, db: Session = Depends(get_db)):

    try:

        conversation, message_count = (
            ConversationService.end(
                db=db,
                conversation_id=request.conversation_id,
            )
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="مکالمه در پایگاه داده یافت نشد.",
            )

        return EndConversationResponse(
            status="success",
            message=(
                "مکالمه پایان یافت و داده‌ها "
                "در پایگاه داده اصلی آرشیو شدند."
            ),
            total_messages_saved=message_count,
        )

    except HTTPException:
        raise

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to end conversation: "
                f"{str(exc)}"
            ),
        )