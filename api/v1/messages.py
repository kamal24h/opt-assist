# api/v1/messages.py

from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy.orm import Session

from database import get_db

from schemas.message import MessageResponse

from services.conversation_service import (
    ConversationService,
)

from services.message_service import (
    MessageService,
)


router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Messages"],
)


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def get_conversation_messages(
    conversation_id: UUID,
    db: Session = Depends(get_db),
):

    conversation = ConversationService.get(
        db,
        conversation_id,
    )

    if not conversation:

        raise HTTPException(
            status_code=404,
            detail="مکالمه یافت نشد.",
        )

    return MessageService.get_messages(
        db,
        conversation_id,
    )