# api/v1/conversations.py
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
#from security import require_api_key
from sqlalchemy.orm import Session

from database import get_db
from services.conversation_service import ConversationService
from services.linkify import linkify_html
from schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationOut,
    ConversationDetailOut,
    ConversationFeedbackRequest,
)

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Conversations"],
    #dependencies=[Depends(require_api_key)],
)


def _message_to_dict_with_linkify(m) -> dict:
    """Build a MessageOut-shaped dict with `answer` linkified.

    Avoids mutating the ORM object (and avoids accidental commits via autoflush).
    """
    answer = m.answer
    if isinstance(answer, str) and answer:
        answer = linkify_html(answer)
    return {
        "id": m.id,
        "conversation_id": m.conversation_id,
        "query": m.query,
        "answer": answer,
        "duration": m.duration,
        "error": m.error,
        "meta_info": m.meta_info,
        "liked": m.liked,
        "created_at": m.created_at,
        "role": m.role,
        "content": m.content,
    }


def _conversation_to_detail_dict_with_linkify(conv) -> dict:
    """Build a ConversationDetailOut-shaped dict with messages linkified."""
    messages = [
        _message_to_dict_with_linkify(m)
        for m in (conv.messages or [])
    ]
    app_ids = getattr(conv, "app_ids", None)
    if not app_ids:
        single = getattr(conv, "app_id", None)
        app_ids = [single] if single else []
    return {
        "id": conv.id,
        "app_ids": list(app_ids),
        "app_id": app_ids[0] if app_ids else None,
        "tenant_id": conv.tenant_id,
        "user_id": conv.user_id,
        "user_name": conv.user_name,
        "tenant_desc": conv.tenant_desc,
        "title": conv.title,
        "status": conv.status,
        "feedback_score": conv.feedback_score,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "ended_at": conv.ended_at,
        "messages": messages,
    }


@router.post("", response_model=ConversationOut)
def create_conversation(req: ConversationCreate, db: Session = Depends(get_db)):
    return ConversationService.create_conversation(
        db=db,
        app_ids=req.appIds,
        tenant_id=req.tenantId,
        user_id=req.userId,
        title=req.title,
        user_name=req.userName,
        tenant_desc=req.tenantDesc,
    )


@router.get("", response_model=List[ConversationOut])
def list_conversations(
    tenantId: str = Query(...),
    userId: str = Query(...),
    appId: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    return ConversationService.list_conversations(
        db=db,
        tenant_id=tenantId,
        user_id=userId,
        app_id=appId,
        limit=limit,
        offset=offset
    )


@router.get("/{conv_id}", response_model=ConversationDetailOut)
def get_conversation(
    conv_id: uuid.UUID,
    tenantId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    conv = ConversationService.get_conversation(db, conv_id, tenant_id=tenantId, user_id=userId)
    if not conv:
        raise HTTPException(status_code=404, detail="مکالمه یافت نشد.")
    return _conversation_to_detail_dict_with_linkify(conv)


@router.patch("/{conv_id}", response_model=ConversationOut)
def update_conversation(
    conv_id: uuid.UUID,
    req: ConversationUpdate,
    db: Session = Depends(get_db)
):
    conv = ConversationService.get_conversation(db, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="مکالمه یافت نشد.")
    return ConversationService.update_conversation(db, conv, title=req.title, status=req.status)


@router.patch("/{conv_id}/feedback")
def set_conversation_feedback(
    conv_id: uuid.UUID,
    req: ConversationFeedbackRequest,
    db: Session = Depends(get_db),
):
    conv = ConversationService.set_conversation_feedback(
        db, conv_id, req.feedbackScore
    )
    if not conv:
        raise HTTPException(status_code=404, detail="مکالمه یافت نشد.")
    return {"status": "success"}