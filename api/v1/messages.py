# api/v1/messages.py
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config import settings

from database import Message, get_db

from services.conversation_service import ConversationService
from services.linkify import linkify_html
from services.rag_service import generate_rag_response
from schemas.conversation import (
    MessageOut,
    SendMessageRequest,
    MessageLikeRequest,
)

router = APIRouter(
    prefix="/api/v1/conversations/{conv_id}/messages",
    tags=["Messages"],
)


def _linkify_message_dict(item: dict) -> dict:
    """Apply linkify_html to `answer` in a message dict (for response_model)."""
    if not isinstance(item, dict):
        return item
    answer = item.get("answer")
    if isinstance(answer, str) and answer:
        item = dict(item)
        item["answer"] = linkify_html(answer)
    return item


@router.get("", response_model=List[MessageOut])
def list_messages(
    conv_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    rows = ConversationService.get_messages(
        db,
        conversation_id=conv_id,
        limit=limit,
        offset=offset,
    )
    # Linkify each message's answer before FastAPI serializes it.
    result = []
    for row in rows:
        # `Message` is an ORM object; build dict with the fields MessageOut needs
        item = {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "query": row.query,
            "answer": linkify_html(row.answer) if row.answer else row.answer,
            "duration": row.duration,
            "error": row.error,
            "meta_info": row.meta_info,
            "liked": row.liked,
            "created_at": row.created_at,
            "role": row.role,
            "content": row.content,
        }
        result.append(item)
    return result


@router.post("", response_model=MessageOut)
def send_message(
    conv_id: uuid.UUID,
    req: SendMessageRequest,
    db: Session = Depends(get_db),
):
    conv = ConversationService.get_conversation(
        db, conv_id, tenant_id=req.tenantId, user_id=req.userId
    )
    if not conv:
        raise HTTPException(status_code=404, detail="مکالمه یافت نشد.")
    if conv.status != "active":
        raise HTTPException(status_code=400, detail="این مکالمه فعال نیست.")

    # خواندن تاریخچه با توجه به liked:
    # - liked=False حذف می‌شود (exclude_disliked=True)
    # - liked=True با پیشوند «[پاسخ تأییدشده توسط کاربر]» می‌آید
    history = ConversationService.get_recent_chat_history(
        db,
        conversation_id=conv_id,
        max_turns=settings.chat_history_max_turns,
        exclude_disliked=True,
    )

    turn = ConversationService.add_query(
        db=db,
        conversation_id=conv_id,
        query=req.prompt,
    )

    started_at = time.perf_counter()
    try:

        # لیست appIdها را از conversation بردار
        app_ids = conv.app_ids or ([conv.app_id] if conv.app_id else [])
        if not app_ids:
            raise HTTPException(
                status_code=400,
                detail="این مکالمه appId معتبری ندارد.",
            )

        answer, sources, form_id = generate_rag_response(
            question=req.prompt,
            app_ids=app_ids,
            chat_history=history,
        )
        duration = time.perf_counter() - started_at

        completed = ConversationService.complete_turn(
            db=db,
            message_id=turn.id,
            answer=answer,
            duration=duration,
            error=None,
            meta_info={
                "sources": sources or [],
                "status": "completed",
                "formId": form_id,
                "searchedAppIds": app_ids,      # ← برای ردیابی
            },
        )
        # Linkify the answer for the response (the DB keeps the raw answer).
        if completed and completed.answer:
            completed.answer = linkify_html(completed.answer)
        return completed

    except Exception as e:
        duration = time.perf_counter() - started_at
        error_text = str(e)

        ConversationService.complete_turn(
            db=db,
            message_id=turn.id,
            answer="متأسفانه در تولید پاسخ مشکلی رخ داد.",
            duration=duration,
            error=error_text,
            meta_info={"status": "failed"},
        )
        raise HTTPException(
            status_code=500,
            detail=f"خطا در پردازش هوش مصنوعی: {error_text}",
        )


@router.patch("/{message_id}/liked", response_model=MessageOut)
def like_message(
    conv_id: uuid.UUID,
    message_id: uuid.UUID,
    req: MessageLikeRequest,
    db: Session = Depends(get_db),
):
    msg = (
        db.query(Message)
        .filter(Message.id == message_id, Message.conversation_id == conv_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="پیام یافت نشد.")
    updated = ConversationService.set_message_like(db, message_id, req.liked)
    # Linkify the answer for the response (the DB keeps the raw answer).
    if updated and updated.answer:
        updated.answer = linkify_html(updated.answer)
    return updated