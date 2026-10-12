# api/v1/messages.py
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from services.conversation_service import ConversationService
from services.rag_service import generate_rag_response
from schemas.conversation import MessageOut, SendMessageRequest

router = APIRouter(
    prefix="/api/v1/conversations/{conv_id}/messages",
    tags=["Messages"],
)


@router.get("", response_model=List[MessageOut])
def list_messages(
    conv_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return ConversationService.get_messages(
        db,
        conversation_id=conv_id,
        limit=limit,
        offset=offset,
    )


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

    # ۱. خواندن تاریخچه (۱۰ پیام آخر = ۵ نوبت user+assistant)
    history = ConversationService.get_recent_chat_history(
        db,
        conversation_id=conv_id,
        max_messages=settings.chat_history_max_messages,
    )

    # ۲. ثبت پیام کاربر (بدون duration و بدون error)
    ConversationService.add_message(
        db=db,
        conversation_id=conv_id,
        role="user",
        content=req.prompt,
    )

    # ۳. تولید پاسخ با RAG (اندازه‌گیری زمان)
    started_at = time.perf_counter()
    try:
        answer, sources = generate_rag_response(
            question=req.prompt,
            chat_history=history,
        )
        duration = time.perf_counter() - started_at

        # ۴. ثبت پاسخ دستیار (با duration)
        assistant_msg = ConversationService.add_message(
            db=db,
            conversation_id=conv_id,
            role="assistant",
            content=answer,
            meta_info={"sources": sources, "status": "completed"},
            duration=duration,
            error=None,
        )
        return assistant_msg

    except Exception as e:
        duration = time.perf_counter() - started_at
        error_text = str(e)

        # ثبت پیام assistant با وضعیت failed و متن خطا
        ConversationService.add_message(
            db=db,
            conversation_id=conv_id,
            role="assistant",
            content="متأسفانه در تولید پاسخ مشکلی رخ داد.",
            meta_info={"status": "failed", "error": error_text},
            duration=duration,
            error=error_text,
        )
        raise HTTPException(
            status_code=500,
            detail=f"خطا در پردازش هوش مصنوعی: {error_text}",
        )