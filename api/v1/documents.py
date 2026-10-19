# api/v1/documents.py
import hashlib
import json
import logging
import os
import shutil
import uuid
from typing import List, Optional, Dict, Any, Literal

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, Document
from ingest_service import (
    process_and_index_document,
    process_and_index_text_document,
    extract_text_from_file,
    compute_content_hash,
    delete_document_vectors_by_doc_id,
    delete_document_vectors_by_logical_id,
    delete_document_vectors_by_app_id,   # ← جدید
    mark_version_not_current,
    mark_version_current,
)
from schemas.document import DocumentOut, DocumentVersionItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


# ---------- Schemas ----------

class TextDocumentItem(BaseModel):
    doc_name: str
    metadata: Optional[Dict[str, Any]] = None
    doc_type: Literal["text"] = "text"
    doc: str


class TextDocumentsRequest(BaseModel):
    app_id: str
    version: Optional[str] = None
    documents: List[TextDocumentItem]


class CreatedDocumentItem(BaseModel):
    documentId: Optional[str] = None
    logicalDocId: str
    version: int
    docName: str
    chunksCreated: int = 0
    status: Literal["created", "skipped"]
    reason: Optional[str] = None


# ---------- Helpers ----------

def _to_doc_out(d: Document) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        logicalDocId=d.logical_doc_id,
        version=d.version,
        isCurrent=d.is_current,
        previousVersionId=d.previous_version_id,
        contentHash=d.content_hash,
        docName=d.doc_name,
        docType=d.doc_type,
        department=d.department,
        appId=d.app_id,
        metadata=d.meta_info,
        chunksCount=d.chunks_count,
        createdAt=d.created_at,
    )


def _find_current_document(
    db: Session, app_id: str, doc_name: str
) -> Optional[Document]:
    return (
        db.query(Document)
        .filter(
            Document.app_id == app_id,
            Document.doc_name == doc_name,
            Document.is_current == True,
        )
        .first()
    )


def _demote_current(db: Session, current: Document) -> None:
    """نسخه جاری را غیرجاری می‌کند."""
    current.is_current = False


def _compensate_qdrant(written_vectors: List[tuple]) -> None:
    """
    جبران best-effort: حذف وکتورهایی که در این درخواست نوشته شدند
    ولی تراکنش Postgres شکست خورد.
    written_vectors: لیست تاپل‌های (doc_id, logical_doc_id, version)
    """
    for doc_id, _logical, _ver in written_vectors:
        try:
            delete_document_vectors_by_doc_id(doc_id)
        except Exception as exc:
            logger.warning(
                "Qdrant compensation failed for doc_id=%s: %s", doc_id, exc
            )


def _created_item(
    doc_id: uuid.UUID,
    logical_doc_id: uuid.UUID,
    version: int,
    doc_name: str,
    chunks_count: int,
) -> CreatedDocumentItem:
    return CreatedDocumentItem(
        documentId=str(doc_id),
        logicalDocId=str(logical_doc_id),
        version=version,
        docName=doc_name,
        chunksCreated=chunks_count,
        status="created",
    )


# ---------- Endpoints ----------

@router.get("", response_model=List[DocumentOut])
def list_documents(
    appId: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    includeAllVersions: bool = Query(False),
    db: Session = Depends(get_db),
):
    q = db.query(Document)
    if appId:
        q = q.filter(Document.app_id == appId)
    if department:
        q = q.filter(Document.department == department)
    if not includeAllVersions:
        q = q.filter(Document.is_current == True)

    docs = q.order_by(Document.created_at.desc()).all()
    return [_to_doc_out(d) for d in docs]


@router.post("/create")
def create_document_file(
    file: UploadFile = File(...),
    docName: str = Form(...),
    docType: str = Form(...),
    department: str = Form(...),
    appId: str = Form(...),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    allowed_types = ["pdf", "docx"]
    if docType.lower() not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"نوع فایل غیرمجاز است. تنها فرمت‌های {allowed_types} پذیرفته می‌شوند.",
        )

    parsed_metadata: Dict[str, Any] = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="متادیتا باید با ساختار معتبر JSON ارسال شود.",
            )

    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # ۱. استخراج متن و hash
        try:
            text = extract_text_from_file(temp_file_path, docType)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"خطا در استخراج متن فایل: {str(e)}",
            )

        content_hash = compute_content_hash(text)

        # ۲. بررسی نسخه جاری
        current = _find_current_document(db, appId, docName)

        if current is not None and current.content_hash == content_hash:
            raise HTTPException(
                status_code=400,
                detail="محتوای سند تغییر نکرده است. نسخه جدیدی ایجاد نشد.",
            )

        if current is not None:
            logical_doc_id = current.logical_doc_id
            new_version = current.version + 1
            previous_version_id = current.id
            _demote_current(db, current)
            mark_version_not_current(str(logical_doc_id), current.version)
        else:
            logical_doc_id = uuid.uuid4()
            new_version = 1
            previous_version_id = None

        doc_id = uuid.uuid4()

        # ۳. ingest به Qdrant
        try:
            chunks_count = process_and_index_document(
                file_path=temp_file_path,
                doc_id=str(doc_id),
                logical_doc_id=str(logical_doc_id),
                version=new_version,
                content_hash=content_hash,
                is_current=True,
                doc_name=docName,
                doc_type=docType,
                department=department,
                app_id=appId,
                metadata=parsed_metadata,
            )
        except Exception as e:
            db.rollback()
            # جبران Qdrant: حذف chunkهای نیمه‌کاره (best-effort)
            try:
                delete_document_vectors_by_doc_id(str(doc_id))
            except Exception as exc:
                logger.warning(
                    "Qdrant compensation (file ingest) failed for doc_id=%s: %s",
                    doc_id,
                    exc,
                )
            raise HTTPException(
                status_code=500,
                detail=f"خطا در پردازش فایل: {str(e)}",
            )

        # ۴. ثبت ردیف جدید
        new_doc = Document(
            id=doc_id,
            logical_doc_id=logical_doc_id,
            version=new_version,
            is_current=True,
            previous_version_id=previous_version_id,
            content_hash=content_hash,
            doc_name=docName,
            doc_type=docType.lower(),
            department=department,
            app_id=appId,
            meta_info=parsed_metadata,
            chunks_count=chunks_count,
        )
        db.add(new_doc)
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            # جبران Qdrant: حذف chunkهای نوشته‌شده چون Postgres شکست خورد
            try:
                delete_document_vectors_by_doc_id(str(doc_id))
            except Exception as exc:
                logger.warning(
                    "Qdrant compensation (commit) failed for doc_id=%s: %s",
                    doc_id,
                    exc,
                )
            raise HTTPException(
                status_code=500,
                detail=f"خطا در ثبت سند: {str(e)}",
            )

        return {
            "status": "success",
            "documentId": str(doc_id),
            "logicalDocId": str(logical_doc_id),
            "version": new_version,
            "isCurrent": True,
            "docName": docName,
            "chunksCreated": chunks_count,
        }

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/create-text", status_code=status.HTTP_200_OK)
def create_text_documents_replace(
    request: TextDocumentsRequest,
    db: Session = Depends(get_db),
):
    """
    جایگزینی کامل اسناد متنی یک app_id.

    رفتار:
    - همه ردیف‌های جدول documents با این app_id حذف می‌شوند.
    - همه وکتورهای همان app_id از Qdrant حذف می‌شوند.
    - لیست اسناد جدید از صفر با version=1 ذخیره می‌شوند.
    - در صورت خطا، تلاش می‌شود وکتورهای نوشته‌شده در این درخواست پاک شوند
      و تراکنش Postgres rollback شود.

    تفاوت با create-text:
    - create-text نسخه‌بندی می‌کند و نسخه قبلی را demote می‌کند.
    - create-text2 همه چیز را حذف و از نو می‌سازد.
    """
    if not request.app_id.strip():
        raise HTTPException(status_code=400, detail="app_id الزامی است.")
    if not request.documents:
        raise HTTPException(
            status_code=400, detail="حداقل یک سند متنی باید ارسال شود."
        )

    app_id = request.app_id.strip()
    created: List[CreatedDocumentItem] = []

    # لیست جبران برای پاک‌سازی Qdrant در صورت خطا
    written_vectors: List[tuple] = []

    # شمارنده حذف‌شده‌ها برای گزارش
    deleted_docs_count = 0

    try:
        # ---------- مرحله ۱: پاک‌سازی کامل app_id ----------
        # ۱.۱ حذف از Qdrant
        try:
            delete_document_vectors_by_app_id(app_id)
        except Exception as exc:
            logger.warning(
                "Failed to delete Qdrant vectors for app_id=%s: %s",
                app_id,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"خطا در حذف وکتورهای قبلی از Qdrant: {str(exc)}",
            )

        # ۱.۲ حذف از Postgres
        existing_docs = (
            db.query(Document).filter(Document.app_id == app_id).all()
        )
        deleted_docs_count = len(existing_docs)
        for d in existing_docs:
            db.delete(d)
        # flush تا حذف‌ها در کوئری‌های بعدی همین تراکنش دیده شوند
        db.flush()

        # ---------- مرحله ۲: درج اسناد جدید با version=1 ----------
        for item in request.documents:
            text = item.doc or ""
            content_hash = compute_content_hash(text)

            logical_doc_id = uuid.uuid4()
            new_version = 1
            doc_id = uuid.uuid4()

            meta = dict(item.metadata or {})
            if request.version:
                meta["version"] = request.version
            meta["source_type"] = "text"

            chunks_count = process_and_index_text_document(
                doc_text=text,
                doc_id=str(doc_id),
                logical_doc_id=str(logical_doc_id),
                version=new_version,
                content_hash=content_hash,
                is_current=True,
                doc_name=item.doc_name,
                app_id=app_id,
                metadata=meta,
            )

            if chunks_count > 0:
                written_vectors.append(
                    (str(doc_id), str(logical_doc_id), new_version)
                )

            new_doc = Document(
                id=doc_id,
                logical_doc_id=logical_doc_id,
                version=new_version,
                is_current=True,
                previous_version_id=None,
                content_hash=content_hash,
                doc_name=item.doc_name,
                doc_type="text",
                department="",
                app_id=app_id,
                meta_info=meta,
                chunks_count=chunks_count,
            )
            db.add(new_doc)
            db.flush()

            created.append(
                _created_item(
                    doc_id=doc_id,
                    logical_doc_id=logical_doc_id,
                    version=new_version,
                    doc_name=item.doc_name,
                    chunks_count=chunks_count,
                )
            )

        db.commit()
        return {
            "status": "success",
            "appId": app_id,
            "deletedDocumentsCount": deleted_docs_count,
            "documents": [c.model_dump() for c in created],
        }

    except HTTPException:
        db.rollback()
        _compensate_qdrant(written_vectors)
        raise
    except Exception as e:
        db.rollback()
        _compensate_qdrant(written_vectors)
        raise HTTPException(
            status_code=500, detail=f"خطا در جایگزینی اسناد: {str(e)}"
        )


@router.post("/create-text2", status_code=status.HTTP_200_OK)
def create_text_documents(
    request: TextDocumentsRequest,
    db: Session = Depends(get_db),
):
    if not request.app_id.strip():
        raise HTTPException(status_code=400, detail="app_id الزامی است.")
    if not request.documents:
        raise HTTPException(
            status_code=400, detail="حداقل یک سند متنی باید ارسال شود."
        )

    app_id = request.app_id.strip()
    created: List[CreatedDocumentItem] = []

    # لیست جبران برای پاک‌سازی Qdrant در صورت خطا
    # هر آیتم: (doc_id, logical_doc_id, version)
    written_vectors: List[tuple] = []

    try:
        for item in request.documents:
            text = item.doc or ""
            content_hash = compute_content_hash(text)

            current = _find_current_document(db, app_id, item.doc_name)
            if current is not None and current.content_hash == content_hash:
                created.append(
                    CreatedDocumentItem(
                        logicalDocId=str(current.logical_doc_id),
                        version=current.version,
                        docName=item.doc_name,
                        chunksCreated=0,
                        status="skipped",
                        reason="content unchanged",
                    )
                )
                continue

            if current is not None:
                logical_doc_id = current.logical_doc_id
                new_version = current.version + 1
                previous_version_id = current.id
                _demote_current(db, current)
                mark_version_not_current(str(logical_doc_id), current.version)
            else:
                logical_doc_id = uuid.uuid4()
                new_version = 1
                previous_version_id = None

            doc_id = uuid.uuid4()

            meta = dict(item.metadata or {})
            if request.version:
                meta["version"] = request.version
            meta["source_type"] = "text"

            chunks_count = process_and_index_text_document(
                doc_text=text,
                doc_id=str(doc_id),
                logical_doc_id=str(logical_doc_id),
                version=new_version,
                content_hash=content_hash,
                is_current=True,
                doc_name=item.doc_name,
                app_id=app_id,
                metadata=meta,
            )

            # ثبت برای جبران (فقط اگر واقعاً چیزی نوشته شد)
            if chunks_count > 0:
                written_vectors.append(
                    (str(doc_id), str(logical_doc_id), new_version)
                )

            new_doc = Document(
                id=doc_id,
                logical_doc_id=logical_doc_id,
                version=new_version,
                is_current=True,
                previous_version_id=previous_version_id,
                content_hash=content_hash,
                doc_name=item.doc_name,
                doc_type="text",
                department="",
                app_id=app_id,
                meta_info=meta,
                chunks_count=chunks_count,
            )
            db.add(new_doc)
            # flush تا _find_current_document در تکرار بعدی این ردیف را ببیند
            # (مشکل چند سند هم‌نام در یک درخواست)
            db.flush()

            created.append(
                _created_item(
                    doc_id=doc_id,
                    logical_doc_id=logical_doc_id,
                    version=new_version,
                    doc_name=item.doc_name,
                    chunks_count=chunks_count,
                )
            )

        db.commit()
        return {
            "status": "success",
            "appId": app_id,
            "documents": [c.model_dump() for c in created],
        }

    except HTTPException:
        db.rollback()
        _compensate_qdrant(written_vectors)
        raise
    except Exception as e:
        db.rollback()
        _compensate_qdrant(written_vectors)
        raise HTTPException(
            status_code=500, detail=f"خطا در ذخیره متن: {str(e)}"
        )


@router.get(
    "/by-logical/{logical_doc_id}/versions",
    response_model=List[DocumentVersionItem],
)
def list_document_versions(
    logical_doc_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    versions = (
        db.query(Document)
        .filter(Document.logical_doc_id == logical_doc_id)
        .order_by(Document.version.desc())
        .all()
    )
    if not versions:
        raise HTTPException(status_code=404, detail="سندی با این شناسه یافت نشد.")

    return [
        DocumentVersionItem(
            id=v.id,
            version=v.version,
            isCurrent=v.is_current,
            contentHash=v.content_hash,
            chunksCount=v.chunks_count,
            createdAt=v.created_at,
            previousVersionId=v.previous_version_id,
        )
        for v in versions
    ]


@router.delete("/by-logical/{logical_doc_id}")
def delete_documents_by_logical_id(
    logical_doc_id: uuid.UUID,
    allVersions: bool = Query(False),
    db: Session = Depends(get_db),
):
    """
    حذف نسخه‌های یک سند منطقی.
    - allVersions=False → فقط نسخه جاری. اگر نسخه قبلی باشد، به current تبدیل می‌شود.
    - allVersions=True  → همه نسخه‌ها از Postgres و Qdrant.
    """
    versions = (
        db.query(Document)
        .filter(Document.logical_doc_id == logical_doc_id)
        .order_by(Document.version.desc())
        .all()
    )
    if not versions:
        raise HTTPException(status_code=404, detail="سندی با این شناسه یافت نشد.")

    try:
        if allVersions:
            delete_document_vectors_by_logical_id(str(logical_doc_id))
            for v in versions:
                db.delete(v)
            db.commit()
            return {
                "status": "success",
                "logicalDocId": str(logical_doc_id),
                "deletedCount": len(versions),
                "mode": "all",
            }

        # فقط نسخه جاری
        current = next((v for v in versions if v.is_current), None)
        if current is None:
            raise HTTPException(
                status_code=400,
                detail="نسخه جاری‌ای برای این سند وجود ندارد.",
            )

        delete_document_vectors_by_doc_id(str(current.id))
        db.delete(current)

        # auto-revert: نسخه قبلی جاری شود
        previous = next(
            (v for v in versions if v.version == current.version - 1), None
        )
        if previous is not None:
            previous.is_current = True
            # همگام‌سازی Qdrant: این نسخه دوباره current می‌شود
            try:
                mark_version_current(str(logical_doc_id), previous.version)
            except Exception as exc:
                logger.warning(
                    "Failed to mark version current in Qdrant "
                    "(logical=%s, version=%s): %s",
                    logical_doc_id,
                    previous.version,
                    exc,
                )
            new_current_id = previous.id
        else:
            new_current_id = None

        db.commit()
        return {
            "status": "success",
            "logicalDocId": str(logical_doc_id),
            "deletedCount": 1,
            "mode": "current-only",
            "newCurrentId": str(new_current_id) if new_current_id else None,
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"خطا در حذف سند: {str(e)}"
        )


@router.delete("/{document_id}")
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    حذف یک نسخه خاص با document_id.
    اگر این نسخه جاری بود و نسخه قبلی وجود داشت، نسخه قبلی auto-current می‌شود.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="سند مورد نظر یافت نشد.")

    try:
        delete_document_vectors_by_doc_id(str(document_id))

        was_current = doc.is_current
        logical_id = doc.logical_doc_id
        version = doc.version

        db.delete(doc)

        new_current_id = None
        if was_current:
            previous = (
                db.query(Document)
                .filter(
                    Document.logical_doc_id == logical_id,
                    Document.version == version - 1,
                )
                .first()
            )
            if previous is not None:
                previous.is_current = True
                # همگام‌سازی Qdrant
                try:
                    mark_version_current(str(logical_id), previous.version)
                except Exception as exc:
                    logger.warning(
                        "Failed to mark version current in Qdrant "
                        "(logical=%s, version=%s): %s",
                        logical_id,
                        previous.version,
                        exc,
                    )
                new_current_id = previous.id

        db.commit()
        return {
            "status": "success",
            "message": f"سند {doc.doc_name} نسخه {version} حذف شد.",
            "newCurrentId": str(new_current_id) if new_current_id else None,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"خطا در حذف سند: {str(e)}")


@router.delete("/by-app/{app_id}")
def delete_documents_by_app(app_id: str, db: Session = Depends(get_db)):
    """
    حذف همه اسناد (همه نسخه‌ها) یک app_id.
    """
    docs = db.query(Document).filter(Document.app_id == app_id).all()
    if not docs:
        raise HTTPException(
            status_code=404, detail="هیچ سندی با این appId یافت نشد."
        )

    try:
        for doc in docs:
            delete_document_vectors_by_doc_id(str(doc.id))
            db.delete(doc)
        db.commit()
        return {
            "status": "success",
            "message": f"{len(docs)} سند با appId '{app_id}' با موفقیت حذف شد.",
            "deletedCount": len(docs),
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"خطا در حذف اسناد: {str(e)}"
        )


    
@router.get("/app-ids", response_model=List[str])
def list_app_ids(
    db: Session = Depends(get_db),
):
    """
    دریافت لیست یکتای app_id ها از جدول documents
    """
    app_ids = (
        db.query(Document.app_id)
        .filter(
            Document.app_id.isnot(None),
            Document.app_id != "",
        )
        .distinct()
        .order_by(Document.app_id.asc())
        .all()
    )

    return [row[0] for row in app_ids]
