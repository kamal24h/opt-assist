# api/v1/documents.py
import hashlib
import json
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
)
from schemas.document import DocumentOut, DocumentVersionItem

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
        db.commit()

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
    created = []

    try:
        for item in request.documents:
            text = item.doc or ""
            content_hash = compute_content_hash(text)

            current = _find_current_document(db, app_id, item.doc_name)
            if current is not None and current.content_hash == content_hash:
                created.append({
                    "docName": item.doc_name,
                    "status": "skipped",
                    "reason": "content unchanged",
                    "logicalDocId": str(current.logical_doc_id),
                    "version": current.version,
                })
                continue

            if current is not None:
                logical_doc_id = current.logical_doc_id
                new_version = current.version + 1
                previous_version_id = current.id
                _demote_current(db, current)
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
            created.append({
                "documentId": str(doc_id),
                "logicalDocId": str(logical_doc_id),
                "version": new_version,
                "docName": item.doc_name,
                "chunksCreated": chunks_count,
                "status": "created",
            })

        db.commit()
        return {"status": "success", "appId": app_id, "documents": created}

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"خطا در ذخیره متن: {str(e)}"
        )


@router.get("/by-logical/{logical_doc_id}/versions", response_model=List[DocumentVersionItem])
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
        previous = next((v for v in versions if v.version == current.version - 1), None)
        if previous is not None:
            previous.is_current = True
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