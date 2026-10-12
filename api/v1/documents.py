# api/v1/documents.py
import uuid
import json
import shutil
import os
from typing import List, Optional, Dict, Any, Literal
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, Document
from ingest_service import (
    process_and_index_document,
    process_and_index_text_document,
    delete_document_vectors
)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# Schemas
class DocumentOut(BaseModel):
    id: uuid.UUID
    docName: str
    docType: str
    department: str
    appId: str
    metadata: Optional[dict] = None
    chunksCount: int

    class Config:
        from_attributes = True

class TextDocumentItem(BaseModel):
    doc_name: str
    metadata: Optional[Dict[str, Any]] = None
    doc_type: Literal["text"] = "text"
    doc: str

class TextDocumentsRequest(BaseModel):
    app_id: str
    version: Optional[str] = None
    documents: List[TextDocumentItem]


@router.get("", response_model=List[DocumentOut])
def list_documents(
    appId: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(Document)
    if appId:
        query = query.filter(Document.app_id == appId)
    if department:
        query = query.filter(Document.department == department)

    docs = query.order_by(Document.created_at.desc()).all()
    return [
        DocumentOut(
            id=d.id,
            docName=d.doc_name,
            docType=d.doc_type,
            department=d.department,
            appId=d.app_id,
            metadata=d.meta_info,
            chunksCount=d.chunks_count
        )
        for d in docs
    ]


@router.post("/create")
def create_document_file(
    file: UploadFile = File(...),
    docName: str = Form(...),
    docType: str = Form(...),
    department: str = Form(...),
    appId: str = Form(...),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    allowed_types = ["pdf", "docx"]
    if docType.lower() not in allowed_types:
        raise HTTPException(status_code=400, detail=f"نوع فایل غیرمجاز است. تنها فرمت‌های {allowed_types} پذیرفته می‌شوند.")

    parsed_metadata = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except Exception:
            raise HTTPException(status_code=400, detail="متادیتا باید با ساختار معتبر JSON ارسال شود.")

    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_id = uuid.uuid4()
    try:
        chunks_count = process_and_index_document(
            file_path=temp_file_path,
            doc_id=str(doc_id),
            doc_name=docName,
            doc_type=docType,
            department=department,
            app_id=appId,
            metadata=parsed_metadata
        )

        new_doc = Document(
            id=doc_id,
            doc_name=docName,
            doc_type=docType.lower(),
            department=department,
            app_id=appId,
            meta_info=parsed_metadata,
            chunks_count=chunks_count
        )
        db.add(new_doc)
        db.commit()

        return {
            "status": "success",
            "documentId": str(doc_id),
            "docName": docName,
            "chunksCreated": chunks_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"خطا در پردازش فایل: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/create-text", status_code=status.HTTP_200_OK)
def create_text_documents(
    request: TextDocumentsRequest,
    db: Session = Depends(get_db)
):
    if not request.app_id.strip():
        raise HTTPException(status_code=400, detail="app_id الزامی است.")
    if not request.documents:
        raise HTTPException(status_code=400, detail="حداقل یک سند متنی باید ارسال شود.")

    app_id = request.app_id.strip()
    created = []

    try:
        for item in request.documents:
            doc_id = uuid.uuid4()
            meta = dict(item.metadata or {})
            if request.version:
                meta["version"] = request.version
            meta["source_type"] = "text"

            chunks_count = process_and_index_text_document(
                doc_text=item.doc,
                doc_id=str(doc_id),
                doc_name=item.doc_name,
                app_id=app_id,
                metadata=meta
            )

            new_doc = Document(
                id=doc_id,
                doc_name=item.doc_name,
                doc_type="text",
                department="",
                app_id=app_id,
                meta_info=meta,
                chunks_count=chunks_count
            )
            db.add(new_doc)
            created.append({
                "documentId": str(doc_id),
                "docName": item.doc_name,
                "chunksCreated": chunks_count
            })

        db.commit()
        return {"status": "success", "appId": app_id, "documents": created}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"خطا در ذخیره متن: {str(e)}")


@router.delete("/{document_id}")
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="سند مورد نظر یافت نشد.")

    try:
        delete_document_vectors(doc_id=str(document_id))
        db.delete(doc)
        db.commit()
        return {"status": "success", "message": f"سند {doc.doc_name} با موفقیت حذف شد."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"خطا در حذف سند: {str(e)}")


@router.delete("/by-app/{app_id}")
def delete_documents_by_app(app_id: str, db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.app_id == app_id).all()
    if not docs:
        raise HTTPException(
            status_code=404,
            detail="هیچ سندی با این appId یافت نشد."
        )

    try:
        for doc in docs:
            delete_document_vectors(doc_id=str(doc.id))
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
            status_code=500,
            detail=f"خطا در حذف اسناد: {str(e)}"
        )
    