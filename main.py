# main.py
import os
import uuid
import shutil
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from typing import Optional, List, Dict, Any, Literal

# ماژول‌های دیتابیس و ریدیس
from database import Base, engine, get_db, Document, Conversation, Message
from redis_client import get_redis_client, ConversationCache
# ایمپورت تابع Ingest
from ingest_service import (
    process_and_index_document,
    process_and_index_text_document,
    delete_document_vectors
)

# ساخت جداول دیتابیس در صورت عدم وجود
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="Enterprise Assistant RAG API", lifespan=lifespan)

# تنظیمات پایه
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "992b61d895665cd63484456f34950944dd181ea656ed5e2bf2e4268dae4e96d8")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "assistant_knowledge1")
#LLM_MODEL = os.getenv("LLM_MODEL", "partai/dorna-llama3")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-m3:q4_k_m")

# ایجاد ارتباط با Qdrant و LLM
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

def get_rag_chain():
    """زنجیره را داینامیک می‌خواند تا پس از هربار Ingest کالکشن جدید بارگذاری شود"""
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.2)

#     template = """Answer the question based only on the following context.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

    template = """
    تو یک دستیار هوشمند فارسی‌زبان برای کاربران سازمانی هستی.

    سبک مکالمه:

    به فارسی طبیعی، روان، امروزی و حرفه‌ای پاسخ بده.

    لحن تو دوستانه و محترمانه باشد؛ نه بیش از حد رسمی و نه بیش از حد خودمانی.

    از جمله‌بندی ترجمه‌وار، عبارت‌های کلیشه‌ای و تکرار بی‌مورد سؤال کاربر پرهیز کن.

    پاسخ را مستقیم شروع کن و از مقدمه‌های غیرضروری دوری کن.

    برای سؤال ساده، پاسخ کوتاه و روشن بده.

    برای مسئله پیچیده، پاسخ را مرحله‌به‌مرحله و با ساختار مناسب ارائه کن.

    اگر اطلاعات کافی نداری، صریح بگو و در صورت نیاز یک سؤال مشخص بپرس.

    اطلاعات، اعداد و نتایج ابزارهای سازمانی را تغییر نده یا حدس نزن.

    در پاسخ‌های مبتنی بر اسناد یا داده‌های سازمانی، به اطلاعات بازیابی‌شده و نتایج واقعی ابزارها وفادار بمان.

    طبیعی بودن لحن نباید باعث تغییر معنای پاسخ یا حذف محدودیت‌ها و عدم‌قطعیت‌ها شود.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

class QueryRequest(BaseModel):
    prompt: str

# ----------------- Schemas -----------------
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

# --- Pydantic Schemas ---
class TextDocumentItem(BaseModel):
    doc_name: str
    metadata: Optional[Dict[str, Any]] = None
    doc_type: Literal["text"] = "text"
    doc: str

class TextDocumentsRequest(BaseModel):
    app_id: str
    version: Optional[str] = None
    documents: List[TextDocumentItem]

class BeginConversationRequest(BaseModel):
    appId: str
    tenantId: str
    userId: str

class QueryConversationRequest(BaseModel):
    conversationId: str
    prompt: str

class EndConversationRequest(BaseModel):
    conversationId: str

# ----------------- اندپوینت‌های مکالمه -----------------

@app.post("/beginConversation")
async def begin_conversation(req: BeginConversationRequest, db: Session = Depends(get_db)):
    try:
        conv_id = uuid.uuid4()

        # ۱. ذخیره در PostgreSQL
        db_conv = Conversation(
            id=conv_id,
            app_id=req.appId,
            tenant_id=req.tenantId,
            user_id=req.userId,
            status="active"
        )
        db.add(db_conv)
        db.commit()

        # ۲. ذخیره متادیتا در Redis
        r = get_redis_client()
        ConversationCache.create_conversation(
            redis_client=r,
            conv_id=str(conv_id),
            app_id=req.appId,
            tenant_id=req.tenantId,
            user_id=req.userId
        )

        return {
            "conversationId": str(conv_id),
            "status": "created",
            "appId": req.appId,
            "tenantId": req.tenantId,
            "userId": req.userId
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to begin conversation: {str(e)}")

@app.post("/query")
async def query_assistant(request: QueryConversationRequest):
    """انجام پرسش با بررسی وضعیت در Redis و ثبت پیام در حافظه موقت"""
    r = get_redis_client()
    meta = ConversationCache.get_conversation_meta(r, request.conversationId)
    if not meta or meta.get("status") != "active":
        raise HTTPException(status_code=404, detail="مکالمه یافت نشد یا پایان یافته است.")

    try:
        # ۱. ثبت پیام کاربر در Redis
        ConversationCache.append_message(r, request.conversationId, role="user", content=request.prompt)

        # ۲. تولید پاسخ از RAG
        chain = get_rag_chain()
        answer = chain.invoke(request.prompt)

        # ۳. ثبت پاسخ دستیار در Redis
        ConversationCache.append_message(r, request.conversationId, role="assistant", content=answer)

        return {
            "conversationId": request.conversationId,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/endConversation")
async def end_conversation(req: EndConversationRequest, db: Session = Depends(get_db)):
    """انتقال کل پیام‌ها از ریدیس به Postgres و بستن وضعیت مکالمه"""
    r = get_redis_client()
    meta = ConversationCache.get_conversation_meta(r, req.conversationId)

    db_conv = db.query(Conversation).filter(Conversation.id == req.conversationId).first()
    if not db_conv:
        raise HTTPException(status_code=404, detail="مکالمه در پایگاه داده یافت نشد.")

    # دریافت پیام‌های نگه‌داری شده در Redis
    cached_messages = ConversationCache.get_messages(r, req.conversationId)

    try:
        # ذخیره کل تاریخچه پیام‌ها در جدول Messages
        for msg in cached_messages:
            db_msg = Message(
                conversation_id=db_conv.id,
                role=msg["role"],
                content=msg["content"]
            )
            db.add(db_msg)

        db_conv.status = "closed"
        db_conv.ended_at = datetime.utcnow()
        db.commit()

        # پاک کردن کش مکالمه از ریدیس
        ConversationCache.clear_conversation(r, req.conversationId)

        return {
            "status": "success",
            "message": "مکالمه پایان یافت و داده‌ها در پایگاه داده اصلی آرشیو شدند.",
            "total_messages_saved": len(cached_messages)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

# ----------------- اندپوینت‌های سیستم و Ingest -----------------

@app.get("/health")
async def health():
    return {"status": "healthy", "llm_model": LLM_MODEL};

# ----------------- Document Management APIs -----------------

@app.get("/api/v1/documents/index", response_model=List[DocumentOut])
async def list_documents(
    appId: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """لیست اسناد موجود با قابلیت فیلتر بر اساس appId و department"""
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

@app.post("/api/v1/documents/create")
async def create_document(
    file: UploadFile = File(...),
    docName: str = Form(...),
    docType: str = Form(...),
    department: str = Form(...),
    appId: str = Form(...),
    metadata: Optional[str] = Form(None), # رشته JSON ورودی
    db: Session = Depends(get_db)
):
    """ایجاد سند در Postgres، وکتورسازی در Qdrant و اتصال متادیتا"""
    allowed_types = ["pdf", "docx"]
    if docType.lower() not in allowed_types:
        raise HTTPException(status_code=400, detail=f"docType مجاز نیست. فقط {allowed_types}")

    parsed_metadata = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except Exception:
            raise HTTPException(status_code=400, detail="فرمت فیلد metadata باید یک JSON معتبر باشد.")

    # ذخیره فایل در پوشه موقت برای خواندن توسط لودر
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_id = uuid.uuid4()

    try:
        # 1. خردسازی و وکتورسازی در Qdrant
        chunks_count = process_and_index_document(
            file_path=temp_file_path,
            doc_id=str(doc_id),
            doc_name=docName,
            doc_type=docType,
            department=department,
            app_id=appId,
            metadata=parsed_metadata
        )

        # 2. ذخیره متادیتا در PostgreSQL
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
        raise HTTPException(status_code=500, detail=f"خطا در پردازش و ذخیره سند: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ----------------- Text Document Ingestion API -----------------

def _find_existing_document(db: Session, doc_name: str, app_id: str):
    """پیدا کردن آخرین نسخه ثبت‌شده یک سند بر اساس doc_name و app_id."""
    return (
        db.query(Document)
        .filter(
            Document.doc_name == doc_name,
            Document.app_id == app_id
        )
        .order_by(Document.created_at.desc())
        .first()
    )

@app.post("/api/v1/documents/create-text")
async def create_text_documents(
    request: TextDocumentsRequest,
    db: Session = Depends(get_db)
):
    """ایجاد/به‌روزرسانی چند سند متنی بدون فایل پیوست با app_id و version مشترک.

    کلید منطقی سند: doc_name + app_id
    در صورت وجود سند قبلی:
      1. vectorهای نسخه قبلی از Qdrant حذف می‌شوند.
      2. رکورد قبلی از PostgreSQL حذف می‌شود.
      3. نسخه جدید index و ذخیره می‌شود.
    """
    if not request.app_id or not request.app_id.strip():
        raise HTTPException(status_code=400, detail="app_id نمی‌تواند خالی باشد.")

    if not request.documents:
        raise HTTPException(status_code=400, detail="لیست documents نمی‌تواند خالی باشد.")

    app_id = request.app_id.strip()
    version = request.version.strip() if request.version else None

    created = []
    indexed_doc_ids = []
    replaced_docs = []

    try:
        # جلوگیری از ارسال دو سند با نام یکسان در یک batch
        batch_doc_names = set()

        for item in request.documents:
            doc_name = item.doc_name.strip()
            if not doc_name:
                raise HTTPException(status_code=400, detail="doc_name نمی‌تواند خالی باشد.")
            if not item.doc or not item.doc.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"متن سند '{doc_name}' نمی‌تواند خالی باشد."
                )

            if doc_name in batch_doc_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"سند تکراری در batch: doc_name='{doc_name}' برای app_id='{app_id}'"
                )
            batch_doc_names.add(doc_name)

            # ۱. بررسی وجود نسخه قبلی با همین doc_name و app_id
            existing_doc = _find_existing_document(
                db,
                doc_name=doc_name,
                app_id=app_id
            )

            if existing_doc:
                replaced_docs.append(existing_doc)

            # ۲. شناسه یکتا برای نسخه جدید
            doc_id = uuid.uuid4()

            parsed_metadata = dict(item.metadata or {})
            if version is not None:
                parsed_metadata["version"] = version

            parsed_metadata["source_type"] = "text"

            # ۳. Chunk + Embedding + ذخیره در Qdrant
            chunks_count = process_and_index_text_document(
                doc_text=item.doc,
                doc_id=str(doc_id),
                doc_name=doc_name,
                app_id=app_id,
                metadata=parsed_metadata
            )

            if chunks_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"سند '{doc_name}' قابل ایندکس نیست."
                )

            indexed_doc_ids.append(str(doc_id))

            # ۴. ثبت رکورد در PostgreSQL
            new_doc = Document(
                id=doc_id,
                doc_name=doc_name,
                doc_type="text",
                department="",
                app_id=app_id,
                meta_info=parsed_metadata,
                chunks_count=chunks_count
            )
            db.add(new_doc)

            created.append({
                "documentId": str(doc_id),
                "docName": doc_name,
                "appId": app_id,
                "docType": "text",
                "version": version,
                "chunksCreated": chunks_count,
                "operation": "updated" if existing_doc else "created"
            })

        # ۵. حذف نسخه‌های قبلی از PostgreSQL و Commit نسخه جدید
        for old_doc in replaced_docs:
            db.delete(old_doc)

        db.commit()

        # ۶. پاک‌سازی وکتورهای نسخه‌های قبلی از Qdrant
        cleanup_errors = []
        for old_doc in replaced_docs:
            try:
                delete_document_vectors(doc_id=str(old_doc.id))
            except Exception as cleanup_error:
                cleanup_errors.append(str(cleanup_error))

        response = {
            "status": "success",
            "appId": app_id,
            "version": version,
            "documentsCount": len(created),
            "documents": created
        }
        if cleanup_errors:
            response["warning"] = "برخی vectorهای نسخه قبلی در Qdrant حذف نشدند و باید cleanup شوند."
            response["cleanupErrors"] = cleanup_errors
        return response

    except HTTPException:
        db.rollback()
        for doc_id in indexed_doc_ids:
            try:
                delete_document_vectors(doc_id)
            except Exception:
                pass
        raise

    except Exception as e:
        db.rollback()
        for doc_id in indexed_doc_ids:
            try:
                delete_document_vectors(doc_id)
            except Exception:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"خطا در پردازش و ذخیره اسناد متنی: {str(e)}"
        )

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """حذف سند از پایگاه داده PostgreSQL و وکتورهای متناظر آن از Qdrant"""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="سند مورد نظر یافت نشد.")

    try:
        # 1. حذف تکه‌ها از Qdrant
        delete_document_vectors(doc_id=str(document_id))

        # 2. حذف از دیتابیس رابطه‌ای
        db.delete(doc)
        db.commit()

        return {"status": "success", "message": f"سند {doc.doc_name} با موفقیت حذف شد."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"خطا در حذف سند: {str(e)}")

