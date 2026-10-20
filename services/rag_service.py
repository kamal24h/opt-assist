# services/rag_service.py
from typing import List, Tuple, Dict, Any, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config import settings

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
)
embeddings = OllamaEmbeddings(
    model=settings.embedding_model,
    base_url=settings.ollama_url,
)

SYSTEM_PROMPT = """
تو یک دستیار هوشمند فارسی‌زبان برای کاربران سازمانی هستی.

فرمت پاسخ:
لطفاً پاسخ را بدون هیچ فرمت Markdown بده.
از ** ، ## ، ```، لیست با - یا * استفاده نکن.
فقط متن ساده با Enter بین خطوط.

سبک مکالمه:
به فارسی طبیعی، روان، امروزی و حرفه‌ای پاسخ بده.
لحن تو دوستانه و محترمانه باشد؛ نه بیش از حد رسمی و نه بیش از حد خودمانی.
از جمله‌بندی ترجمه‌وار، عبارت‌های کلیشه‌ای و تکرار بی‌مورد سؤال کاربر پرهیز کن.
پاسخ را مستقیم شروع کن و از مقدمه‌های غیرضروری دوری کن.
برای سؤال ساده، پاسخ کوتاه و روشن بده.
برای مسئله پیچیده، پاسخ را مرحله‌به‌مرحله و با ساختار مناسب ارائه کن.
اگر اطلاعات کافی نداری، صریح بگو و در صورت نیاز یک سؤال مشخص بپرس.
اطلاعات، اعداد و نتایج ابزارهای سازمانی را تغییر نده یا حدس نزن.
در پاسخ‌های مبتنی بر اسناد یا داده‌های سازمانی، به اطلاعات بازیابی‌شده وفادار بمان.
از حدس و گمان بپرهیز و در جواب به مستندات که در اختیار داری ارجاع کن. 

Context اسناد مرتبط:
{context}
"""

# SYSTEM_PROMPT = """
# تو یک دستیار هوشمند فارسی‌زبان برای سیستم‌های سازمانی و مستندات محصول هستی.
# به فارسی طبیعی، روان، دقیق و بدون مقدمه‌های اضافی پاسخ بده.
# تنها با استناد به محتوای بازیابی‌شده پاسخ بده. اگر پاسخ در اسناد نیست صریحاً بگو اطلاعات کافی نداری.

# Context:
# {context}
# """

def generate_rag_response(
    question: str,
    app_id: str,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    k_docs: int = 4,
) -> Tuple[str, List[Dict[str, Any]], Optional[int]]:
    """تولید پاسخ RAG با محدودسازی نتایج به app_id و نسخه‌های جاری.

    برای هر منبع، اطلاعات بازیابی‌شده از Qdrant به‌همراه متن chunk ثبت می‌شود
    تا در meta_info.sources پیام دستیار قابل ردیابی باشد.
    """
    import hashlib
    import logging

    logger = logging.getLogger(__name__)
    app_id = (app_id or "").strip()
    if not app_id:
        raise ValueError("app_id برای جست‌وجوی RAG الزامی است.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )

    # توجه: این مسیرها مطابق payload پیش‌فرض LangChain-Qdrant هستند.
    query_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.app_id",
                match=MatchValue(value=app_id),
            ),
            FieldCondition(
                key="metadata.is_current",
                match=MatchValue(value=True),
            ),
        ]
    )

    # برخلاف retriever.invoke، این متد score واقعی هر نتیجه را هم برمی‌گرداند.
    scored_docs = vector_store.similarity_search_with_score(
        question,
        k=k_docs,
        filter=query_filter,
    )

    sources: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    core_keys = {
        "doc_id", "doc_name", "logical_doc_id", "doc_version", "app_id",
        "chunk_id", "chunk_hash", "page", "start_index", "is_current",
        "content_hash", "department",
    }

    for rank, item in enumerate(scored_docs, start=1):
        doc, raw_score = item
        md = dict(doc.metadata or {})
        chunk_text = doc.page_content or ""
        chunk_hash = md.get("chunk_hash") or hashlib.sha256(
            chunk_text.encode("utf-8")
        ).hexdigest()
        # نسخه‌های مختلف LangChain ممکن است شناسه Qdrant را در Document.id
        # یا metadata برگردانند؛ در نبود آن، hash محتوایی شناسه جایگزین است.
        chunk_id = getattr(doc, "id", None) or md.get("chunk_id") or chunk_hash

        context_parts.append(chunk_text)
        extra_metadata = {key: value for key, value in md.items() if key not in core_keys}
        source: Dict[str, Any] = {
            "doc_id": md.get("doc_id"),
            "doc_name": md.get("doc_name"),
            "app_id": md.get("app_id", app_id),
            "chunk_id": str(chunk_id),
            "chunk_hash": str(chunk_hash),
            "chunk_text": chunk_text,
            "score": float(raw_score) if raw_score is not None else None,
            "retrieval_rank": rank,
            "metadata": extra_metadata,
        }
        if md.get("page") is not None:
            source["page"] = md["page"]
        if md.get("start_index") is not None:
            source["start_index"] = md["start_index"]
        # فیلدهای نسخه‌بندی برای تشخیص منبع/نسخه سند مفیدند.
        if md.get("logical_doc_id") is not None:
            source["logical_doc_id"] = md["logical_doc_id"]
        if md.get("doc_version") is not None:
            source["doc_version"] = md["doc_version"]
        sources.append(source)

    logger.info(
        "RAG retrieved %d docs for app_id=%s, question=%r",
        len(scored_docs), app_id, question[:80],
    )
    context_text = "\n\n".join(context_parts)

    history_messages = []
    if chat_history:
        for role, text in chat_history:
            if role == "user":
                history_messages.append(HumanMessage(content=text))
            elif role == "assistant":
                history_messages.append(AIMessage(content=text))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_url,
        temperature=0.2,
    )
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context_text,
        "chat_history": history_messages,
        "question": question,
    })

    from services.form_resolver import resolve_form_id
    form_id = resolve_form_id(question=question, answer=answer, use_llm_fallback=True)
    return answer, sources, form_id
