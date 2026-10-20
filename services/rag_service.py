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
    """
    تولید پاسخ RAG محدود به اسناد app_id مشخص و فقط نسخه‌های جاری.
    """

    # نرمال‌سازی: strip و lower برای اطمینان از تطبیق
    app_id = (app_id or "").strip()
    if not app_id:
        raise ValueError("app_id برای جست‌وجوی RAG الزامی است.")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )

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

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k_docs, "filter": query_filter}
    )
    docs = retriever.invoke(question)

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "RAG retrieved %d docs for app_id=%s, question=%r",
        len(docs),
        app_id,
        question[:80],
    )

    context_text = "\n\n".join([d.page_content for d in docs])

    sources = [
        {
            "doc_name": d.metadata.get("doc_name"),
            "doc_id": d.metadata.get("doc_id"),
            "logical_doc_id": d.metadata.get("logical_doc_id"),
            "doc_version": d.metadata.get("doc_version"),
        }
        for d in docs
    ]

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