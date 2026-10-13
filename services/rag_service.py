
# services/rag_service.py
from typing import List, Tuple, Dict, Any
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

from config import settings

def _get_vector_store() -> QdrantVectorStore:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    embeddings = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_url)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
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
از حدس و گمان بپرهیز و در جواب به مستندات وکتور شده ارجاع کن. 

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
    chat_history: List[Tuple[str, str]] = None,
    k_docs: int = 4
) -> Tuple[str, List[Dict[str, Any]]]:
    vector_store = _get_vector_store()
    
        
    retriever = vector_store.as_retriever(search_kwargs={"k": k_docs})
    docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    sources = [
        {"doc_name": d.metadata.get("doc_name"), "doc_id": d.metadata.get("doc_id")}
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
        ("human", "{question}")
    ])

    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_url, temperature=0.2)
    chain = prompt_template | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context_text,
        "chat_history": history_messages,
        "question": question
    })

    return answer, sources









