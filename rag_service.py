# rag_service.py
import os
from typing import List, Tuple
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "992b61d895665cd63484456f34950944dd181ea656ed5e2bf2e4268dae4e96d8")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "assistant_knowledge1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-m3:q4_k_m")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

SYSTEM_PROMPT = """
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

Context اسناد مرتبط:
{context}
"""

def generate_rag_response(
    question: str, 
    chat_history: List[Tuple[str, str]] = None,
    k_docs: int = 4
) -> Tuple[str, List[dict]]:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # 1. بازیابی اسناد مرتبط
    retriever = vector_store.as_retriever(search_kwargs={"k": k_docs})
    docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # متادیتای اسناد برای لاگ کردن در Message.meta_info
    sources_meta = [{"doc_name": d.metadata.get("doc_name"), "doc_id": d.metadata.get("doc_id")} for d in docs]

    # 2. آماده‌سازی تاریخچه پیام‌ها
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

    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.2)
    chain = prompt_template | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context_text,
        "chat_history": history_messages,
        "question": question
    })

    return answer, sources_meta