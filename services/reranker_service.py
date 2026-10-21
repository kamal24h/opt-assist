# services/reranker_service.py
import logging
from typing import List
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

# یک نمونه سراسری از Reranker برای جلوگیری از بارگذاری مکرر مدل
_reranker = None

def get_reranker():
    """بارگذاری و کش کردن مدل Reranker (فقط یک بار)."""
    global _reranker
    if _reranker is None:
        try:
            logger.info("Loading reranker model: BAAI/bge-reranker-v2-m3")
            # مدل چندزبانه، مناسب برای فارسی
            cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
            _reranker = CrossEncoderReranker(model=cross_encoder, top_n=5) # انتخاب ۵ سند برتر
            logger.info("Reranker model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load reranker model: {exc}")
            _reranker = None
    return _reranker

def rerank_documents(query: str, documents: List[Document]) -> List[Document]:
    """
    رتبه‌بندی مجدد لیستی از اسناد بر اساس ارتباط با query.
    اگر مدل بارگذاری نشده باشد یا خطایی رخ دهد، لیست اصلی را برمی‌گرداند.
    """
    reranker = get_reranker()
    if reranker is None or not documents:
        return documents

    try:
        # CrossEncoderReranker.compress_documents(query, documents) را فراخوانی می‌کند
        reranked_docs = reranker.compress_documents(documents, query)
        logger.info(f"Reranked {len(documents)} docs down to {len(reranked_docs)}")
        return reranked_docs
    except Exception as exc:
        logger.error(f"Reranking failed: {exc}")
        return documents