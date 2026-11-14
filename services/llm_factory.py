# services/llm_factory.py
import logging
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)


def get_llm(
    temperature: Optional[float] = None,
    reasoning: bool = False,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    purpose: str = "chat",   # "chat" | "form_resolver" | "small_talk"
):
    """
    ساخت LLM بر اساس settings.llm_provider.
    - "ollama": ChatOllama (رفتار فعلی)
    - "external": ChatOpenAI سازگار با OpenAI API
    fallback اختیاری به Ollama در صورت خطا در لایه‌های بالاتر مدیریت می‌شود.
    """
    provider = (settings.llm_provider or "ollama").lower()

    if provider == "external":
        if not settings.external_llm_base_url or not settings.external_llm_model:
            logger.warning(
                "External LLM تنظیم نشده؛ بازگشت به Ollama."
            )
            provider = "ollama"
        else:
            try:
                return ChatOpenAI(
                    model=settings.external_llm_model,
                    base_url=settings.external_llm_base_url,
                    api_key=settings.external_llm_api_key or "sk-no-key",
                    temperature=(
                        temperature
                        if temperature is not None
                        else settings.llm_temperature
                    ),
                    timeout=settings.external_llm_timeout,
                    max_retries=settings.external_llm_max_retries,
                    streaming=False,
                )
            except Exception as exc:
                logger.error("External LLM init failed: %s", exc)
                if not settings.external_llm_fallback_to_ollama:
                    raise
                provider = "ollama"

    # پیش‌فرض: Ollama
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_url,
        temperature=(
            temperature if temperature is not None else settings.llm_temperature
        ),
        reasoning=reasoning,
        num_ctx=num_ctx or settings.llm_num_ctx,
        num_predict=num_predict,
        repeat_penalty=repeat_penalty,
    )