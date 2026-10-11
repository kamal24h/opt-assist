# services/assistant_service.py

import os
import uuid

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient

from redis_client import (
    get_redis_client,
    ConversationCache,
)


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://qdrant:6333",
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama:11434",
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "assistant_knowledge1",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qllama/bge-m3:q4_k_m",
)


class AssistantService:

    def __init__(self):

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )

        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )

    def get_rag_chain(self):

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 4
            }
        )

        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            temperature=0.2,
        )

        template = """
Answer the question based only on the following context.

If you don't know the answer, just say that you don't know.
Do not make up an answer.

Context:
{context}

Question:
{question}
"""

        prompt = ChatPromptTemplate.from_template(
            template
        )

        def format_docs(docs):

            return "\n\n".join(
                doc.page_content
                for doc in docs
            )

        return (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, prompt: str) -> str:

        chain = self.get_rag_chain()

        return chain.invoke(prompt)

    def ask_conversation(
        self,
        conversation_id: uuid.UUID,
        prompt: str,
    ) -> str:

        redis_client = get_redis_client()

        ConversationCache.append_message(
            redis_client,
            str(conversation_id),
            role="user",
            content=prompt,
        )

        try:

            answer = self.ask(prompt)

            ConversationCache.append_message(
                redis_client,
                str(conversation_id),
                role="assistant",
                content=answer,
            )

            return answer

        except Exception:
            raise