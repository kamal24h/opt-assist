from langchain_ollama import OllamaEmbeddings

from core.config import settings


class EmbeddingService:

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        result: list[list[float]] = []

        batch_size = settings.embedding_batch_size

        for start in range(0, len(texts), batch_size):

            batch = texts[
                start:start + batch_size
            ]

            vectors = await self.embeddings.aembed_documents(
                batch
            )

            result.extend(vectors)

        return result

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:

        return await self.embeddings.aembed_query(text)