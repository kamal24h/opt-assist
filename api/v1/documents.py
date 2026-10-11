from fastapi import (APIRouter, Depends, HTTPException, status)

from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.schemas.document import (CreateTextDocumentRequest,
                                   CreateTextDocumentResponse)
from app.services.document_service import (DocumentService)
from app.services.embedding_service import (EmbeddingService)
from app.services.qdrant_document_service import (QdrantDocumentService)


router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


qdrant_client = AsyncQdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


qdrant_service = QdrantDocumentService(
    client=qdrant_client,
    collection_name=settings.qdrant_collection,
)


embedding_service = EmbeddingService()


document_service = DocumentService(
    qdrant_service=qdrant_service,
    embedding_service=embedding_service,
)


def get_document_service() -> DocumentService:

    return document_service


@router.post("/create-text",
    response_model=CreateTextDocumentResponse,
    status_code=status.HTTP_200_OK)

async def create_text_document(
    request: CreateTextDocumentRequest,
    service: DocumentService = Depends(
        get_document_service
    ),
):

    try:

        return await service.create_or_update_text(
            request
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process document."
            ),
        ) from exc