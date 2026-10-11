from pydantic import BaseModel, Field


class CreateTextDocumentRequest(BaseModel):
    document_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Stable unique identifier of the document",
    )

    title: str | None = Field(
        default=None,
        max_length=500,
    )

    text: str = Field(
        ...,
        min_length=1,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class CreateTextDocumentResponse(BaseModel):
    document_id: str
    version: int
    status: str

    content_hash: str

    chunk_count: int

    previous_version: int | None = None

    message: str