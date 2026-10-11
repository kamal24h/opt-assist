import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from schemas.document import (
    CreateTextDocumentRequest,
    CreateTextDocumentResponse,
)
from services.embedding_service import (
    EmbeddingService,
)
from services.qdrant_document_service import (
    QdrantDocumentService,
)


class DocumentService:

    def __init__(
        self,
        qdrant_service: QdrantDocumentService,
        embedding_service: EmbeddingService,
    ):

        self.qdrant = qdrant_service
        self.embedding = embedding_service

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,

            separators=[
                "\n\n",
                "\n",
                ". ",
                "؟ ",
                "! ",
                "؛ ",
                "، ",
                " ",
                "",
            ],
        )

    # ---------------------------------------------------------
    # Hash
    # ---------------------------------------------------------

    @staticmethod
    def calculate_hash(
        text: str,
    ) -> str:

        normalized = text.strip()

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    # ---------------------------------------------------------
    # Create / Update
    # ---------------------------------------------------------

    async def create_or_update_text(
        self,
        request: CreateTextDocumentRequest,
    ) -> CreateTextDocumentResponse:

        document_id = request.document_id.strip()

        text = request.text.strip()

        if not text:
            raise ValueError(
                "Document text cannot be empty."
            )

        content_hash = self.calculate_hash(
            text
        )

        # ---------------------------------------------
        # Read current manifest
        # ---------------------------------------------

        manifest = await self.qdrant.get_manifest(
            document_id
        )

        if manifest:

            current_hash = manifest.get(
                "content_hash"
            )

            current_version = manifest.get(
                "active_version"
            )

            # -----------------------------------------
            # Same content => NO-OP
            # -----------------------------------------

            if current_hash == content_hash:

                return CreateTextDocumentResponse(
                    document_id=document_id,
                    version=current_version,
                    previous_version=current_version,
                    status="unchanged",
                    content_hash=content_hash,
                    chunk_count=manifest.get(
                        "chunk_count",
                        0,
                    ),
                    message=(
                        "Document content is unchanged. "
                        "No new version was created."
                    ),
                )

            next_version = (
                current_version + 1
            )

        else:

            current_version = None
            next_version = 1

        # ---------------------------------------------
        # Chunk
        # ---------------------------------------------

        chunks = self.splitter.split_text(
            text
        )

        if not chunks:
            raise ValueError(
                "Document produced no chunks."
            )

        # ---------------------------------------------
        # Embedding
        # ---------------------------------------------

        vectors = (
            await self.embedding.embed_documents(
                chunks
            )
        )

        if len(vectors) != len(chunks):

            raise RuntimeError(
                "Embedding count does not match "
                "chunk count."
            )

        # ---------------------------------------------
        # Ensure collection
        # ---------------------------------------------

        await self.qdrant.ensure_collection(
            vector_size=len(vectors[0])
        )

        # ---------------------------------------------
        # Stage new version
        # ---------------------------------------------

        inserted = (
            await self.qdrant.upsert_chunks(
                document_id=document_id,
                version=next_version,
                chunks=chunks,
                vectors=vectors,
                metadata=request.metadata,
                content_hash=content_hash,
                title=request.title,
            )
        )

        # ---------------------------------------------
        # Validate
        # ---------------------------------------------

        valid = (
            await self.qdrant.validate_version(
                document_id=document_id,
                version=next_version,
                expected_count=len(chunks),
            )
        )

        if not valid:

            raise RuntimeError(
                "Qdrant validation failed. "
                "Previous active version was not changed."
            )

        # ---------------------------------------------
        # COMMIT
        #
        # This is the important safety point.
        # The manifest changes only after the complete
        # new version exists.
        # ---------------------------------------------

        await self.qdrant.commit_version(
            document_id=document_id,
            version=next_version,
            content_hash=content_hash,
            chunk_count=len(chunks),
        )

        # ---------------------------------------------
        # Cleanup old versions
        #
        # Cleanup happens AFTER commit.
        # If cleanup fails, active version is still valid.
        # ---------------------------------------------

        if current_version is not None:

            try:

                await self.qdrant.delete_old_versions(
                    document_id=document_id,
                    active_version=next_version,
                )

            except Exception:
                # Cleanup failure should NOT invalidate
                # the newly committed version.
                pass

        # ---------------------------------------------
        # Response
        # ---------------------------------------------

        return CreateTextDocumentResponse(
            document_id=document_id,

            version=next_version,

            previous_version=current_version,

            status="created"
            if current_version is None
            else "updated",

            content_hash=content_hash,

            chunk_count=inserted,

            message=(
                "Document created successfully."
                if current_version is None
                else "Document updated successfully."
            ),
        )