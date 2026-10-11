import hashlib
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)


class QdrantDocumentService:

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
    ):
        self.client = client
        self.collection_name = collection_name

    # ---------------------------------------------------------
    # Collection
    # ---------------------------------------------------------

    async def ensure_collection(
        self,
        vector_size: int,
    ) -> None:

        exists = await self.client.collection_exists(
            self.collection_name
        )

        if exists:
            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    @staticmethod
    def manifest_point_id(
        document_id: str,
    ) -> str:

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"document-manifest:{document_id}",
            )
        )

    async def get_manifest(
        self,
        document_id: str,
    ) -> dict | None:

        point_id = self.manifest_point_id(
            document_id
        )

        result = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )

        if not result:
            return None

        payload = result[0].payload

        if not payload:
            return None

        return payload

    # ---------------------------------------------------------
    # Version
    # ---------------------------------------------------------

    async def get_current_version(
        self,
        document_id: str,
    ) -> int | None:

        manifest = await self.get_manifest(
            document_id
        )

        if manifest is None:
            return None

        return manifest.get(
            "active_version"
        )

    # ---------------------------------------------------------
    # Deterministic chunk point ID
    # ---------------------------------------------------------

    @staticmethod
    def chunk_point_id(
        document_id: str,
        version: int,
        chunk_index: int,
    ) -> str:

        key = (
            f"{document_id}:"
            f"{version}:"
            f"{chunk_index}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                key,
            )
        )

    # ---------------------------------------------------------
    # Upsert chunks
    # ---------------------------------------------------------

    async def upsert_chunks(
        self,
        document_id: str,
        version: int,
        chunks: list[str],
        vectors: list[list[float]],
        metadata: dict,
        content_hash: str,
        title: str | None,
    ) -> int:

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks and vectors must match."
            )

        if not chunks:
            raise ValueError(
                "Document contains no chunks."
            )

        points: list[PointStruct] = []

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):

            point_id = self.chunk_point_id(
                document_id=document_id,
                version=version,
                chunk_index=index,
            )

            payload = {
                "record_type": "chunk",

                "document_id": document_id,

                "version": version,

                "chunk_index": index,

                "chunk_count": len(chunks),

                "text": chunk,

                "content_hash": content_hash,

                "title": title,

                "metadata": metadata,

                "status": "staged",
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        batch_size = 64

        for start in range(
            0,
            len(points),
            batch_size,
        ):

            batch = points[
                start:start + batch_size
            ]

            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        return len(points)

    # ---------------------------------------------------------
    # Validate inserted version
    # ---------------------------------------------------------

    async def validate_version(
        self,
        document_id: str,
        version: int,
        expected_count: int,
    ) -> bool:

        result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="record_type",
                        match=MatchValue(
                            value="chunk"
                        ),
                    ),
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    ),
                    FieldCondition(
                        key="version",
                        match=MatchValue(
                            value=version
                        ),
                    ),
                ]
            ),
            exact=True,
        )

        return result.count == expected_count

    # ---------------------------------------------------------
    # Commit version
    # ---------------------------------------------------------

    async def commit_version(
        self,
        document_id: str,
        version: int,
        content_hash: str,
        chunk_count: int,
    ) -> None:

        point_id = self.manifest_point_id(
            document_id
        )

        # Manifest does not participate in vector search.
        # We use a zero vector.
        collection_info = (
            await self.client.get_collection(
                self.collection_name
            )
        )

        vector_size = (
            collection_info.config.params.vectors.size
        )

        manifest = PointStruct(
            id=point_id,
            vector=[0.0] * vector_size,
            payload={
                "record_type": "manifest",

                "document_id": document_id,

                "active_version": version,

                "content_hash": content_hash,

                "chunk_count": chunk_count,

                "status": "active",
            },
        )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[manifest],
            wait=True,
        )

    # ---------------------------------------------------------
    # Remove old versions
    # ---------------------------------------------------------

    async def delete_old_versions(
        self,
        document_id: str,
        active_version: int,
    ) -> None:

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="record_type",
                        match=MatchValue(
                            value="chunk"
                        ),
                    ),

                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),                                        
                    ),
                ],
                must_not=[
                    FieldCondition(
                        key="version",
                        match=MatchValue(
                            value=active_version
                        ),
                    )
                ],
            ),
            wait=True,
        )

    # ---------------------------------------------------------
    # Search active version
    # ---------------------------------------------------------

    async def search(
        self,
        document_id: str,
        query_vector: list[float],
        limit: int = 5,
    ):

        manifest = await self.get_manifest(
            document_id
        )

        if manifest is None:
            return []

        active_version = manifest[
            "active_version"
        ]

        result = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="record_type",
                        match=MatchValue(
                            value="chunk"
                        ),
                    ),
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    ),
                    FieldCondition(
                        key="version",
                        match=MatchValue(
                            value=active_version
                        ),
                    ),
                ]
            ),
            limit=limit,
        )

        return result