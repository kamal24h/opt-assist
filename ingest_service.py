# ingest_service.py

import os

from typing import (
    Any,
    Dict,
    List,
)

from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)

from langchain_core.documents import Document as LangChainDocument

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_ollama import (
    OllamaEmbeddings,
)

from langchain_qdrant import (
    QdrantVectorStore,
)

from qdrant_client import (
    QdrantClient,
)

from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
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

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qllama/bge-m3:q4_k_m",
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
    )
)

EMBED_BATCH_SIZE = int(
    os.getenv(
        "EMBED_BATCH_SIZE",
        "32",
    )
)


def get_qdrant_client() -> QdrantClient:

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )


def get_embeddings() -> OllamaEmbeddings:

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )


def get_text_splitter():

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )


def ensure_collection_exists(
    client: QdrantClient,
    embeddings: OllamaEmbeddings,
):

    if client.collection_exists(
        COLLECTION_NAME
    ):
        return

    sample_vector = embeddings.embed_query(
        "init"
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=len(sample_vector),
            distance=Distance.COSINE,
        ),
    )


def _get_vector_store():

    client = get_qdrant_client()
    embeddings = get_embeddings()

    ensure_collection_exists(
        client,
        embeddings,
    )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def _add_documents_in_batches(
    vector_store: QdrantVectorStore,
    documents: List[LangChainDocument],
):

    for index in range(
        0,
        len(documents),
        EMBED_BATCH_SIZE,
    ):

        batch = documents[
            index:index + EMBED_BATCH_SIZE
        ]

        vector_store.add_documents(
            batch
        )


def _split_and_enrich(
    documents: List[LangChainDocument],
    base_metadata: Dict[str, Any],
):

    splitter = get_text_splitter()

    chunks = splitter.split_documents(
        documents
    )

    for chunk in chunks:

        chunk.metadata.update(
            base_metadata
        )

    return chunks


def process_and_index_document(
    file_path: str,
    doc_id: str,
    doc_name: str,
    doc_type: str,
    department: str,
    app_id: str,
    metadata: Dict[str, Any],
) -> int:

    normalized_type = doc_type.lower()

    if normalized_type == "pdf":

        loader = PyPDFLoader(
            file_path
        )

    elif normalized_type in (
        "doc",
        "docx",
    ):

        loader = UnstructuredWordDocumentLoader(
            file_path
        )

    else:

        raise ValueError(
            f"نوع سند '{doc_type}' پشتیبانی نمی‌شود."
        )

    raw_documents = loader.load()

    if not raw_documents:
        return 0

    base_metadata = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "department": department,
        "app_id": app_id,
        **(metadata or {}),
    }

    chunks = _split_and_enrich(
        raw_documents,
        base_metadata,
    )

    if not chunks:
        return 0

    vector_store = _get_vector_store()

    _add_documents_in_batches(
        vector_store,
        chunks,
    )

    return len(chunks)


def process_and_index_text_document(
    doc_text: str,
    doc_id: str,
    doc_name: str,
    app_id: str,
    metadata: Dict[str, Any],
) -> int:

    if not doc_text or not doc_text.strip():
        return 0

    document = LangChainDocument(
        page_content=doc_text,
        metadata={},
    )

    base_metadata = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "app_id": app_id,
        **(metadata or {}),
    }

    chunks = _split_and_enrich(
        [document],
        base_metadata,
    )

    if not chunks:
        return 0

    vector_store = _get_vector_store()

    _add_documents_in_batches(
        vector_store,
        chunks,
    )

    return len(chunks)


def delete_document_vectors(
    doc_id: str,
):

    client = get_qdrant_client()

    if not client.collection_exists(
        COLLECTION_NAME
    ):
        return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(
                        value=doc_id
                    ),
                )
            ]
        ),
    )