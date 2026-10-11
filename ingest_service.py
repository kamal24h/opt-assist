# ingest_service.py
from typing import Any, Dict, List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from config import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_url,
    )


# جداکننده‌ها به ترتیب اولویت: ابتدا پاراگراف، بعد خط، بعد پایان جمله،
# سپس علائم نیم‌جمله و در نهایت فاصله و کاراکتر. علائم فارسی (؟ ؛ ،) هم پوشش داده شده‌اند.
PERSIAN_AWARE_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "؟ ",
    "? ",
    "! ",
    "؛ ",
    "، ",
    " ",
    "",
]


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=PERSIAN_AWARE_SEPARATORS,
        add_start_index=True,
    )


def ensure_collection_exists(client: QdrantClient, embeddings: OllamaEmbeddings):
    if client.collection_exists(settings.collection_name):
        return

    sample_vector = embeddings.embed_query("init")
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(
            size=len(sample_vector),
            distance=Distance.COSINE,
        ),
    )


def _get_vector_store():
    client = get_qdrant_client()
    embeddings = get_embeddings()
    ensure_collection_exists(client, embeddings)

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )


def _add_documents_in_batches(vector_store: QdrantVectorStore, documents: List[LangChainDocument]):
    for index in range(0, len(documents), settings.embed_batch_size):
        batch = documents[index : index + settings.embed_batch_size]
        vector_store.add_documents(batch)


def _split_and_enrich(documents: List[LangChainDocument], base_metadata: Dict[str, Any]):
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata.update(base_metadata)
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
        loader = PyPDFLoader(file_path)
    elif normalized_type in ("doc", "docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"نوع سند '{doc_type}' پشتیبانی نمی‌شود.")

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

    chunks = _split_and_enrich(raw_documents, base_metadata)
    if not chunks:
        return 0

    vector_store = _get_vector_store()
    _add_documents_in_batches(vector_store, chunks)
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

    document = LangChainDocument(page_content=doc_text, metadata={})
    base_metadata = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "app_id": app_id,
        **(metadata or {}),
    }

    chunks = _split_and_enrich([document], base_metadata)
    if not chunks:
        return 0

    vector_store = _get_vector_store()
    _add_documents_in_batches(vector_store, chunks)
    return len(chunks)


def delete_document_vectors(doc_id: str):
    client = get_qdrant_client()
    if not client.collection_exists(settings.collection_name):
        return

    client.delete(
        collection_name=settings.collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
    )