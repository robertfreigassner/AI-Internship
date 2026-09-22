"""Pinecone vector store helpers. Secrets come only from the environment."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv(Path(__file__).resolve().parent / ".env")

# Must match the Pinecone index dimension (1536 for text-embedding-3-small).
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EXPECTED_DIMENSIONS = 1536


def embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable {name}")
    return value


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    _require_env("OPENAI_API_KEY")
    return OpenAI()


@lru_cache(maxsize=1)
def _pinecone_client() -> Pinecone:
    return Pinecone(api_key=_require_env("PINECONE_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed with the same model used at ingest and query time."""

    if not texts:
        return []
    response = _openai_client().embeddings.create(model=embedding_model(), input=texts)
    by_index = {item.index: item.embedding for item in response.data}
    return [by_index[i] for i in range(len(texts))]


def get_index():
    """Target the configured index by host (preferred) or by name."""

    pc = _pinecone_client()
    host = os.getenv("PINECONE_INDEX_HOST", "").strip()
    if host:
        return pc.index(host=host)
    return pc.index(name=_require_env("PINECONE_INDEX_NAME"))


def chunk_settings() -> tuple[int, int]:
    size = int(os.getenv("CHUNK_SIZE", "800"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    return size, overlap


def chunk_text(text: str) -> list[str]:
    """Split with RecursiveCharacterTextSplitter. Size/overlap come from env."""

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size, chunk_overlap = chunk_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [part.strip() for part in splitter.split_text(text) if part.strip()]


def upsert_texts(ids: list[str], texts: list[str], metadatas: list[dict] | None = None) -> int:
    if len(ids) != len(texts):
        raise ValueError("ids and texts must be the same length")
    vectors = embed_texts(texts)
    payload = []
    for i, (vector_id, values) in enumerate(zip(ids, vectors)):
        item: dict = {"id": vector_id, "values": values}
        if metadatas:
            item["metadata"] = metadatas[i]
        payload.append(item)
    get_index().upsert(vectors=payload)
    return len(payload)


def ingest_document(document_id: str, text: str, source: str = "") -> int:
    """Embed chunks and upsert them with document_id, chunk_index, and source."""

    chunks = chunk_text(text)
    if not chunks:
        return 0
    ids = [f"{document_id}:{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "document_id": document_id,
            "chunk_index": index,
            "source": source or "",
            "text": chunk,
        }
        for index, chunk in enumerate(chunks)
    ]
    return upsert_texts(ids, chunks, metadatas)


def query_similar(text: str, top_k: int = 5) -> list[dict]:
    vector = embed_texts([text])[0]
    result = get_index().query(vector=vector, top_k=top_k, include_metadata=True)
    matches = getattr(result, "matches", None) or result.get("matches", [])
    out = []
    for match in matches:
        out.append(
            {
                "id": getattr(match, "id", None) or match.get("id"),
                "score": getattr(match, "score", None) or match.get("score"),
                "metadata": getattr(match, "metadata", None) or match.get("metadata"),
            }
        )
    return out


def pinecone_status() -> dict:
    """Reachability check: no secrets, only index name/stats and embedding model."""

    index_name = _require_env("PINECONE_INDEX_NAME")
    index = get_index()
    stats = index.describe_index_stats()
    dimension = getattr(stats, "dimension", None)
    if dimension is None and isinstance(stats, dict):
        dimension = stats.get("dimension")
    total = getattr(stats, "total_vector_count", None)
    if total is None and isinstance(stats, dict):
        total = stats.get("total_vector_count")
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces")

    return {
        "ok": True,
        "index_name": index_name,
        "embedding_model": embedding_model(),
        "expected_dimensions": EXPECTED_DIMENSIONS,
        "index_dimensions": dimension,
        "total_vector_count": total,
        "namespace_count": len(namespaces or {}),
        "dimensions_match": dimension in (None, EXPECTED_DIMENSIONS),
    }
