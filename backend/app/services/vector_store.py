import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from ..config import settings
from .embeddings import VECTOR_SIZE

COLLECTION_NAME = "document_chunks"

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _client


def ensure_collection():
    client = get_client()
    collections = client.get_collections().collections
    existing = [c.name for c in collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def index_chunks(
    document_id: str,
    texts: list[str],
    embeddings: list[list[float]],
    metadata: dict | None = None,
):
    ensure_collection()
    client = get_client()

    points = []
    bm25_chunks = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        point_id = str(uuid.uuid4())
        payload = {
            "document_id": document_id,
            "chunk_index": i,
            "text": text,
        }
        if metadata:
            payload.update(metadata)

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        ))

        bm25_chunks.append({
            "id": point_id,
            "document_id": document_id,
            "chunk_index": i,
            "text": text,
        })

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    from .bm25_search import get_bm25_index
    bm25_index = get_bm25_index()
    bm25_index.add_documents(bm25_chunks)

    return len(points)


def search_chunks(
    query_embedding: list[float],
    document_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    ensure_collection()
    client = get_client()

    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                ),
            ],
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=limit,
    )

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "document_id": hit.payload.get("document_id", ""),
            "chunk_index": hit.payload.get("chunk_index", 0),
        }
        for hit in results.points
    ]


def delete_document_chunks(document_id: str):
    ensure_collection()
    client = get_client()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                ),
            ],
        ),
    )

    from .bm25_search import get_bm25_index
    bm25_index = get_bm25_index()
    bm25_index.remove_document(document_id)
