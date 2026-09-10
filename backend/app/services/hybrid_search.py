from .bm25_search import get_bm25_index
from .vector_store import search_chunks as dense_search
from .embeddings import embed_text


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Combine multiple result lists using RRF scoring."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.get("id") or doc.get("text", "")[:50]
            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [
        {**doc_map[doc_id], "rrf_score": scores[doc_id]}
        for doc_id in sorted_ids
    ]


def hybrid_search(
    query: str,
    document_id: str | None = None,
    limit: int = 5,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[dict]:
    """Search using both BM25 (keyword) and Dense (semantic) retrieval."""
    bm25_index = get_bm25_index()
    bm25_results = bm25_index.search(query, limit=limit * 2)

    query_embedding = embed_text(query)
    dense_results = dense_search(
        query_embedding=query_embedding,
        document_id=document_id,
        limit=limit * 2,
    )

    if document_id:
        bm25_results = [
            r for r in bm25_results
            if r.get("document_id") == document_id
        ]

    if not bm25_results and not dense_results:
        return []

    if not bm25_results:
        return dense_results[:limit]

    if not dense_results:
        return bm25_results[:limit]

    combined = reciprocal_rank_fusion([bm25_results, dense_results])

    return combined[:limit]
