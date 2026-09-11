from .bm25_search import get_bm25_index
from .vector_store import search_chunks as dense_search
from .embeddings import embed_text


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Combine multiple result lists using RRF scoring. Dedup by document_id:chunk_index or text hash."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    def _key(doc: dict) -> str:
        # Prefer stable logical key so BM25 and dense same chunk dedup (now share same id, but fallback for old data)
        if doc.get("document_id") is not None and doc.get("chunk_index") is not None:
            return f"{doc['document_id']}:{doc['chunk_index']}"
        if doc.get("id"):
            return str(doc["id"])
        # Fallback: text hash
        return str(hash(doc.get("text", "")[:200]))

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = _key(doc)
            if key not in scores:
                scores[key] = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (k + rank + 1)

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

    # Final dedup by normalized text to avoid showing same invoice content twice (e.g., duplicate uploads)
    seen_text = set()
    deduped = []
    for doc in combined:
        norm = " ".join(doc.get("text", "").split()[:30]).lower()  # first 30 words normalized
        if norm not in seen_text:
            seen_text.add(norm)
            deduped.append(doc)
        if len(deduped) >= limit:
            break

    return deduped
