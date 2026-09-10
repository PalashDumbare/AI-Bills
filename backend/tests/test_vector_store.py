import uuid
from app.services.vector_store import (
    index_chunks,
    search_chunks,
    delete_document_chunks,
    ensure_collection,
    get_client,
)


def cleanup_collection():
    client = get_client()
    try:
        client.delete_collection("document_chunks")
    except Exception:
        pass


class TestVectorStore:
    def setup_method(self):
        cleanup_collection()

    def teardown_method(self):
        cleanup_collection()

    def test_ensure_collection(self):
        ensure_collection()
        client = get_client()
        collections = [c.name for c in client.get_collections().collections]
        assert "document_chunks" in collections

    def test_index_and_search(self):
        doc_id = str(uuid.uuid4())
        texts = ["LG washing machine", "Purchase date 2026-01-15"]
        from app.services.embeddings import embed_texts
        embeddings = embed_texts(texts)

        count = index_chunks(doc_id, texts, embeddings)
        assert count == 2

        query_emb = embed_texts(["washing machine"])[0]
        results = search_chunks(query_emb, document_id=doc_id)
        assert len(results) > 0
        assert results[0]["document_id"] == doc_id

    def test_search_filters_by_document(self):
        doc1 = str(uuid.uuid4())
        doc2 = str(uuid.uuid4())
        from app.services.embeddings import embed_texts

        index_chunks(doc1, ["apple iphone"], embed_texts(["apple iphone"]))
        index_chunks(doc2, ["samsung galaxy"], embed_texts(["samsung galaxy"]))

        query_emb = embed_texts(["phone"])[0]
        results = search_chunks(query_emb, document_id=doc1)
        for r in results:
            assert r["document_id"] == doc1

    def test_delete_document_chunks(self):
        doc_id = str(uuid.uuid4())
        from app.services.embeddings import embed_texts

        index_chunks(doc_id, ["test text"], embed_texts(["test text"]))
        delete_document_chunks(doc_id)

        query_emb = embed_texts(["test"])[0]
        results = search_chunks(query_emb, document_id=doc_id)
        assert len(results) == 0

    def test_metadata_stored(self):
        doc_id = str(uuid.uuid4())
        from app.services.embeddings import embed_texts

        index_chunks(
            doc_id,
            ["test chunk"],
            embed_texts(["test chunk"]),
            metadata={"document_type": "appliance_invoice"},
        )

        query_emb = embed_texts(["test"])[0]
        results = search_chunks(query_emb, document_id=doc_id)
        assert len(results) > 0
