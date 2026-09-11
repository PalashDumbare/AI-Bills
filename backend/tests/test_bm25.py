from app.services.bm25_search import BM25Index, get_bm25_index


class TestBM25:
    def test_add_and_search(self):
        index = BM25Index()
        chunks = [
            {"document_id": "doc1", "text": "LG washing machine purchase"},
            {"document_id": "doc2", "text": "Samsung refrigerator warranty"},
            {"document_id": "doc3", "text": "Electricity bill payment"},
        ]
        index.add_documents(chunks)

        results = index.search("washing machine", limit=2)
        assert len(results) > 0
        assert "washing machine" in results[0]["text"].lower()

    def test_search_empty_index(self):
        index = BM25Index()
        results = index.search("test", limit=5)
        assert results == []

    def test_remove_document(self):
        index = BM25Index()
        chunks = [
            {"document_id": "doc1", "text": "apple iphone purchase"},
            {"document_id": "doc2", "text": "samsung galaxy purchase"},
        ]
        index.add_documents(chunks)
        assert len(index.texts) == 2

        index.remove_document("doc1")
        assert len(index.texts) == 1

    def test_clear(self):
        index = BM25Index()
        index.add_documents([{"document_id": "doc1", "text": "test"}])
        index.clear()
        assert len(index.texts) == 0

    def test_global_index(self):
        index = get_bm25_index()
        assert isinstance(index, BM25Index)


class TestHybridSearch:
    def test_hybrid_search(self):
        from app.services.bm25_search import BM25Index
        from app.services.hybrid_search import reciprocal_rank_fusion

        bm25_results = [
            {"id": "1", "text": "washing machine", "score": 2.5},
            {"id": "2", "text": "refrigerator", "score": 1.0},
        ]
        dense_results = [
            {"id": "1", "text": "washing machine", "score": 0.9},
            {"id": "3", "text": "dryer", "score": 0.7},
        ]

        combined = reciprocal_rank_fusion([bm25_results, dense_results])
        assert len(combined) > 0
        assert combined[0]["id"] == "1"

    def test_rrf_empty_lists(self):
        from app.services.hybrid_search import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []
