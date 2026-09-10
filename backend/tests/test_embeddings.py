from app.services.embeddings import embed_text, embed_texts, VECTOR_SIZE


class TestEmbeddings:
    def test_embed_single_text(self):
        embedding = embed_text("hello world")
        assert isinstance(embedding, list)
        assert len(embedding) == VECTOR_SIZE

    def test_embed_multiple_texts(self):
        texts = ["hello world", "foo bar", "test sentence"]
        embeddings = embed_texts(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == VECTOR_SIZE

    def test_embed_empty_list(self):
        assert embed_texts([]) == []

    def test_embedding_is_normalized(self):
        embedding = embed_text("test")
        magnitude = sum(x**2 for x in embedding) ** 0.5
        assert 0.99 <= magnitude <= 1.01

    def test_similar_texts_have_higher_similarity(self):
        emb1 = embed_text("washing machine purchase")
        emb2 = embed_text("bought a washing machine")
        emb3 = embed_text("electricity bill payment")

        sim_related = sum(a * b for a, b in zip(emb1, emb2))
        sim_unrelated = sum(a * b for a, b in zip(emb1, emb3))

        assert sim_related > sim_unrelated
