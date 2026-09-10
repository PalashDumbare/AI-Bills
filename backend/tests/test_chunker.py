from app.services.chunker import chunk_text


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].index == 0

    def test_long_text_multiple_chunks(self):
        text = "word " * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_overlap_preserves_context(self):
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        chunks = chunk_text(text, chunk_size=10, overlap=5)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_boundaries(self):
        text = "word " * 300
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char

    def test_no_duplicates(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        chunks = chunk_text(text, chunk_size=15, overlap=3)
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts))
