import uuid
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt_tab", quiet=True)


class BM25Index:
    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.texts: list[str] = []
        self.ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def add_documents(self, chunks: list[dict]):
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            self.documents[doc_id] = chunk
            self.ids.append(doc_id)
            self.texts.append(chunk["text"])

        self._rebuild_index()

    def _rebuild_index(self):
        if self.texts:
            tokenized = [word_tokenize(text.lower()) for text in self.texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.bm25 is None or not self.texts:
            return []

        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        scored_results = list(zip(self.ids, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored_results[:limit]:
            if score > 0:
                chunk = self.documents[doc_id].copy()
                chunk["score"] = float(score)
                chunk["id"] = doc_id
                results.append(chunk)

        return results

    def remove_document(self, document_id: str):
        to_remove = [
            i for i, (doc_id, chunk) in enumerate(
                zip(self.ids, [self.documents[did] for did in self.ids])
            )
            if chunk.get("document_id") == document_id
        ]

        for i in sorted(to_remove, reverse=True):
            doc_id = self.ids[i]
            del self.documents[doc_id]
            self.ids.pop(i)
            self.texts.pop(i)

        if to_remove:
            self._rebuild_index()

    def clear(self):
        self.documents.clear()
        self.texts.clear()
        self.ids.clear()
        self.bm25 = None


_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _index
