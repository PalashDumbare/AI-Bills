import json
import os
import uuid
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt_tab", quiet=True)

# Persist BM25 to disk so restart doesn't lose in-memory index (Qdrant persists via Docker volume)
BM25_PERSIST_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "bm25_index.json")
# Fallback to qdrant_storage if backend/storage not writable
_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "qdrant_storage", "bm25_index.json")


class BM25Index:
    def __init__(self, persist: bool = True):
        self.documents: dict[str, dict] = {}
        self.texts: list[str] = []
        self.ids: list[str] = []
        self.bm25: BM25Okapi | None = None
        self._loaded = False
        self._persist_enabled = persist
        # In pytest, avoid polluting global persisted index — tests get isolated in-memory index
        if persist and not os.getenv("PYTEST_CURRENT_TEST"):
            self._load_from_disk()

    def _persist_path(self) -> str:
        # Prefer backend/storage, fallback to qdrant_storage
        try:
            os.makedirs(os.path.dirname(BM25_PERSIST_PATH), exist_ok=True)
            return BM25_PERSIST_PATH
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(_FALLBACK_PATH), exist_ok=True)
            return _FALLBACK_PATH
        except Exception:
            return BM25_PERSIST_PATH

    def _save_to_disk(self):
        if not getattr(self, "_persist_enabled", True):
            return
        if os.getenv("PYTEST_CURRENT_TEST"):
            return
        try:
            path = self._persist_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "ids": self.ids,
                "texts": self.texts,
                "documents": self.documents,
            }
            # Atomic write
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            pass  # fail silently, BM25 will rebuild from Qdrant

    def _load_from_disk(self):
        if self._loaded:
            return
        self._loaded = True
        for path in (BM25_PERSIST_PATH, _FALLBACK_PATH):
            try:
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    self.ids = data.get("ids", [])
                    self.texts = data.get("texts", [])
                    self.documents = data.get("documents", {})
                    if self.texts:
                        self._rebuild_index()
                    return
            except Exception:
                continue
        # If no file, try rebuild from Qdrant (Qdrant persists via Docker volume)
        try:
            self._rebuild_from_qdrant()
        except Exception:
            pass

    def _rebuild_from_qdrant(self):
        try:
            from .vector_store import get_client, COLLECTION_NAME
            client = get_client()
            # Ensure collection exists
            try:
                client.get_collection(COLLECTION_NAME)
            except Exception:
                return
            points, next_page = client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,
                with_vectors=False,
                limit=100,
            )
            if not points:
                return
            chunks: list[dict] = []
            for p in points:
                payload = p.payload or {}
                chunks.append({
                    "id": str(p.id),
                    "document_id": payload.get("document_id", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                })
            # Handle pagination if >100
            while next_page:
                points, next_page = client.scroll(
                    collection_name=COLLECTION_NAME,
                    with_payload=True,
                    with_vectors=False,
                    limit=100,
                    offset=next_page,
                )
                for p in points:
                    payload = p.payload or {}
                    chunks.append({
                        "id": str(p.id),
                        "document_id": payload.get("document_id", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                        "text": payload.get("text", ""),
                    })
            if chunks:
                # Directly populate without calling add_documents to avoid double save
                self.documents.clear()
                self.ids.clear()
                self.texts.clear()
                for c in chunks:
                    doc_id = c["id"]
                    self.documents[doc_id] = c
                    self.ids.append(doc_id)
                    self.texts.append(c["text"])
                self._rebuild_index()
                self._save_to_disk()
        except Exception:
            pass

    def add_documents(self, chunks: list[dict]):
        for chunk in chunks:
            # Respect upstream id (Qdrant point_id) so BM25 and dense share same id for RRF dedup
            doc_id = chunk.get("id") or str(uuid.uuid4())
            # Avoid duplicate ids if same chunk re-indexed
            if doc_id in self.documents:
                continue
            self.documents[doc_id] = chunk
            self.ids.append(doc_id)
            self.texts.append(chunk["text"])

        self._rebuild_index()
        self._save_to_disk()

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
            self._save_to_disk()

    def clear(self):
        self.documents.clear()
        self.texts.clear()
        self.ids.clear()
        self.bm25 = None
        self._save_to_disk()


_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _index
