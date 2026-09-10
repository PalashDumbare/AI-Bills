import io
import os
import uuid
import pytest
from unittest.mock import patch, AsyncMock


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_no_documents_indexed(self, client):
        with patch("app.services.hybrid_search.hybrid_search", return_value=[]):
            response = await client.post(
                "/chat",
                json={"question": "What is my purchase date?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["sources"] == []

    @pytest.mark.asyncio
    async def test_chat_with_document_filter(self, client):
        fake_id = str(uuid.uuid4())
        response = await client.post(
            "/chat",
            json={
                "question": "What did I buy?",
                "document_id": fake_id,
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_full_flow(self, client):
        file_content = b"LG Washing Machine\nAmount: 32999\nWarranty: 2 years"
        upload_response = await client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            data={"user_id": "test-user"},
        )
        doc_id = upload_response.json()["document_id"]

        extracted_dir = os.path.join(os.path.dirname(__file__), "..", "app", "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        extracted_path = os.path.join(extracted_dir, f"{doc_id}_extracted.txt")
        with open(extracted_path, "w") as f:
            f.write("LG Washing Machine\nAmount: 32999\nWarranty: 2 years")

        try:
            index_response = await client.post(f"/documents/{doc_id}/index")
            assert index_response.status_code == 200

            with patch("app.services.chat.chat_with_documents", new_callable=AsyncMock) as mock_chat:
                mock_chat.return_value = {
                    "answer": "You paid 32999 for the LG washing machine.",
                    "sources": [
                        {
                            "document_id": doc_id,
                            "chunk_index": 0,
                            "text": "LG Washing Machine",
                            "score": 0.9,
                        }
                    ],
                }

                response = await client.post(
                    "/chat",
                    json={"question": "How much did I pay?"},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "You paid 32999 for the LG washing machine."
            assert len(data["sources"]) == 1
            assert data["sources"][0]["document_id"] == doc_id
        finally:
            if os.path.exists(extracted_path):
                os.remove(extracted_path)
