import io
import os
import uuid
import pytest
from unittest.mock import patch, AsyncMock


class TestIndexEndpoint:
    @pytest.mark.asyncio
    async def test_index_document_not_found(self, client):
        fake_id = str(uuid.uuid4())
        response = await client.post(f"/documents/{fake_id}/index")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_index_without_extract(self, client):
        file_content = b"test content"
        upload_response = await client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            data={"user_id": "test-user"},
        )
        doc_id = upload_response.json()["document_id"]

        response = await client.post(f"/documents/{doc_id}/index")
        assert response.status_code == 404
        assert "extract" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_index_success(self, client):
        file_content = b"test content"
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
            response = await client.post(f"/documents/{doc_id}/index")
            assert response.status_code == 200
            data = response.json()
            assert data["document_id"] == doc_id
            assert data["chunks_indexed"] > 0
            assert data["total_chunks"] > 0
        finally:
            if os.path.exists(extracted_path):
                os.remove(extracted_path)
