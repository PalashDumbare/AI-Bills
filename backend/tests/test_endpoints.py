"""Integration tests for API endpoints."""
import io
import pytest
from unittest.mock import patch, AsyncMock
from datetime import date


class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_document(self, client):
        file_content = b"test pdf content"
        response = await client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            data={"user_id": "test-user-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.pdf"
        assert data["user_id"] == "test-user-123"

    @pytest.mark.asyncio
    async def test_upload_empty_file(self, client):
        response = await client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
            data={"user_id": "test-user-123"},
        )
        assert response.status_code == 400


class TestExtractEndpoint:
    @pytest.mark.asyncio
    async def test_extract_document_not_found(self, client):
        response = await client.post("/documents/nonexistent/extract")
        assert response.status_code == 404


class TestStructureEndpoint:
    @pytest.mark.asyncio
    async def test_structure_document_not_found(self, client):
        response = await client.post(
            "/documents/nonexistent/structure",
            json={"user_id": "test-user-123"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_structure_without_extract(self, client):
        # First upload a document
        file_content = b"test content"
        upload_response = await client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            data={"user_id": "test-user-123"},
        )
        doc_id = upload_response.json()["document_id"]

        # Try to structure without extracting first
        response = await client.post(
            f"/documents/{doc_id}/structure",
            json={"user_id": "test-user-123"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_structure_success(self, client):
        # Upload document
        file_content = b"test content"
        upload_response = await client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            data={"user_id": "test-user-123"},
        )
        doc_id = upload_response.json()["document_id"]

        # Create extracted text file manually
        import os
        extracted_dir = os.path.join(os.path.dirname(__file__), "..", "app", "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        extracted_path = os.path.join(extracted_dir, f"{doc_id}_extracted.txt")
        with open(extracted_path, "w") as f:
            f.write("LG Washing Machine\nAmount: 32999\nWarranty: 2 years")

        # Mock LLM to return None (use regex fallback)
        with patch("app.services.llm_client.extract_with_llm", new_callable=AsyncMock, return_value=None):
            response = await client.post(
                f"/documents/{doc_id}/structure",
                json={"user_id": "test-user-123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert data["document_type"] == "appliance_invoice"
        assert data["structured_data"]["brand"] == "LG"
        assert float(data["structured_data"]["amount"]) == 32999.0

        # Cleanup
        os.remove(extracted_path)

    @pytest.mark.asyncio
    async def test_structure_idempotent(self, client):
        # Upload document
        file_content = b"test content"
        upload_response = await client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            data={"user_id": "test-user-123"},
        )
        doc_id = upload_response.json()["document_id"]

        # Create extracted text file
        import os
        extracted_dir = os.path.join(os.path.dirname(__file__), "..", "app", "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        extracted_path = os.path.join(extracted_dir, f"{doc_id}_extracted.txt")
        with open(extracted_path, "w") as f:
            f.write("LG Washing Machine\nAmount: 32999\nWarranty: 2 years")

        # Mock LLM
        with patch("app.services.llm_client.extract_with_llm", new_callable=AsyncMock, return_value=None):
            # First structure call
            response1 = await client.post(
                f"/documents/{doc_id}/structure",
                json={"user_id": "test-user-123"},
            )
            assert response1.status_code == 200

            # Second structure call should return same data
            response2 = await client.post(
                f"/documents/{doc_id}/structure",
                json={"user_id": "test-user-123"},
            )
            assert response2.status_code == 200
            assert response1.json()["structured_data"]["id"] == response2.json()["structured_data"]["id"]

        # Cleanup
        os.remove(extracted_path)
