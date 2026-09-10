"""Unit tests for LLM extractor with mocked Ollama."""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.llm_client import extract_with_llm, _normalize_response


SAMPLE_TEXT = """LG Washing Machine
Purchase Date: 15 Jan 2026
Amount: 32999
Model: FHM1207
Warranty: 2 years"""


class TestLLMExtraction:
    @pytest.mark.asyncio
    @patch("app.services.llm_client.httpx.AsyncClient")
    async def test_successful_llm_extraction(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "document_type": "appliance_invoice",
                "brand": "LG",
                "product": "Washing Machine",
                "model": "FHM1207",
                "purchase_date": "2026-01-15",
                "amount": 32999,
                "warranty_months": 24,
            })
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = await extract_with_llm(SAMPLE_TEXT)

        assert result is not None
        assert result["document_type"] == "appliance_invoice"
        assert result["brand"] == "LG"
        assert result["product"] == "Washing Machine"
        assert result["model"] == "FHM1207"
        assert result["amount"] == 32999.0
        assert result["warranty_months"] == 24

    @pytest.mark.asyncio
    @patch("app.services.llm_client.httpx.AsyncClient")
    async def test_llm_json_with_markdown(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '```json\n{"document_type": "bill", "provider": "BSES", "bill_type": "electricity", "amount": 1500}\n```'
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = await extract_with_llm("Electricity bill")

        assert result is not None
        assert result["document_type"] == "bill"
        assert result["provider"] == "BSES"
        assert result["amount"] == 1500.0

    @pytest.mark.asyncio
    @patch("app.services.llm_client.httpx.AsyncClient")
    async def test_llm_invalid_json(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "not valid json"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = await extract_with_llm("Some text")
        assert result is None

    @pytest.mark.asyncio
    @patch("app.services.llm_client.httpx.AsyncClient")
    async def test_llm_connection_error(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = await extract_with_llm("Some text")
        assert result is None


class TestNormalizeResponse:
    def test_normalize_appliance(self):
        data = {
            "document_type": "appliance_invoice",
            "brand": "Samsung",
            "product": "TV",
            "model": "UA55",
            "purchase_date": "2026-01-15",
            "amount": 45000,
            "warranty_months": 12,
        }
        result = _normalize_response(data)
        assert result is not None
        assert result["brand"] == "Samsung"
        assert result["amount"] == 45000.0
        assert result["warranty_months"] == 12

    def test_normalize_bill(self):
        data = {
            "document_type": "bill",
            "provider": "Jio",
            "bill_type": "mobile",
            "amount": 599,
        }
        result = _normalize_response(data)
        assert result is not None
        assert result["document_type"] == "bill"
        assert result["provider"] == "Jio"
        assert result["amount"] == 599.0

    def test_normalize_invalid_type(self):
        data = {"document_type": "unknown"}
        result = _normalize_response(data)
        assert result is None

    def test_normalize_null_fields(self):
        data = {
            "document_type": "appliance_invoice",
            "brand": None,
            "product": None,
            "model": None,
            "purchase_date": None,
            "amount": None,
            "warranty_months": None,
        }
        result = _normalize_response(data)
        assert result is not None
        assert result["brand"] is None
        assert result["amount"] is None
