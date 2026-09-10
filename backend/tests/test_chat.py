import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestChatService:
    @pytest.mark.asyncio
    async def test_chat_with_documents(self):
        from app.services.chat import chat_with_documents

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "The washing machine cost 32999 rupees."
        }
        mock_response.raise_for_status = MagicMock()

        context_chunks = [
            {
                "document_id": "doc-123",
                "chunk_index": 0,
                "text": "LG Washing Machine\nAmount: 32999",
                "score": 0.85,
            }
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await chat_with_documents(
                question="How much did I pay?",
                context_chunks=context_chunks,
            )

        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["document_id"] == "doc-123"

    @pytest.mark.asyncio
    async def test_chat_handles_llm_error(self):
        from app.services.chat import chat_with_documents

        context_chunks = [
            {
                "document_id": "doc-123",
                "chunk_index": 0,
                "text": "Some text",
                "score": 0.85,
            }
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.side_effect = Exception("Connection failed")
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await chat_with_documents(
                question="Test question",
                context_chunks=context_chunks,
            )

        assert "answer" in result
        assert len(result["sources"]) == 1
