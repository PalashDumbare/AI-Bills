import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestWebSearchService:
    @pytest.mark.asyncio
    async def test_search_web_mock(self):
        from app.services.web_search import search_web

        html = '''
        <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.lg.com%2Fin%2Fsupport">LG Care India</a></h2>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.lg.com%2Fin%2Fsupport">LG toll free 1800 315 9999</a>
        '''
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = await search_web("LG customer care", limit=1)
        assert len(results) == 1
        assert "LG" in results[0]["title"]
        assert results[0]["url"].startswith("https://www.lg.com")

    @pytest.mark.asyncio
    async def test_extract_care_numbers(self):
        from app.services.web_search import extract_care_numbers
        assert "1800 266 1880" in extract_care_numbers("Call 1800 266 1880 for Bosch")
        assert "1-800-243-0000" in extract_care_numbers("LG US 1-800-243-0000")
        assert extract_care_numbers("no number here") == []


class TestWebSearchEndpoint:
    @pytest.mark.asyncio
    async def test_get_web_search(self, client):
        from unittest.mock import AsyncMock, patch

        mock_results = [{"title": "LG Care", "url": "https://lg.com", "snippet": "1800 315 9999", "care_numbers": ["1800 315 9999"]}]
        with patch("app.services.web_search.search_with_details", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            resp = await client.get("/web-search", params={"q": "LG customer care"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "LG customer care"
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_post_web_search(self, client):
        from unittest.mock import AsyncMock, patch

        mock_results = [{"title": "Sony Bravia Spec", "url": "https://sony.com/tv", "snippet": "QD-OLED 4K panel", "care_numbers": None}]
        with patch("app.services.web_search.search_with_details", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            resp = await client.post("/web-search", json={"query": "Sony Bravia 55OLED specifications", "limit": 2})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    @pytest.mark.asyncio
    async def test_chat_web_fallback(self, client):
        from unittest.mock import patch, AsyncMock

        mock_results = [{"title": "LG Care", "url": "https://lg.com", "snippet": "1800 315 9999"}]
        with patch("app.services.hybrid_search.hybrid_search", return_value=[]):
            with patch("app.services.web_search.search_with_details", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = mock_results
                with patch("app.services.chat.chat_with_documents", new_callable=AsyncMock) as mock_chat:
                    mock_chat.return_value = {"answer": "LG care is 1800 315 9999 [1]", "sources": [{"document_id": "web", "chunk_index": 0, "text": "LG 1800...", "score": 0.9}]}
                    resp = await client.post("/chat", json={"question": "LG customer care number", "use_web_search": True})
        assert resp.status_code == 200
        assert "from web search" in resp.json()["answer"]
