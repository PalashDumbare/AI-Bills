import re
import httpx
from urllib.parse import quote_plus

# Free, no API key — DuckDuckGo HTML search + optional fallback API
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
DDG_API_URL = "https://api.duckduckgo.com/"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Regex for toll-free / customer care numbers (India 1800, US 1-800, 1860)
_CARE_NUMBER_RE = re.compile(
    r"(?:\+?91[\s-]?)?(?:1[\s-]?800|1800|1860)[\s-]?\d{3}[\s-]?\d{4}|\b\d{3,4}[\s-]\d{6,8}\b"
)

def extract_care_numbers(text: str) -> list[str]:
    """Extract deduped care numbers from text."""
    nums = _CARE_NUMBER_RE.findall(text)
    seen: set[str] = set()
    out: list[str] = []
    for n in nums:
        norm = re.sub(r"\s+", " ", n.strip())
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


async def _search_ddg_html(query: str, limit: int = 5) -> list[dict]:
    """Search DuckDuckGo HTML (free, no key). Parses titles/urls/snippets via regex."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
    html = ""
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        # Try GET first
        try:
            resp = await client.get(DDG_HTML_URL, params={"q": query}, headers=headers)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            html = ""
        # If GET gave empty/captcha (status 202 or 0 results), try POST fallback (DDG form uses POST)
        if not html or "result__a" not in html:
            try:
                resp = await client.post(DDG_HTML_URL, data={"q": query}, headers=headers)
                resp.raise_for_status()
                html = resp.text
            except Exception:
                pass

    if not html or "result__a" not in html:
        return []

    # DDG html pattern: result__title > result__a href, result__snippet
    # Tolerant: class="result__a" may have rel before it, so use [^>]*class="result__a"
    pattern = re.compile(
        r'class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    # Fallback pattern where snippet is div
    pattern_div = re.compile(
        r'class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</div>',
        re.DOTALL | re.IGNORECASE,
    )

    results: list[dict] = []
    for m in pattern.finditer(html):
        title = re.sub(r"<.*?>", "", m.group("title")).strip()
        url = m.group("url").strip()
        # DDG wraps url as //duckduckgo.com/l/?uddg=<encoded>&... — extract real url if present
        uddg = re.search(r"uddg=([^&]+)", url)
        if uddg:
            try:
                from urllib.parse import unquote
                url = unquote(uddg.group(1))
            except Exception:
                pass
        snippet = re.sub(r"<.*?>", "", m.group("snippet")).strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= limit:
                break

    if len(results) < limit:
        for m in pattern_div.finditer(html):
            title = re.sub(r"<.*?>", "", m.group("title")).strip()
            url = m.group("url").strip()
            uddg = re.search(r"uddg=([^&]+)", url)
            if uddg:
                try:
                    from urllib.parse import unquote
                    url = unquote(uddg.group(1))
                except Exception:
                    pass
            snippet = re.sub(r"<.*?>", "", m.group("snippet")).strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if title and url and not any(r["url"] == url for r in results):
                results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= limit:
                    break

    return results[:limit]


async def _search_ddg_api(query: str, limit: int = 5) -> list[dict]:
    """Fallback: DuckDuckGo JSON API (no key, limited)."""
    params = {"q": query, "format": "json", "pretty": "1", "no_html": "1"}
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(DDG_API_URL, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results: list[dict] = []
    # Abstract is primary
    if data.get("AbstractText") and data.get("AbstractURL"):
        results.append({
            "title": data.get("Heading") or query,
            "url": data["AbstractURL"],
            "snippet": data["AbstractText"],
        })
    for topic in data.get("RelatedTopics", [])[:limit]:
        if isinstance(topic, dict) and topic.get("Result"):
            # Result is HTML <a href="...">Title</a> - snippet
            html = topic.get("Result", "")
            m = re.search(r'href="([^"]+)"[^>]*>(.*?)</a>\s*[-—]?\s*(.*)', html)
            if m:
                url, title, snippet = m.groups()
                title = re.sub(r"<.*?>", "", title).strip()
                snippet = re.sub(r"<.*?>", "", snippet).strip()
                results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= limit:
                    break
    return results[:limit]


async def search_web(query: str, limit: int = 5) -> list[dict]:
    """Public: search web (free, no API key). Tries HTML then API fallback."""
    query = query.strip()
    if not query:
        return []
    limit = max(1, min(limit, 10))
    try:
        results = await _search_ddg_html(query, limit=limit)
        if results:
            return results
    except Exception:
        pass
    try:
        results = await _search_ddg_api(query, limit=limit)
        if results:
            return results
    except Exception:
        pass
    return []


async def fetch_page_text(url: str, max_chars: int = 8000) -> str:
    """Fetch URL and return stripped text (for care number / spec extraction). Free, no key."""
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text
    # Strip scripts/styles and tags
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<.*?>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


async def search_with_details(query: str, limit: int = 5, fetch_details: bool = False) -> list[dict]:
    """Search + optionally fetch top page to extract care numbers / specs."""
    results = await search_web(query, limit=limit)
    if not results:
        return results
    # Always extract care numbers from snippet/title (no fetch needed) — useful for Bosch etc.
    for r in results:
        snippet_numbers = extract_care_numbers(f"{r.get('title','')} {r.get('snippet','')}")
        if snippet_numbers:
            r["care_numbers"] = snippet_numbers
    if not fetch_details:
        return results
    # If fetch_details and query hints care, also try fetching top page to enrich
    lower_q = query.lower()
    wants_care = any(k in lower_q for k in ("care", "customer", "helpline", "support", "toll"))
    if wants_care:
        try:
            text = await fetch_page_text(results[0]["url"])
            fetched_numbers = extract_care_numbers(text)
            if fetched_numbers:
                # Merge, dedup
                existing = results[0].get("care_numbers") or []
                merged = existing + [n for n in fetched_numbers if n not in existing]
                results[0]["care_numbers"] = merged[:5]
                if not existing:
                    results[0]["snippet"] = f"Care: {', '.join(merged[:3])} — " + results[0]["snippet"]
        except Exception:
            pass
    return results
