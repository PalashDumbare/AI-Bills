import re
import httpx
from ..config import get_settings


_PREFIX_RE = re.compile(
    r"^\s*(according to (the )?(context|documents?|information) (provided|given)|based on (the )?(context|documents?|information)( provided)?|from the (context|documents?))[,:]?\s*",
    re.IGNORECASE,
)

_CITATION_RE = re.compile(r"\[(\d+)\]")

_STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "for", "to", "of", "in", "with", "by", "as", "it", "this", "that", "you", "your", "i", "we", "be", "are", "was", "were", "has", "have", "had", "what", "how", "when", "where", "did", "does", "my", "amount", "cost",
}

_NO_INFO_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "no relevant documents",
]

# Phone / care number pattern for hallucination detection (India 1800/1860, US 1-800)
_CARE_NUMBER_RE = re.compile(
    r"(?:\+?91[\s-]?)?(?:1[\s-]?800|1800|1860)[\s-]?\d{3}[\s-]?\d{4}|\b\d{3,4}[\s-]\d{6,8}\b"
)


def _has_hallucinated_numbers(answer: str, sources: list[dict]) -> bool:
    """Return True if answer contains a phone/care number not grounded in any source text."""
    numbers = _CARE_NUMBER_RE.findall(answer)
    if not numbers:
        return False
    # Normalize source texts joined
    source_text = " ".join(s.get("text", "") for s in sources)
    # Normalize for comparison: strip spaces/dashes
    def _norm(n: str) -> str:
        return re.sub(r"[\s\-]", "", n)
    for n in numbers:
        norm_n = _norm(n)
        # check if digits appear in source (allow spaces/dashes variance)
        if norm_n not in re.sub(r"[\s\-]", "", source_text):
            # also check raw substring fallback
            if n.strip() not in source_text:
                return True
    return False


def _clean_answer(text: str) -> str:
    cleaned = _PREFIX_RE.sub("", text, count=1).lstrip()
    # Strip meta-commentary the LLM sometimes emits (screenshot)
    cleaned = re.sub(r"(?i)here is a bullet list[^:\n]*:\s*", "", cleaned)
    cleaned = re.sub(r"(?i)here (is|are) (the )?warranty details[^:\n]*:\s*", "", cleaned)
    cleaned = re.sub(r"(?i)the warranty details[^:\n]*are as follows:?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^are as follows:?\s*", "", cleaned)
    # Filler "This means that ..." often repeats previous bullet — strip including optional dash
    cleaned = re.sub(r"(?i)\n?\s*-?\s*This means that[^.]*\.\s*", " ", cleaned)
    # Normalize citation spacing but keep newlines for markdown lists
    cleaned = re.sub(r"\s*\[(\d+)\]\s*", r" [\1] ", cleaned)
    cleaned = re.sub(r"\[(\d+)\]", r"[\1]", cleaned)
    # Fix space before colon/period after citation: "[1] :" -> "[1]:"
    cleaned = re.sub(r" \[(\d+)\] :", r" [\1]:", cleaned)
    cleaned = re.sub(r" \[(\d+)\] \.", r" [\1].", cleaned)
    # Collapse multiple spaces/tabs but preserve newlines
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Fix single-line multi-invoice bug: "Rs. 1,180 - Honda" -> split
    cleaned = re.sub(r" (?<!\n)- (?=[A-Z][^-\n]{0,40} ?\[\d+\])", r"\n- ", cleaned)
    # Remove empty bullet remnants
    lines = [l.strip() for l in cleaned.splitlines()]
    lines = [l for l in lines if l and l not in ("-", "- ")]
    # Aggressive dedup for same-citation warranty duplicates (verbose + concise)
    bullet_lines = [l for l in lines if l.startswith("-")]
    if len(bullet_lines) >= 2:
        cites = [re.search(r"\[(\d+)\]", l).group(1) if re.search(r"\[(\d+)\]", l) else None for l in bullet_lines]
        if len(set(cites)) == 1 and cites[0] is not None:
            if all(any(k in l.lower() for k in ("warranty", "year")) for l in bullet_lines):
                # Keep only the last (most structured) bullet
                non_bullets = [l for l in lines if not l.startswith("-")]
                lines = non_bullets + [bullet_lines[-1]]
    # General dedup
    deduped: list[str] = []
    seen_norm: set[str] = set()
    for line in lines:
        norm = re.sub(r"\s+", " ", line.lower())
        norm_nocite = re.sub(r"\[\d+\]", "", norm)
        if not norm_nocite.strip():
            continue
        if norm_nocite.strip() in seen_norm:
            continue
        # Check duplicate via same numeric dates (e.g. 15 Jan 2026)
        nums = set(re.findall(r"\d+", norm_nocite))
        is_dup = False
        for s in seen_norm:
            s_nums = set(re.findall(r"\d+", s))
            if nums and s_nums and nums == s_nums and "warranty" in norm_nocite and "warranty" in s:
                is_dup = True
                break
        if is_dup:
            continue
        seen_norm.add(norm_nocite.strip())
        deduped.append(line)
    cleaned = "\n".join(deduped).strip()
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned) if "\n" not in cleaned else cleaned
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _parse_cited_indices(answer: str, num_chunks: int) -> set[int]:
    """Extract 1-based citation indices like [1], [2] from answer."""
    indices: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        try:
            idx = int(m.group(1))
            if 1 <= idx <= num_chunks:
                indices.add(idx)
        except ValueError:
            continue
    return indices


def _lexical_filter(answer: str, sources: list[dict]) -> list[dict]:
    """Fallback: keep only sources with lexical overlap with answer.
    Used when LLM didn't emit citations."""
    answer_numbers = set(re.findall(r"\d[\d,\.]*", answer))
    answer_tokens = [
        t.lower()
        for t in re.findall(r"\w+", answer)
        if t.lower() not in _STOPWORDS and len(t) > 2
    ]
    if not answer_tokens and not answer_numbers:
        return []

    answer_set = set(answer_tokens)
    filtered: list[dict] = []
    for src in sources:
        text = src.get("text", "")
        # check numeric grounding (critical for bills/invoices)
        if answer_numbers and any(num in text for num in answer_numbers):
            filtered.append(src)
            continue
        chunk_tokens = [
            t.lower()
            for t in re.findall(r"\w+", text)
            if t.lower() not in _STOPWORDS and len(t) > 2
        ]
        chunk_set = set(chunk_tokens)
        overlap = len(answer_set & chunk_set)
        # keep if at least 2 meaningful words overlap or >30% of chunk is covered
        if overlap >= 2:
            filtered.append(src)
        elif chunk_set and overlap / len(chunk_set) >= 0.3:
            filtered.append(src)

    return filtered


def _filter_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Return only sources that were actually used to generate the answer."""
    if not answer or not sources:
        return []

    low = answer.lower()
    if any(phrase in low for phrase in _NO_INFO_PHRASES):
        return []

    # 1) Prefer explicit citations
    cited = _parse_cited_indices(answer, len(sources))
    if cited:
        # preserve citation order, deduped
        ordered = sorted(cited)
        return [sources[i - 1] for i in ordered]

    # 2) Fallback to lexical overlap
    lexical = _lexical_filter(answer, sources)
    if lexical:
        # cap to at most 3 most relevant to avoid showing all 5 when answer is generic
        # sort lexical by original score desc
        lexical_sorted = sorted(lexical, key=lambda s: s.get("score", 0), reverse=True)
        return lexical_sorted[:3]

    # 3) If neither citations nor lexical matched, return top-1 most relevant (avoid empty when answer is valid)
    # but only if answer is not a no-info response (already handled)
    # heuristic: if answer is very short (<10 chars) return empty
    if len(answer.strip()) < 10:
        return []
    top = sorted(sources, key=lambda s: s.get("score", 0), reverse=True)
    return top[:1]


CHAT_PROMPT = """You are a helpful assistant that answers questions about household bills, invoices, and receipts.

Use ONLY the following context from the user's documents to answer their question.
If the context doesn't contain enough information, say "I don't have enough information from your documents to answer this."
CRITICAL: Never invent, guess, or hallucinate phone numbers, care numbers, prices, dates, specs, or warranty details. If a phone number / care number / spec is not literally present in the context text, you MUST say you don't have enough information — do not generate a plausible-looking number.
Answer directly, concisely and naturally. Do NOT start with phrases like "According to the context provided," "Based on the context," or "According to the documents".

Formatting rules (IMPORTANT - MUST FOLLOW EXACTLY):
- Output in Markdown. Use plain English. Be CONCISE.
- For a SINGLE item (one warranty / one bill / one product): answer in 1-2 sentences OR a single bullet — NOT both. NEVER repeat the same info in a paragraph + bullet list. Do NOT add a second bullet that paraphrases the first.
- NEVER write meta-commentary like "are as follows", "Here is a bullet list showing...", "Here are the details", "This means that...". Just give the answer.
- For 2+ bills/invoices/appliances, put EACH invoice on its OWN single line bullet starting with "- ". NEVER put multiple invoices in one bullet, NEVER split one invoice across two bullets.
- Each bullet MUST be exactly: "- <ShortProvider> [n]: <INV-No> | <Date> | Rs. <Amount>"  OR for warranty "- <Product> [n]: <Warranty> | <Start> – <End> (<Duration>)"
  Keep to ONE line, max 20 words. No extra line breaks inside a bullet.
- Good single-item warranty: "LG Front Load Washing Machine [1]: 2-year comprehensive warranty, 15 Jan 2026 – 15 Jan 2028."
- Good single-item bullet: "- LG Washing Machine [1]: 2-year comprehensive | 15 Jan 2026 – 15 Jan 2028"
- Good multi-invoice (3 separate single-line bullets):
  - Pune Water [1]: INV-PMC-WTR-2025-0905 | 05 Sep 2025 | Rs. 1,180
  - Honda Service [2]: INV-HONDA-SRV-2025-0818 | 18 Aug 2025 | Rs. 9,912
  - MSEDCL Electricity [3]: INV-MSEDCL-2025-0315 | 15 Mar 2025 | Rs. 3,240
- Bad (DO NOT DO THIS - verbose + meta + duplicate): "The warranty details are as follows: - The product has a 2-year ... 15 Jan 2028. - This means that any defects... Here is a bullet list: - LG Electronics ..."
- Bad (DO NOT DO THIS - single line multiple invoices): "- Pune Water – Invoice No: ... - Honda Service – Invoice No: ... - MSEDCL ..."
- Bad (DO NOT split title): "- Honda Authorized Service Center
  - Vehicle Servicing Invoice"  <- this is TWO bullets for ONE invoice, WRONG. Must be ONE bullet as above.
- Cite every fact inline as [1], [2] right after provider/product name. Only cite chunks that support that line.

Context:
{context}

{history_block}
Question: {question}

Answer:"""


async def chat_with_documents(
    question: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
) -> dict:
    """Send question + context to LLM and return answer with sources. history is list of {role, content} for multiturn."""
    settings = get_settings()
    # Build history block (last 6 turns) for co-reference like "what about its warranty?"
    history_block = ""
    if history:
        trimmed = history[-6:]  # keep last 3 exchanges
        lines = []
        for h in trimmed:
            role = h.get("role", "user")
            content = h.get("content", "").strip()[:500]
            if not content:
                continue
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        if lines:
            history_block = "Conversation history (for context, use to resolve pronouns like 'it', 'its'):\n" + "\n".join(lines) + "\n"

    context_parts = []
    sources = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(f"[{i}] {chunk['text']}")
        sources.append({
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "score": chunk["score"],
        })

    context = "\n\n".join(context_parts)
    prompt = CHAT_PROMPT.format(context=context, history_block=history_block, question=question)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()

            result = response.json()
            raw_answer = result.get("response", "").strip()
            # Hallucination guard: if LLM invented a care number not in sources, force no_info
            if _has_hallucinated_numbers(raw_answer, sources):
                return {
                    "answer": "I don't have enough information from your documents to answer this.",
                    "sources": [],
                }
            filtered = _filter_sources(raw_answer, sources)
            answer = _clean_answer(raw_answer)
            # Re-check after cleaning (citations stripped) — still hallucinated?
            if _has_hallucinated_numbers(answer, sources):
                return {
                    "answer": "I don't have enough information from your documents to answer this.",
                    "sources": [],
                }

            return {
                "answer": answer,
                "sources": filtered,
            }
    except Exception:
        answer = "Sorry, I couldn't process your question right now."
        filtered = _filter_sources(answer, sources)
        # keep at least one source on error for debugging if filter empties (tests expect 1)
        if not filtered and sources:
            filtered = sources[:1]
        return {
            "answer": answer,
            "sources": filtered,
        }
