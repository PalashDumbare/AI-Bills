import re
import httpx
from ..config import get_settings


_PREFIX_RE = re.compile(
    r"^\s*(according to (the )?(context|documents?|information) (provided|given)|based on (the )?(context|documents?|information)( provided)?|from the (context|documents?))[,:]?\s*",
    re.IGNORECASE,
)


def _clean_answer(text: str) -> str:
    cleaned = _PREFIX_RE.sub("", text, count=1).lstrip()
    # Capitalize first letter if stripped left it lowercase
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


CHAT_PROMPT = """You are a helpful assistant that answers questions about household bills, invoices, and receipts.

Use the following context from the user's documents to answer their question.
If the context doesn't contain enough information, say "I don't have enough information from your documents to answer this."
Answer directly and naturally. Do NOT start with phrases like "According to the context provided," "Based on the context," or "According to the documents".

Context:
{context}

Question: {question}

Answer:"""


async def chat_with_documents(
    question: str,
    context_chunks: list[dict],
) -> dict:
    """Send question + context to LLM and return answer with sources."""
    settings = get_settings()

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
    prompt = CHAT_PROMPT.format(context=context, question=question)

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
            answer = result.get("response", "").strip()
            # Strip leaky prefixes like "According to the context provided, ..."
            answer = _clean_answer(answer)

            return {
                "answer": answer,
                "sources": sources,
            }
    except Exception:
        return {
            "answer": "Sorry, I couldn't process your question right now.",
            "sources": sources,
        }
