import httpx
from ..config import get_settings


CHAT_PROMPT = """You are a helpful assistant that answers questions about household bills, invoices, and receipts.

Use the following context from the user's documents to answer their question.
If the context doesn't contain enough information, say "I don't have enough information from your documents to answer this."

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

            return {
                "answer": answer,
                "sources": sources,
            }
    except Exception:
        return {
            "answer": "Sorry, I couldn't process your question right now.",
            "sources": sources,
        }
