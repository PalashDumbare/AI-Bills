import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database import get_session, init_db
from .models import Document, Appliance, Bill
from .schemas import StructureRequest, StructureResponse, IndexResponse, ChatRequest, ChatResponse, WebSearchRequest, WebSearchResponse
from .services.extractor import extract_structured_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Warm BM25 from disk / Qdrant so restart doesn't lose hybrid search (BM25 was in-memory)
    try:
        from .services.bm25_search import get_bm25_index

        get_bm25_index()
    except Exception:
        pass
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/documents")
async def list_documents(
    user_id: str | None = None,
    skip: int = 0,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List documents, optionally filtered by user_id. Ordered by newest first."""
    query = select(Document).order_by(Document.created_at.desc())
    if user_id:
        query = query.where(Document.user_id == user_id)
    query = query.offset(skip).limit(limit)
    result = await session.execute(query)
    docs = result.scalars().all()
    return [
        {
            "id": d.id,
            "user_id": d.user_id,
            "filename": d.filename,
            "file_path": d.file_path,
            "document_type": d.document_type,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@app.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get single document with structured data (appliance/bill) if available."""
    result = await session.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    structured = None
    if doc.document_type == "appliance_invoice":
        r = await session.execute(select(Appliance).where(Appliance.document_id == document_id))
        structured = r.scalar_one_or_none()
    elif doc.document_type == "bill":
        r = await session.execute(select(Bill).where(Bill.document_id == document_id))
        structured = r.scalar_one_or_none()
    else:
        # Fallback: try both
        r = await session.execute(select(Appliance).where(Appliance.document_id == document_id))
        structured = r.scalar_one_or_none()
        if not structured:
            r = await session.execute(select(Bill).where(Bill.document_id == document_id))
            structured = r.scalar_one_or_none()

    return {
        "id": doc.id,
        "user_id": doc.user_id,
        "filename": doc.filename,
        "file_path": doc.file_path,
        "document_type": doc.document_type,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "structured_data": structured,
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    """Receive PDF/image, validate, generate unique ID, save file, create DB record."""
    from .services.document import save_uploaded_file

    return await save_uploaded_file(file, user_id, session)


@app.post("/documents/{document_id}/extract")
async def extract_document(document_id: str):
    """Extract text from uploaded document (Milestone 2)."""
    from .services.document import extract_document_text

    return extract_document_text(document_id)


@app.post("/documents/{document_id}/structure", response_model=StructureResponse)
async def structure_document(
    document_id: str,
    request: StructureRequest,
    session: AsyncSession = Depends(get_session),
):
    """Extract structured data from text and store in PostgreSQL."""
    # Check if document exists in DB
    result = await session.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Upload first.")

    # Check if already structured
    if doc.document_type:
        existing = await session.execute(
            select(Appliance).where(Appliance.document_id == document_id)
        )
        appliance = existing.scalar_one_or_none()
        if appliance:
            return StructureResponse(
                document_id=document_id,
                document_type="appliance_invoice",
                structured_data=appliance,
            )

        existing = await session.execute(
            select(Bill).where(Bill.document_id == document_id)
        )
        bill = existing.scalar_one_or_none()
        if bill:
            return StructureResponse(
                document_id=document_id,
                document_type="bill",
                structured_data=bill,
            )

    # Read extracted text
    extracted_path = os.path.join(
        os.path.dirname(__file__), "extracted", f"{document_id}_extracted.txt"
    )
    if not os.path.exists(extracted_path):
        raise HTTPException(status_code=404, detail="Extracted text not found. Run /extract first.")

    with open(extracted_path) as f:
        text = f.read()

    # Try LLM extraction first, fallback to regex
    structured = None
    try:
        from .services.llm_client import extract_with_llm
        structured = await extract_with_llm(text)
    except Exception:
        pass

    if not structured:
        structured = extract_structured_data(text)

    if not structured:
        raise HTTPException(status_code=422, detail="Could not extract structured data from text.")

    # Update existing Document record
    doc.document_type = structured.get("document_type")
    doc.user_id = request.user_id
    session.add(doc)

    if structured["document_type"] == "appliance_invoice":
        entry = Appliance(
            document_id=document_id,
            brand=structured.get("brand"),
            product=structured.get("product"),
            model=structured.get("model"),
            purchase_date=structured.get("purchase_date"),
            amount=structured.get("amount"),
            warranty_months=structured.get("warranty_months"),
            warranty_expiry=structured.get("warranty_expiry"),
        )
    else:
        entry = Bill(
            document_id=document_id,
            provider=structured.get("provider"),
            bill_type=structured.get("bill_type"),
            amount=structured.get("amount"),
        )

    session.add(entry)
    await session.commit()
    await session.refresh(entry)

    return StructureResponse(
        document_id=document_id,
        document_type=structured["document_type"],
        structured_data=entry,
    )


@app.post("/documents/{document_id}/index", response_model=IndexResponse)
async def index_document(document_id: str, session: AsyncSession = Depends(get_session)):
    """Chunk text, generate embeddings, store in Qdrant."""
    result = await session.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Upload first.")

    extracted_path = os.path.join(
        os.path.dirname(__file__), "extracted", f"{document_id}_extracted.txt"
    )
    if not os.path.exists(extracted_path):
        raise HTTPException(status_code=404, detail="Extracted text not found. Run /extract first.")

    with open(extracted_path) as f:
        text = f.read()

    from .services.chunker import chunk_text
    from .services.embeddings import embed_texts
    from .services.vector_store import index_chunks, delete_document_chunks

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks generated from text.")

    chunk_texts = [c.text for c in chunks]
    embeddings = embed_texts(chunk_texts)

    delete_document_chunks(document_id)
    num_indexed = index_chunks(
        document_id=document_id,
        texts=chunk_texts,
        embeddings=embeddings,
        metadata={"document_type": doc.document_type},
    )

    return IndexResponse(
        document_id=document_id,
        chunks_indexed=num_indexed,
        total_chunks=len(chunks),
    )


@app.get("/web-search", response_model=WebSearchResponse)
async def web_search_get(q: str, limit: int = 5, fetch_details: bool = False):
    """Web search (free, no API key) — fetch care numbers, model specs, etc. via DuckDuckGo."""
    from .services.web_search import search_with_details

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' is required")
    limit = max(1, min(limit, 10))
    results = await search_with_details(query=q.strip(), limit=limit, fetch_details=fetch_details)
    return WebSearchResponse(query=q.strip(), results=results, count=len(results))


@app.post("/web-search", response_model=WebSearchResponse)
async def web_search_post(request: WebSearchRequest):
    """Web search (free, no API key) — POST variant."""
    from .services.web_search import search_with_details

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    limit = max(1, min(request.limit, 10))
    results = await search_with_details(query=query, limit=limit, fetch_details=request.fetch_details)
    return WebSearchResponse(query=query, results=results, count=len(results))


def _is_web_intent(question: str) -> bool:
    q = question.lower()
    web_keywords = (
        "customer care", "care number", "helpline", "toll free", "toll-free",
        "support number", "contact number", "contact us", "customer support",
        "specification", "specs", "spec", "features", "model details",
    )
    return any(k in q for k in web_keywords)


async def _web_fallback(question: str):
    from .services.web_search import search_with_details
    from .services.chat import chat_with_documents

    web_results = await search_with_details(query=question, limit=3, fetch_details=True)
    if not web_results:
        return None
    web_chunks = [
        {
            "document_id": "web",
            "chunk_index": i,
            "text": f"{r['title']} — {r['snippet']} ({r['url']})",
            "score": 0.9 - i * 0.05,
        }
        for i, r in enumerate(web_results)
    ]
    resp = await chat_with_documents(question=question, context_chunks=web_chunks)
    return ChatResponse(
        answer=resp["answer"] + " (from web search)",
        sources=[
            {"document_id": "web", "chunk_index": s["chunk_index"], "text": s["text"], "score": s["score"]}
            for s in resp["sources"]
        ],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer questions about documents using hybrid RAG (BM25 + Dense). Auto-falls back to web search for care/spec queries."""
    from .services.hybrid_search import hybrid_search
    from .services.chat import chat_with_documents

    question = request.question.strip()
    should_try_web = request.use_web_search or _is_web_intent(question)

    results = hybrid_search(query=question, document_id=request.document_id, limit=5)

    if not results:
        if should_try_web:
            fallback = await _web_fallback(question)
            if fallback:
                return fallback
        return ChatResponse(
            answer="No relevant documents found. Please upload and index documents first.",
            sources=[],
        )

    response = await chat_with_documents(question=question, context_chunks=results)

    # If RAG says no info (or filtered to 0 sources) and question looks like web intent, fallback to web
    no_info = "don't have enough information" in response["answer"].lower() or not response["sources"]
    if no_info and should_try_web:
        fallback = await _web_fallback(question)
        if fallback:
            return fallback

    return ChatResponse(answer=response["answer"], sources=response["sources"])
