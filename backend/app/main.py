import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database import get_session, init_db
from .models import Document, Appliance, Bill
from .schemas import StructureRequest, StructureResponse, IndexResponse, ChatRequest, ChatResponse
from .services.extractor import extract_structured_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer questions about documents using hybrid RAG (BM25 + Dense)."""
    from .services.hybrid_search import hybrid_search
    from .services.chat import chat_with_documents

    results = hybrid_search(
        query=request.question,
        document_id=request.document_id,
        limit=5,
    )

    if not results:
        return ChatResponse(
            answer="No relevant documents found. Please upload and index documents first.",
            sources=[],
        )

    response = await chat_with_documents(
        question=request.question,
        context_chunks=results,
    )

    return ChatResponse(
        answer=response["answer"],
        sources=response["sources"],
    )
