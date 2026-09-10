import os
import uuid
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import Document

EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "..", "extracted")
os.makedirs(EXTRACTED_DIR, exist_ok=True)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _get_extension(filename: str | None, content_type: str | None) -> str:
    """Determine file extension from filename or content type."""
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext:
            return ext

    type_map = {
        "application/pdf": ".pdf",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
    }
    return type_map.get(content_type, ".pdf")


async def save_uploaded_file(file: UploadFile, user_id: str, session: AsyncSession):
    """Save uploaded PDF/image with generated unique ID and create DB record."""
    document_id = str(uuid.uuid4())
    original_filename = file.filename or "unknown"
    ext = _get_extension(file.filename, file.content_type)

    file_path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    with open(file_path, "wb") as f:
        f.write(contents)

    doc = Document(
        id=document_id,
        user_id=user_id,
        filename=original_filename,
        file_path=file_path,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    return {
        "document_id": document_id,
        "filename": original_filename,
        "file_type": file.content_type or "application/octet-stream",
        "user_id": user_id,
    }


def extract_document_text(document_id: str):
    """Extract text from uploaded document (Milestone 2)."""
    file_path = None

    if os.path.isdir(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(document_id):
                file_path = os.path.join(UPLOAD_DIR, fname)
                break

    if not file_path:
        raise HTTPException(status_code=404, detail="Document not found")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = _extract_from_pdf(file_path)
    else:
        text = _extract_from_image(file_path)

    extracted_path = os.path.join(EXTRACTED_DIR, f"{document_id}_extracted.txt")
    with open(extracted_path, "w") as f:
        f.write(text)

    return {
        "document_id": document_id,
        "extracted_text": text,
        "extracted_path": extracted_path,
    }


def _extract_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber (selectable text) or OCR (scanned)."""
    import pdfplumber
    from PIL import Image
    import pytesseract

    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(text)
            else:
                image = page.to_image().pil_image
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    text_parts.append(ocr_text)
    return "\n".join(text_parts)


def _extract_from_image(file_path: str) -> str:
    """Extract text from image file using OCR."""
    from PIL import Image
    import pytesseract

    image = Image.open(file_path)
    return pytesseract.image_to_string(image)
