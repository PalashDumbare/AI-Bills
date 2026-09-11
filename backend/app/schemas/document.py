from datetime import date
from decimal import Decimal
from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    user_id: str


class StructureRequest(BaseModel):
    user_id: str


class ApplianceResponse(BaseModel):
    id: str
    document_id: str
    brand: str | None
    product: str | None
    model: str | None
    purchase_date: date | None
    amount: Decimal | None
    warranty_months: int | None
    warranty_expiry: date | None

    model_config = {"from_attributes": True}


class BillResponse(BaseModel):
    id: str
    document_id: str
    provider: str | None
    bill_type: str | None
    amount: Decimal | None
    due_date: date | None
    billing_period: str | None

    model_config = {"from_attributes": True}


class StructureResponse(BaseModel):
    document_id: str
    document_type: str
    structured_data: ApplianceResponse | BillResponse


class IndexResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    total_chunks: int


class ChatRequest(BaseModel):
    question: str
    document_id: str | None = None
    use_web_search: bool = False


class WebSearchRequest(BaseModel):
    query: str
    limit: int = 5
    fetch_details: bool = False


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    care_numbers: list[str] | None = None


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult]
    count: int


class SourceChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class DocumentListItem(BaseModel):
    id: str
    user_id: str
    filename: str
    file_path: str
    document_type: str | None
    created_at: str

    model_config = {"from_attributes": True}


class DocumentDetailResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    file_path: str
    document_type: str | None
    created_at: str
    structured_data: ApplianceResponse | BillResponse | None = None

    model_config = {"from_attributes": True}
