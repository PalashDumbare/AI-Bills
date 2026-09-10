# AI Bills — Intelligent Document Processing

An AI-powered system that extracts, classifies, and structures data from bills, invoices, and receipts using LLM + regex fallback.

## What It Does

1. **Upload** — Accept PDFs and images of bills/invoices
2. **Extract** — Pull text via PDF parsing (pdfplumber) or OCR (Tesseract)
3. **Structure** — Classify document type and extract fields using Ollama LLM with regex fallback
4. **Store** — Save structured data in PostgreSQL

## Extracted Fields

| Field | Appliance Invoice | Bill |
|-------|------------------|------|
| Brand / Provider | ✅ | ✅ |
| Product / Bill Type | ✅ | ✅ |
| Model | ✅ | — |
| Purchase Date | ✅ | — |
| Amount | ✅ | ✅ |
| Warranty | ✅ | — |
| Due Date | — | ✅ |

## Tech Stack

- **Backend:** FastAPI (async)
- **Database:** PostgreSQL + SQLAlchemy (async) + Alembic
- **LLM:** Ollama (llama3.2) — runs locally, no API costs
- **OCR:** Tesseract + Pillow
- **PDF:** pdfplumber
- **Testing:** pytest + httpx (45 tests)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup database
createdb ai_bills
alembic upgrade head

# 3. Start Ollama (optional, for LLM extraction)
ollama pull llama3.2
ollama serve

# 4. Run server
uvicorn app.main:app --reload

# 5. Open docs
open http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload PDF/image |
| POST | `/documents/{id}/extract` | Extract text |
| POST | `/documents/{id}/structure` | Classify + structure data |

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI routes
│   ├── config.py             # Settings (.env)
│   ├── database.py           # SQLAlchemy async engine
│   ├── models/               # ORM models
│   ├── schemas/              # Pydantic schemas
│   └── services/
│       ├── document.py       # Upload + extraction
│       ├── extractor.py      # Regex field extraction
│       └── llm_client.py     # Ollama LLM client
├── tests/                    # 45 passing tests
├── alembic/                  # DB migrations
└── pyproject.toml            # pytest config
```

## License

MIT
