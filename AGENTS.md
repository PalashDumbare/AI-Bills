# AI Bill & Document Assistant — AGENTS.md

## Project Overview
An Android app where users upload household bills/documents and ask AI questions about them. Long-term vision: a personal document vault that understands bills, invoices, receipts, and warranties.

## Development Roadmap (Milestones 1–8)
Start with Milestone 1. Do not start with RAG or the Android UI.

- **Milestone 1** — File Upload: `POST /documents/upload` → FastAPI saves to local folder, returns document ID.
- **Milestone 2** — Document Extraction: PDF/image → OCR/PDF parser → text.
- **Milestone 3** — Structured Extraction: text → LLM/parser → structured JSON → PostgreSQL.
- **Milestone 4** — RAG: text → chunking → embeddings → Qdrant.
- **Milestone 5** — Chat: question → retrieval → context → LLM → answer + source.
- **Milestone 6** — Hybrid Retrieval: BM25 + dense search → reranker → top-k.
- **Milestone 7** — Warranty Management: extract purchase date + warranty period → calculate expiry.
- **Milestone 8** — Analytics: combine SQL queries with RAG for analytics questions.

## Technology Stack (from summary §16)
| Component | Choice |
|---|---|
| Android | Kotlin + Jetpack Compose |
| Backend | Python + FastAPI |
| Database | PostgreSQL |
| Vector DB | Qdrant |
| Initial File Storage | Local folder |
| Future File Storage | Supabase Storage |
| Auth | To be finalized |
| OCR | Free/local solution |
| Embeddings | Free/local model |
| LLM | Free/local or suitable free-tier option |
| Version Control | GitHub |

## Key Architectural Decisions
- **Local storage first** (summary §14): No Qdrant, OCR, LLM, Supabase, or RAG in MVP. Backend runs locally. Upload PDF/image → FastAPI → save to local folder → return document ID.
- **RAG + structured data hybrid** (summary §5, §9): Structured data stored in PostgreSQL; unstructured information suitable for RAG/vector search. Query router directs user questions to SQL database or vector search, with LLM grounding the answer.
- **Document retention** (summary §8): Original PDFs stored in file storage; structured data in PostgreSQL; chunks + embeddings in Qdrant. Original PDF is the source of truth for user verification.
- **Privacy/isolation** (summary §10): Every document/chunk associated with a `user_id`. Isolation enforced in file storage, PostgreSQL, and Qdrant retrieval filters.
- **Supabase preferred over Firebase** (summary §11–12): Supabase provides PostgreSQL + Storage + Auth; Firebase considered later for Android notifications only.

## Commands & Workflow
- Backend runs locally via FastAPI (no dedicated script defined yet — see `app/main.py` when code exists).
- Upload endpoint: `POST /documents/upload`
- Files stored locally under `backend/storage/documents/` with generated unique IDs (summary §13).
- Original filename stored as metadata.

## Repo-Specific Conventions
- Use generated unique IDs for filenames, not user's original filenames (summary §13).
- Store original filename as metadata.
- MVP does not need Supabase or Firebase — backend runs locally (summary §12).
- The LLM sits on top of the systems (PostgreSQL + Qdrant) rather than acting as the database (summary §17).