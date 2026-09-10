# AI Bill & Document Assistant — Project Summary

## 1. Core Idea

Build an **Android application** where users can upload household bills/documents and later ask questions about them using AI.

The long-term product vision is:

> **A personal household document vault that understands bills, invoices, receipts, and warranties, and lets users chat with them.**

---

## 2. Supported Document Types

### Utility Bills
- Electricity
- Gas
- Water
- Internet
- Mobile

### Purchase Documents
- Refrigerator
- Washing machine
- TV
- Laptop
- AC
- Phone
- Furniture
- Other appliances/electronics

### Other
- Warranty cards
- Receipts
- Invoices
- Miscellaneous bills

For the **MVP**, start with a smaller set instead of supporting everything immediately.

---

## 3. Main User Experience

The user uploads a PDF/image:

```text
User
  ↓
Upload Bill
  ↓
Server
  ↓
Store Document
  ↓
Extract Information
  ↓
Create Searchable Representation
  ↓
User Can Ask Questions
```

Example appliance invoice:

```text
LG Washing Machine
Purchase Date: 15 Jan 2026
Amount: ₹32,999
Model: FHM1207
Warranty: 2 years
```

Possible questions:

- When did I buy my washing machine?
- How much did I pay?
- What is the model number?
- When does the warranty expire?

---

## 4. Product Differentiation

There are already apps for:
- Tracking electricity/gas bills
- Bill payment
- Warranty tracking
- Receipt storage

Therefore, the app should **not** be positioned simply as another bill reminder or warranty tracker.

The key differentiator should be:

> **"Chat with your household documents."**

For example, a user could upload:

```text
Electricity bill
Gas bill
Washing machine invoice
TV invoice
Laptop invoice
AC warranty
Furniture receipt
```

Then ask:

- How much was my electricity bill in August?
- Which appliances are still under warranty?
- When did I buy my TV?
- How much did I spend on appliances?
- What is my washing machine's model number?
- Show me my gas bills from the last 6 months.

The application becomes a **personal document intelligence system**.

---

# 5. RAG + Structured Data

We should **not use only RAG**.

There are two types of information.

## Structured Data

Example:

```json
{
  "product": "LG Washing Machine",
  "purchase_date": "2026-01-15",
  "amount": 32999,
  "warranty_months": 24
}
```

This should be stored in a **relational database** such as PostgreSQL.

## Unstructured Information

Example:

```text
The product is covered under a 2-year
comprehensive warranty from the date of purchase...
```

This is suitable for **RAG/vector search**.

Therefore:

```text
                User Question
                     ↓
                Query Router
                 /         \
                /           \
               ↓             ↓
          SQL Database    Vector Search
               │             │
               └──────┬──────┘
                      ↓
                     LLM
                      ↓
              Grounded Answer
```

---

# 6. Planned Architecture

Eventually:

```text
                         Android
                            ↓
                         FastAPI
                            │
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
        File Storage    PostgreSQL       Qdrant
             │              │              │
       Original Files   Structured      Chunks +
                         Data          Embeddings
             │              │              │
             └──────────────┴──────┬───────┘
                                   ↓
                              Query Router
                                   │
                          ┌────────┴────────┐
                          ↓                 ↓
                         SQL            Retrieval
                          │                 │
                          └────────┬────────┘
                                   ↓
                                Reranker
                                   ↓
                                  LLM
                                   ↓
                         Answer + Source
```

---

# 7. Qdrant

Use **Qdrant** as the vector database because RAG is a major part of the project.

Qdrant will store things such as:

```text
chunk_id
document_id
user_id
chunk_text
embedding
metadata
```

Example metadata:

```json
{
  "user_id": "user_123",
  "document_id": "doc_456",
  "document_type": "appliance_invoice",
  "brand": "LG",
  "purchase_date": "2026-01-15"
}
```

This allows retrieval to be filtered by:

- user
- document
- document type
- date
- brand
- other metadata

---

# 8. PostgreSQL

PostgreSQL will store structured information.

Possible tables:

```text
users
documents
bills
appliances
warranties
chat_sessions
chat_messages
```

Example `documents` table:

```text
documents
--------------------------------
id
user_id
file_name
file_path
document_type
created_at
```

Example `appliances` table:

```text
appliances
--------------------------------
id
document_id
brand
product
model
purchase_date
amount
warranty_period
warranty_expiry
```

---

# 9. Should We Store the Original PDF?

Technically:

> **No, RAG does not require the original PDF.**

We could do:

```text
PDF
 ↓
Extract Text
 ↓
Chunk
 ↓
Embedding
 ↓
Qdrant
 ↓
Discard PDF
```

However, for this product we decided to **retain the original document**.

Reasons:

### 1. User can view the original

If the AI says:

> Your washing machine cost ₹32,999.

The user should be able to tap:

> 📄 View original invoice

### 2. Reprocessing

If OCR or extraction improves:

```text
Original PDF
     ↓
New extraction model
     ↓
Better structured data
```

### 3. Source verification

The original document remains the **source of truth**.

Therefore:

```text
Original PDF → File/Object Storage
Structured Data → PostgreSQL
Chunks + Embeddings → Qdrant
```

---

# 10. Privacy and Data Isolation

Bills may contain sensitive information such as:

- Address
- Consumer number
- Phone number
- Invoice number
- Payment information
- Purchase history

Therefore, documents must be isolated per user.

Conceptually:

```text
User A
   ↓
Only User A's documents

User B
   ↓
Only User B's documents
```

The same isolation must be enforced in:
- File storage
- PostgreSQL
- Qdrant retrieval filters

Every document/chunk should be associated with a `user_id`.

---

# 11. Supabase

Supabase was considered because it can provide:

```text
Supabase
├── PostgreSQL
├── Storage
├── Authentication
└── APIs
```

Supabase's Free plan has storage/database/usage limits, so it is **not unlimited free storage**.

For a future cloud deployment, Supabase Storage could hold the original PDFs/images.

---

# 12. Firebase

Firebase was also considered.

Firebase can provide:

```text
Firebase
├── Authentication
├── Firestore
├── Storage
└── Notifications
```

However, Firebase Storage has a billing-plan consideration for newly provisioned buckets.

Since the requirement is:

> **Use free services and databases wherever possible.**

Supabase is currently preferred over Firebase for the main database/storage stack.

Firebase could still be considered later for Android-specific functionality such as notifications if useful.

---

# 13. MVP Storage Decision

For the **current MVP**, we do **not need Supabase or Firebase**.

The backend is running locally.

Store uploaded files in a local folder on the machine running FastAPI.

Architecture:

```text
Android
   ↓
FastAPI
   ↓
Local Folder
```

Example project structure:

```text
bill-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── services/
│   │   └── models/
│   │
│   └── storage/
│       └── documents/
│           ├── 8f3a1c2d.pdf
│           ├── 7b21e4a1.pdf
│           └── ...
│
└── android/
```

Use a generated unique ID for the actual filename rather than relying on the user's original filename.

The original filename can be stored as metadata.

---

# 14. Why Local Storage First?

We don't need to solve cloud storage, authentication, deployment, OCR, vector search, and LLM integration simultaneously.

The first goal is simply:

```text
Android
   ↓
Upload PDF/Image
   ↓
FastAPI
   ↓
Save File
   ↓
Return Document ID
```

No Qdrant, OCR, LLM, Supabase, or RAG yet.

This makes development and debugging incremental.

---

# 15. Development Roadmap

## Milestone 1 — File Upload

```text
Android
 ↓
FastAPI
 ↓
Local Storage
```

First endpoint:

```http
POST /documents/upload
```

It should:

1. Receive PDF/image
2. Validate the file
3. Generate a unique ID
4. Save the file
5. Return document information

Example response:

```json
{
  "document_id": "8f3a1c2d",
  "filename": "lg_washing_machine.pdf",
  "file_type": "application/pdf"
}
```

---

## Milestone 2 — Document Extraction

```text
PDF/Image
 ↓
OCR / PDF Parser
 ↓
Text
```

The goal is to convert the uploaded document into usable text.

---

## Milestone 3 — Structured Extraction

```text
Text
 ↓
LLM / Parser
 ↓
Structured JSON
 ↓
PostgreSQL
```

Example:

```json
{
  "document_type": "appliance_invoice",
  "brand": "LG",
  "product": "washing_machine",
  "purchase_date": "2026-01-15",
  "amount": 32999
}
```

---

## Milestone 4 — RAG

```text
Text
 ↓
Chunking
 ↓
Embeddings
 ↓
Qdrant
```

Store:
- chunks
- embeddings
- metadata

---

## Milestone 5 — Chat

```text
Question
 ↓
Retrieval
 ↓
Context
 ↓
LLM
 ↓
Answer + Source
```

Initial questions:

- What is the purchase date?
- How much did I pay?
- What is the model number?
- What is the warranty period?

---

## Milestone 6 — Hybrid Retrieval

Eventually:

```text
                Query
                  │
          ┌───────┴───────┐
          ↓               ↓
       BM25            Dense
       Search          Search
          │               │
          └───────┬───────┘
                  ↓
              Reranker
                  ↓
                Top-K
```

This will demonstrate more advanced RAG concepts.

---

## Milestone 7 — Warranty Management

Extract:

```text
Purchase Date
Warranty Period
```

Calculate:

```text
Purchase Date + Warranty Period
          ↓
Warranty Expiry
```

Example:

```text
LG Washing Machine
Warranty: Active
Expires: 15 Jan 2028
```

---

## Milestone 8 — Analytics

Eventually support questions such as:

- What was my average electricity bill?
- Which month had the highest electricity consumption?
- How much did I spend on appliances?
- Show my gas bills for the last 6 months.

These questions can combine structured SQL queries with RAG when necessary.

---

# 16. Planned Technology Stack

| Component | Choice |
|---|---|
| Android | Kotlin + Jetpack Compose |
| Backend | Python + FastAPI |
| Database | PostgreSQL |
| Vector DB | Qdrant |
| Initial File Storage | Local folder |
| Future File Storage | Supabase Storage |
| Authentication | To be finalized |
| OCR | Free/local solution |
| Embeddings | Free/local model |
| LLM | Free/local or suitable free-tier option |
| Version Control | GitHub |

The final service choices should be checked against current free-tier limits before cloud deployment.

---

# 17. Final Data Architecture

The core separation should be:

```text
                    DOCUMENT
                       │
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Original        Structured       Unstructured
     File             Data             Text
       │               │                │
       ↓               ↓                ↓
 File Storage      PostgreSQL         Qdrant
                                      │
                                  Embeddings
                                      │
                                      ↓
                                     RAG
```

The LLM sits **on top of these systems** rather than acting as the database.

---

# 18. Immediate Next Step

Do **not** start with RAG or the Android UI.

Start with:

> **Milestone 1: FastAPI file upload.**

The first working flow should be:

```text
Android
   ↓
Upload PDF/Image
   ↓
FastAPI
   ↓
Save to local folder
   ↓
Return document ID
```

Once this works, build the document-processing pipeline on top of it.

---

# 19. Overall Project Vision

The final application should evolve into:

```text
                  Household Documents
                         │
                         ↓
                Document Understanding
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      Electricity       Gas         Appliances
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                 Structured Data
                         +
                  Document Chunks
                         ↓
                    PostgreSQL
                         +
                      Qdrant
                         ↓
                  Query / RAG Layer
                         ↓
                       LLM
                         ↓
                 Answer + Evidence
                         ↓
                    Android App
```

**Core product statement:**

> **An AI-powered personal document wallet that understands bills, invoices, receipts, and warranties, allowing users to search and chat with their household documents.**
