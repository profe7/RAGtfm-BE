# RAGtfm-BE

[![CI](https://github.com/profe7/RAGtfm-BE/actions/workflows/ci.yml/badge.svg)](https://github.com/profe7/RAGtfm-BE/actions/workflows/ci.yml)

Containerized FastAPI backend for a multimodal RAG pipeline over PDF documents. The system ingests PDFs, stores originals in S3-compatible object storage, tracks documents in PostgreSQL, partitions and chunks content, captions image chunks, embeds content with Ollama, stores vectors in ChromaDB, retrieves with hybrid search, reranks results, and generates grounded answers. Users authenticate with JWT tokens and all documents are scoped per user.

## Architecture

- [x] FastAPI API container
- [x] PostgreSQL container for document catalog metadata
- [x] MinIO container for S3-compatible document object storage
- [x] ChromaDB server container for vector storage
- [x] Ollama container for local model serving
- [x] Redis container for message brokering
- [x] Celery worker container for background asynchronous document ingestion
- [x] Docker Compose orchestration for local cloud-style development

Local service mapping:

```text
FastAPI   -> http://localhost:8000
MinIO UI  -> http://localhost:9001
MinIO S3  -> http://minio:9000 inside Docker
Chroma    -> http://chroma:8000 inside Docker (exposed on host at :8001)
Postgres  -> postgres:5432 inside Docker (exposed on host at :5433)
Ollama    -> ollama:11434 inside Docker
Redis     -> redis:6379 inside Docker
Worker    -> Celery process inside Docker
```

## Reference Dev System

```text
CPU : Intel Core i9 14900K 24C/32T
RAM : 64GB
GPU : Nvidia Geforce RTX 4090 24GB
```

## Available Features

### Health Checks
- [x] Health check endpoints

### Auth
- [x] User registration and login with JWT bearer tokens
- [x] Per-user document scoping across all endpoints
- [x] Database migrations available with alembic

### Ingestion
- [x] PDF upload endpoint
- [x] PDF file validation and upload size limit handling
- [x] S3-compatible PDF persistence with MinIO
- [x] PostgreSQL document catalog with status tracking (`PENDING` → `READY` / `FAILED`)
- [x] Background ingestion via Celery task queue with Redis broker
- [x] PDF partitioning with Unstructured (high-resolution, table inference, image extraction)
- [x] Source order tracking and source location metadata (page number, coordinates)
- [x] Chunking by detected titles with separate handling for text, table, and image chunks
- [x] Table metadata preservation with HTML representation
- [x] Context-aware image captioning: surrounding-text context and Tesseract OCR are fed to the Ollama vision model, which returns a structured caption (verbatim visible text + interpretation)
- [x] Embedding text preparation combining caption and OCR for image chunks, so lexical search can match exact labels and values
- [x] Original image bytes persisted to S3 (`images/{document_id}/{chunk_id}.png`) for grounded multimodal answer generation
- [x] Text embeddings with Ollama `nomic-embed-text`
- [x] Persistent vector storage in ChromaDB with serialized nested metadata
- [x] Duplicate upload detection

### Retrieval
- [x] Dense vector retrieval from ChromaDB
- [x] BM25 keyword retrieval with LangChain `BM25Retriever`
- [x] Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- [x] Cross-encoder reranking with `sentence-transformers`
- [x] Optional filtering by specific `document_ids` in retrieval and RAG query
- [x] Retrieval diagnostics: dense rank, BM25 rank, RRF score, rerank score, rerank rank

### Generation
- [x] Streaming grounded answers from the Ollama generation model with inline source citations
- [x] Multimodal generation: the original image is attached for retrieved image chunks, so answers are grounded in the image rather than only its caption (toggle with `enable_image_generation`)
- [x] Faithfulness-hardened system prompt that answers "I do not know" rather than substituting a related-but-different fact

### Documents
- [x] Document listing endpoint (per-user)
- [x] Document detail endpoint
- [x] Document deletion across PostgreSQL, ChromaDB, and object storage

### Real-time Events
- [x] Server-Sent Events (SSE) stream for live document ingestion status (`PENDING` → `READY` / `FAILED`)
- [x] Per-user event channels backed by Redis pub/sub
- [x] Ticket-based SSE authentication: since `EventSource` cannot send an `Authorization` header, the client trades its JWT for a short-lived, single-use ticket, so the long-lived token never appears in a URL, access log, or browser history

### Evaluation
- [x] Evaluation endpoint for test metrics
- [x] Test dataset support for PDF question-answer evaluation
- [x] Metrics: recall@k, hitrate@k, MRR, faithfulness, and relevance

## Current API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/register` | Register a new user |
| `POST` | `/auth/login` | Login and receive a JWT token |
| `POST` | `/auth/logout` | Invalidate the current token |
| `POST` | `/ingest/pdf` | Upload and process a PDF (async) |
| `GET` | `/documents` | List all documents for the current user |
| `GET` | `/documents/{document_id}` | Get metadata for a specific document |
| `DELETE` | `/documents/{document_id}` | Delete a document and all associated resources |
| `POST` | `/documents/events/ticket` | Mint a short-lived, single-use ticket for the SSE stream |
| `GET` | `/documents/events` | Stream live document status updates via SSE (authenticated with a ticket) |
| `GET` | `/retrieve/chunks` | Retrieve relevant chunks for a query |
| `POST` | `/rag/query` | Retrieve context and generate a grounded answer |
| `GET` | `/test/metrics` | Run the evaluation dataset and return metrics |
| `GET` | `/health/live` | Check if the API is alive |
| `GET` | `/health/ready` | Check if the API can serve traffic |

## RAG Query Request

```json
{
  "query": "What is the methodology used?",
  "limit": 5,
  "document_ids": ["uuid-1", "uuid-2"]
}
```

`document_ids` is optional. If omitted, all documents belonging to the current user are searched.

## Retrieval Pipeline

1. Embed the query with Ollama `nomic-embed-text`
2. Retrieve dense candidates from ChromaDB (filtered by `user_id` and optionally `document_ids`)
3. Retrieve lexical candidates with BM25 (same filters applied)
4. Merge candidates with Reciprocal Rank Fusion (RRF)
5. Rerank fused candidates with a cross-encoder
6. Send final context chunks to the generation model, attaching the original image for image chunks so answers are grounded in the image and not only its caption

## Real-time Status Stream

Document ingestion runs asynchronously, so the client subscribes to an SSE stream to watch each document move from `PENDING` to `READY` / `FAILED` without polling.

1. `POST /documents/events/ticket` with the normal `Authorization: Bearer <jwt>` header
2. Receive a short-lived, single-use ticket (TTL configurable via `SSE_TICKET_TTL_SECONDS`, default 30s)
3. Open `EventSource("/documents/events?ticket=<ticket>")` — the ticket is validated and atomically consumed (`GETDEL`) so it cannot be replayed
4. Receive `document_status` events from the user's Redis pub/sub channel as ingestion progresses
5. On reconnect, mint a fresh ticket (the previous one is already consumed)

## Current Limitations

- [ ] Faithfulness and relevance metrics are lexical approximations, not LLM-as-judge metrics

## Requirements

- Docker
- Docker Compose
- NVIDIA GPU with container runtime support (recommended for Ollama)

Required Ollama models (pull into the Ollama container after first start):

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen3.5:latest
```

## Run With Docker Compose

```bash
docker compose up -d --build
```

Then open the interactive API docs:

```
http://localhost:8000/docs
```

MinIO console (default credentials: `minioadmin` / `minioadmin`):

```
http://localhost:9001
```

The `minio-init` service automatically creates the `ragtfm-documents` bucket on startup.

### Rebuild after code changes

```bash
docker compose up -d --build api worker
```

## Cloud Equivalents

| Local | Cloud |
|-------|-------|
| FastAPI container | Cloud Run, ECS, AKS, GKE, Azure Container Apps |
| PostgreSQL container | RDS, Cloud SQL, Azure Database for PostgreSQL |
| MinIO | S3, GCS, Azure Blob Storage |
| ChromaDB server | Managed vector DB or standalone vector DB service |
| Ollama | GPU VM, vLLM, managed LLM provider endpoint |
| Redis | ElastiCache, Memorystore, Azure Cache for Redis |
| Celery worker | Dedicated worker nodes, Cloud Run Jobs |