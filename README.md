# RAGtfm-BE

Containerized FastAPI backend for a multimodal RAG pipeline over PDF documents. The current system ingests PDFs, stores originals in S3-compatible object storage, tracks documents in PostgreSQL, partitions and chunks document content, captions image chunks, embeds content with Ollama, stores vectors in ChromaDB, retrieves with hybrid search, reranks results, and generates grounded answers.

## Architecture

- [x] FastAPI API container
- [x] PostgreSQL container for document catalog metadata
- [x] MinIO container for S3-compatible document object storage
- [x] ChromaDB server container for vector storage
- [x] Ollama container for local model serving
- [x] Docker Compose orchestration for local cloud-style development

Local service mapping:

```text
FastAPI   -> http://localhost:8000
MinIO UI  -> http://localhost:9001
MinIO S3  -> http://minio:9000 inside Docker
Postgres  -> postgres:5432 inside Docker
Chroma    -> chroma:8000 inside Docker
Ollama    -> ollama:11434 inside Docker
```

## Reference Dev System 
```text
CPU : Intel Core i9 14900K 24C/32T
RAM : 64GB 
GPU : Nvidia Geforce RTX 4090 24GB
```

## Available Features

- [x] FastAPI application setup
- [x] PDF upload endpoint
- [x] PDF file validation
- [x] Upload size limit handling
- [x] S3-compatible PDF persistence with MinIO
- [x] PostgreSQL document catalog
- [x] Document listing endpoint
- [x] Document detail endpoint
- [x] Document deletion across PostgreSQL, ChromaDB, and object storage
- [x] PDF partitioning with Unstructured
- [x] High-resolution PDF parsing
- [x] Table structure inference
- [x] Image extraction from PDFs
- [x] Source order tracking for partitioned elements
- [x] Source location metadata with page number and coordinates
- [x] Text element chunking by detected titles
- [x] Separate handling for text, table, and image chunks
- [x] Table metadata preservation with HTML representation when available
- [x] Image payload extraction for captioning
- [x] Image captioning with Ollama vision model
- [x] Embedding text preparation for text chunks
- [x] Embedding text preparation for table chunks
- [x] Caption-based embedding preparation for image chunks
- [x] Text embeddings with Ollama `nomic-embed-text`
- [x] Persistent vector storage with ChromaDB server
- [x] Chroma metadata serialization for nested source metadata
- [x] Dense vector retrieval from ChromaDB
- [x] BM25 keyword retrieval with LangChain `BM25Retriever`
- [x] Hybrid retrieval with Reciprocal Rank Fusion
- [x] Cross-encoder reranking with `sentence-transformers`
- [x] Retrieval endpoint for inspecting retrieved chunks
- [x] RAG query endpoint for generated answers
- [x] Source metadata returned with generated answers
- [x] Retrieval diagnostics including dense rank, BM25 rank, RRF score, rerank score, and rerank rank
- [x] Evaluation endpoint for test metrics
- [x] Test dataset support for PDF question-answer evaluation
- [x] Metrics for recall@k, hitrate@k, MRR, faithfulness, and relevance

## Current API Endpoints

- [x] `POST /ingest/pdf` uploads, processes, embeds, and stores a PDF
- [x] `GET /documents` lists ingested documents
- [x] `GET /documents/{document_id}` returns document metadata
- [x] `DELETE /documents/{document_id}` deletes a document and associated resources
- [x] `GET /retrieve/chunks` retrieves relevant chunks for a query
- [x] `POST /rag/query` retrieves context and generates an answer
- [x] `GET /test/metrics` runs the current evaluation dataset

## Current Retrieval Pipeline

- [x] Embed the query with Ollama
- [x] Retrieve dense candidates from ChromaDB
- [x] Retrieve lexical candidates with BM25
- [x] Merge dense and BM25 candidates with Reciprocal Rank Fusion
- [x] Rerank fused candidates with a cross-encoder
- [x] Send final context chunks to the generation model

## Current Limitations

- [ ] No duplicate upload detection yet
- [ ] No user, tenant, or access-control layer yet
- [ ] No background ingestion job queue yet
- [ ] No database migrations yet
- [ ] No health checks yet
- [ ] Faithfulness and relevance metrics are lexical approximations, not LLM-as-judge metrics

## Requirements

- Docker
- Docker Compose
- NVIDIA GPU container support if using Ollama with GPU acceleration

Required Ollama models inside the Ollama container:

- [x] `nomic-embed-text` for embeddings
- [x] A vision-capable model for image captioning
- [x] A chat model for answer generation

## Run With Docker Compose

```bash
docker compose up -d --build
```

Pull required models into the Ollama container:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull gemma4:latest
```

Then open the API docs:

```text
http://localhost:8000/docs
```

MinIO console:

```text
http://localhost:9001
```

Default local MinIO credentials:

```text
minioadmin / minioadmin
```

The `minio-init` service creates the `ragtfm-documents` bucket automatically.

## Cloud Equivalents

This local stack mirrors common cloud deployment boundaries:

- [x] FastAPI container -> Cloud Run, ECS, AKS, GKE, or Azure Container Apps
- [x] PostgreSQL container -> managed PostgreSQL such as RDS or Cloud SQL
- [x] MinIO -> S3, GCS, Azure Blob, or another object store
- [x] ChromaDB server -> managed vector database or standalone vector DB service
- [x] Ollama -> dedicated model-serving endpoint, GPU VM, vLLM, or managed LLM provider
