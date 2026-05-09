# RAGtfm-BE

FastAPI backend for a local RAG pipeline over PDF documents. The current system ingests PDFs, partitions and chunks document content, captions image chunks, embeds content locally with Ollama, stores vectors in ChromaDB, retrieves with hybrid search, reranks results, and generates grounded answers.

## Available Features

- [x] FastAPI application setup
- [x] PDF upload endpoint
- [x] PDF file validation
- [x] Upload size limit handling
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
- [x] Local text embeddings with Ollama `nomic-embed-text`
- [x] Persistent vector storage with ChromaDB
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

- [ ] Original uploaded PDFs are not persisted
- [ ] No document catalog or document listing endpoint yet
- [ ] No document deletion endpoint yet
- [ ] No duplicate upload detection yet
- [ ] No user, tenant, or access-control layer yet
- [ ] Image files are not stored separately after ingestion
- [ ] ChromaDB is used as local persistence, not a managed production vector service
- [ ] Faithfulness and relevance metrics are lexical approximations, not LLM-as-judge metrics

## Local Requirements

- Python
- FastAPI
- Uvicorn
- Unstructured
- LangChain Core
- LangChain Community
- ChromaDB
- Ollama
- Sentence Transformers

Required local Ollama models:

- [x] `nomic-embed-text` for embeddings
- [x] A vision-capable model for image captioning
- [x] A chat model for answer generation

## Run Locally

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

