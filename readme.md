<div align="center">

# OpenMultiRAG
### Realtime Multimodal RAG System

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-black?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-f55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Cohere](https://img.shields.io/badge/Cohere-0050FF?style=for-the-badge&logo=cohere&logoColor=white)](https://cohere.ai/)
[![Langfuse](https://img.shields.io/badge/Langfuse-000000?style=for-the-badge&logo=langfuse&logoColor=white)](https://langfuse.com/)
[![MinIO](https://img.shields.io/badge/MinIO-C72E49?style=for-the-badge&logo=minio&logoColor=white)](https://min.io/)

---

**OpenMultiRAG** is a production-grade, realtime multimodal Retrieval-Augmented Generation (RAG) system. It seamlessly ingests complex PDF documents—extracting text, tables, and images—to provide a high-performance chat interface for deep technical queries and cross-document comparison.

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [End-to-End Data Flow](#3-end-to-end-data-flow)
4. [Engineering Deep Dive](#4-engineering-deep-dive)
5. [Engineering Challenges and Solutions](#5-engineering-challenges-and-solutions)
6. [Data Science Perspective](#6-data-science-perspective)
7. [Tools and Infrastructure](#7-tools-and-infrastructure)
8. [Environment Configuration](#8-environment-configuration)
9. [Performance Characteristics](#9-performance-characteristics)
10. [How to Run and Demo](#10-how-to-run-and-demo)
11. [Roadmap](#11-roadmap)

---

## 1) Executive Summary

OpenMultiRAG is a realtime multimodal RAG system that ingests PDFs, extracts text, tables, and images, indexes them, and provides a chat interface for querying across documents.

### Key Capabilities
- **Multimodal Extraction**: Deep parsing of text, complex tables, and high-resolution images.
- **Cross-Document Querying**: Analyze multiple documents simultaneously for comparative analysis.
- **Context-Aware Retrieval**: Hybrid search combining semantic (dense) and keyword (sparse) methods.
- **Source-Grounded Answers**: Reliable citations linked directly to document chunks and images.

### Design Principles
- **Scalable Architecture**: Background processing using RQ workers for heavy ingestion workloads.
- **Efficiency**: Incremental indexing ensures only new or changed content is processed.
- **Observability**: End-to-end tracing and monitoring for production reliability.
- **Safety**: Built-in guardrails for PII redaction and prompt injection detection.

---

## 2) System Architecture

The entire system is containerized for seamless deployment using Docker Compose.

![OpenMultiRAG Architecture](screenshots/architecture_minimal.png)

### Core Services
- **Frontend**: Responsive Streamlit interface for user interaction.
- **API Engine**: FastAPI backend managing orchestrations and lifecycles.
- **Execution Worker**: RQ (Redis Queue) based background processing.
- **Relational Storage**: PostgreSQL for metadata and structured data.
- **Message Broker**: Redis for task queuing and realtime SSE communication.
- **Vector Database**: Qdrant for high-dimensional embeddings and semantic caching.
- **Object Storage**: MinIO for persistent storage of extracted images and PDFs.

### Intelligent Integration
- **LLM Inference**: Groq-powered models for query intent, rewriting, and image captioning.
- **Embeddings & Reranking**: Cohere industrial-grade models for precision retrieval.
- **Tracing**: Langfuse for granular observability into the LLM logic chain.

---

## 3) End-to-End Data Flow

### Upload and Ingestion
**Endpoint**: `POST /upload`

1. **Validation**: Rigorous file type and size checks.
2. **Identity**: SHA-256 fingerprinting to prevent duplicate processing.
3. **Persistence**: Concurrent storage in local cache and object storage.
4. **Queueing**: Dispatch to background worker via Redis.
5. **Monitoring**: Realtime status streaming via Server-Sent Events (SSE).

### Background Processing
**Entry**: `worker.async_process_document`

- **Text Extraction**: High-fidelity parsing via PyMuPDF.
- **Table Extraction**: Heuristic-based parsing via pdfplumber.
- **Visual Analysis**: Image extraction with automated Groq-based captioning.
- **Storage**: Structured metadata including page numbers, file hashes, and visual paths.

### Incremental Indexing
- **Chunk Hashing**: Unique ID generation per content unit.
- **Delta Analysis**: Comparison against existing vector space.
- **Synchronization**: CRUD operations (Insert/Update/Delete) based on document state.

### Intelligent Query Pipeline
**Endpoint**: `POST /chat`

1. **Query Rewriting**: Context-aware expansion using conversation history.
2. **Intent Detection**: Advanced resolution of query scope (document vs. page).
3. **Semantic Cache**: High-speed lookup for previously resolved queries.
4. **Hybrid Retrieval**: Parallel execution of dense and sparse search.
5. **RRF & Reranking**: Reciprocal Rank Fusion combined with Cohere reranking.
6. **Synthesized Generation**: Multimodal answer generation with grounded citations.

---

## 4) Engineering Deep Dive

### API Architecture
- FastAPI lifecycle-managed dependencies for robust resource cleanup.
- Dedicated health check endpoints (shallow and deep) for orchestration monitoring.
- Native rate limiting and CORS enforcement for secure deployment.

### Async Execution
- Decoupled API/Worker architecture prevents UI blocking during heavy ingestion.
- Redis Pub/Sub pattern enables sub-second status updates to the frontend.

### Multimodal Pipeline
- Type-specific chunking strategies (Text vs. Table vs. Image).
- Automated summary generation for document-level grounding.
- Metadata enrichment for lightning-fast filtered retrieval.

### Advanced Retrieval
- **Dense**: Vector search for semantic conceptual matching.
- **Sparse**: BM25 for precise keyword and technical term matching.
- **Fusion**: Reciprocal Rank Fusion (RRF) to unify retrieval sources.

### Semantic Caching
- Vector-based cache lookup in Qdrant.
- Advanced scoping to prevent context leakage across different documents.
- Strict similarity thresholding to ensure answer accuracy.

### Guardrails & Safety
- **Validation**: Multistage file and input validation.
- **Security**: Prompt injection detection and PII redaction.
- **Veracity**: citation cross-validation before final response delivery.

---

## 5) Engineering Challenges and Solutions

| Challenge | Solution |
| :--- | :--- |
| **Query Rewriting Drift** | Input restriction to user-only messages for stable context. |
| **Cache Scope Conflicts** | Intent-first resolution and metadata scoping in Qdrant. |
| **Initial Resource Cost** | Implementation of shared singleton instances for cache clients. |
| **Unstructured Table Noise** | Applied stricter validation heuristics and layout-aware parsing. |
| **Race Conditions** | Individual execution of DDL statements for database migrations. |
| **Telemetry Gaps** | Implementation of explicit flush boundaries in worker signals. |

---

## 6) Data Science Perspective

### Retrieval-First RAG
We prioritize the "R" in RAG, focusing on:
- High-recall chunking strategies.
- Contextual relevance through hybrid search.
- Semantic integrity of extracted tables and captions.

### Evaluation Metrics
Evaluated using the **RAGAS** framework and `golden_dataset.json`:
- **Faithfulness**: Measuring hallucination rates.
- **Answer Relevancy**: Evaluating user intent alignment.
- **Context Precision/Recall**: Auditing the retrieval engine's accuracy.

---

## 7) Tools and Infrastructure

### Core Stack
- **Languages**: Python 3.10
- **Deployment**: Docker / Docker Compose
- **Frameworks**: FastAPI, Streamlit
- **Orchestration**: LangGraph, LangChain

### Database Layer
- **Relational**: PostgreSQL
- **Caching/Queue**: Redis
- **Vector Space**: Qdrant
- **Object Store**: MinIO

### Retrieval & AI
- **LLM**: Groq (Llama/Mixtral variants)
- **Embeddings**: Cohere Multi-lingual v3
- **Ranking**: BM25 (Sparse) + Cohere Rerank

---

## 8) Environment Configuration

Generate a `.env` file in the project root with the following parameters:

```bash
# --- LLM / AI APIs ---
GROQ_API_KEY="your_groq_key"
COHERE_API_KEY="your_cohere_key"
HUGGINGFACE_API_KEY="your_hf_key"

# --- Observability (Langfuse) ---
LANGFUSE_SECRET_KEY="your_secret"
LANGFUSE_PUBLIC_KEY="your_public"
LANGFUSE_BASE_URL="https://cloud.langfuse.com"

# --- Postgres ---
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=multirag
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# --- Redis ---
REDIS_BROKER_URL=redis://redis:6379/0
REDIS_CACHE_URL=redis://redis:6379/1

# --- Vector / Object Storage ---
QDRANT_URL=http://qdrant:6333
MINIO_ENDPOINT_URL=http://minio:9000
MINIO_ACCESS_KEY_ID=minioadmin
MINIO_SECRET_ACCESS_KEY=minioadmin
MINIO_BUCKET_NAME=multirag
```

---

## 9) Performance Characteristics

| Metric | Measured Value |
| :--- | :--- |
| **Average Response Latency** | ~770 ms |
| **P95 Latency** | < 1.2 s |
| **Cache Hit Response** | < 300 ms |
| **Retrieval Speed** | Sub-second |
| **SSE Realtime Offset** | Near-zero |

---

## 10) How to Run and Demo

### Setup & Launch
1. Configure credentials in `.env`.
2. Build and start containers:
   ```bash
   docker compose up --build
   ```
3. Access Interfaces:
   - **User Interface**: `http://localhost:8501`
   - **API Documentation**: `http://localhost:8010/docs`

### Example Interactions
- "Summarize the key findings in the supply chain PDF."
- "What do the charts on page 3 indicate regarding revenue?"
- "Compare the risk factors between document A and document B."
- "Extract the quarterly growth data from the tables."

---

## 11) Roadmap

- [ ] Advanced visual reasoning using GPT-4o-mini/Gemini.
- [ ] Direct citation linking to PDF viewer overlay.
- [ ] Support for Audio and Video ingestion (Full Multimodal).
- [ ] Distributed worker scaling for enterprise workloads.
