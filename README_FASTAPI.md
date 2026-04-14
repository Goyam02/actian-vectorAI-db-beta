# FastAPI RAG Application with Actian VectorAI DB

A production-ready FastAPI application demonstrating Retrieval-Augmented Generation (RAG) using Actian VectorAI DB.

## Features

- **Document Indexing**: Upload and chunk documents for semantic search
- **Vector Search**: Semantic similarity search using embeddings
- **RAG Query**: Retrieve relevant context and generate answers with LLM
- **Async Operations**: Full async/await support for high performance

## Quick Start

### 1. Start VectorAI DB

```bash
docker compose up -d
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file with your settings:

```env
# VectorAI DB
VECTOR_DB_URL=localhost:50051
COLLECTION_NAME=documents

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Azure OpenAI (Recommended)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini

# Or standard OpenAI (alternative)
# OPENAI_API_KEY=your-api-key
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
```

### 5. Access the API

- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### Index a Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here...", "metadata": {"source": "example"}}'
```

### Semantic Search
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is VectorAI DB?", "top_k": 5}'
```

### RAG Query
```bash
curl -X POST "http://localhost:8000/api/v1/rag" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key features?", "top_k": 5}'
```

## Project Structure

```
app/
├── main.py          # FastAPI application
├── config.py        # Settings and configuration
├── models.py        # Pydantic models
├── api/
│   └── routes.py    # API endpoints
└── services/
    ├── vector_db.py  # VectorAI DB service
    ├── embeddings.py # Embedding service
    └── rag.py        # RAG service
```
