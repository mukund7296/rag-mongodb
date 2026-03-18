# Advanced RAG with MongoDB Vector Search

> Production-grade Retrieval-Augmented Generation · Claude AI · Python · Flask

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-green) ![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248) ![Claude](https://img.shields.io/badge/Claude-Sonnet--4-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project implements a **production-ready RAG pipeline** that combines MongoDB Atlas Vector Search with Anthropic's Claude AI to deliver accurate, grounded answers from your own document corpus.

Rather than relying solely on an LLM's parametric memory, this pipeline retrieves the most relevant passages at query time and injects them into Claude's context window — dramatically reducing hallucinations and enabling answers from private or proprietary knowledge bases.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG PIPELINE                                    │
│                                                                          │
│  User Query                                                              │
│      │                                                                   │
│      ▼                                                                   │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐  │
│  │  Embedding   │    │  MongoDB Atlas    │    │   Claude Reranker    │  │
│  │  Service     │───▶│  Vector Search    │───▶│  (Cross-encoder)     │  │
│  │  (Voyage-3)  │    │  + Full-text      │    │                      │  │
│  └──────────────┘    └───────────────────┘    └──────────┬───────────┘  │
│                                                           │              │
│                                                           ▼              │
│                                               ┌──────────────────────┐  │
│                                               │  Claude Generation   │  │
│                                               │  (claude-sonnet-4)   │  │
│                                               └──────────────────────┘  │
│                                                           │              │
│                                                      Final Answer        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Retrieval Methods

| Method | Best For | How It Works |
|--------|----------|--------------|
| `vector` | Semantic / conceptual queries | HNSW kNN over Voyage-3 embeddings |
| `text` | Exact keyword queries | MongoDB Atlas Search with fuzzy matching |
| `hybrid` ✅ | General-purpose production | RRF fusion of vector + text ranked lists |

---

## Project Structure

```
rag_mongodb/
├── app.py                  # Flask REST API
├── demo.py                 # End-to-end demo script
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── src/
│   ├── rag_engine.py       # Core pipeline (embeddings, retrieval, generation)
│   └── config.py           # Centralised configuration
└── tests/
    └── test_rag.py         # Unit tests (mock-based, no API keys needed)
```

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/mukund7296/rag-mongodb.git
cd rag-mongodb
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your keys:
# ANTHROPIC_API_KEY=sk-ant-...
# MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
```

### 3. Create the vector search index (once)
```bash
python app.py &
curl -X POST http://localhost:5000/setup-index
```

### 4. Run the demo
```bash
python demo.py
```

---

## API Reference

### `POST /ingest`
Chunk, embed, and store documents in MongoDB.
```json
{
  "documents": [
    { "content": "MongoDB Atlas supports kNN vector search...", "metadata": { "source": "docs" } }
  ]
}
```

### `POST /query`
Query the RAG pipeline and receive grounded answers with citations.
```json
{
  "question": "How does hybrid search work?",
  "retrieval_method": "hybrid",
  "metadata_filter": { "category": "databases" }
}
```

### `GET /health`
Returns API health status.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_METHOD` | `hybrid` | `vector` \| `text` \| `hybrid` |
| `TOP_K` | `6` | Candidate docs to retrieve |
| `RERANK_TOP_K` | `3` | Docs after reranking |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `EMBEDDING_DIM` | `1024` | Voyage-3 dimension |

---

## Running Tests
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Tech Stack

- **[Anthropic Claude](https://anthropic.com)** — generation + cross-encoder reranking
- **[Voyage-3](https://www.voyageai.com)** — 1024-dim semantic embeddings
- **[MongoDB Atlas](https://www.mongodb.com/atlas)** — vector search (HNSW) + full-text search
- **[Flask](https://flask.palletsprojects.com)** — REST API layer
- **Reciprocal Rank Fusion** — hybrid search score fusion

---
