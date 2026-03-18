"""
Flask REST API for the Advanced RAG Pipeline.
Exposes endpoints for document ingestion and querying.
"""

import os
import logging
from flask import Flask, request, jsonify
from src.rag_engine import (
    RAGPipeline, MongoVectorStore, EmbeddingService,
    DocumentProcessor, Reranker, Document
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─── Initialise pipeline on startup ──────────────────────────────────────────

MONGO_URI    = os.getenv("MONGODB_URI", "mongodb+srv://<user>:<pass>@cluster.mongodb.net/")
DB_NAME      = os.getenv("MONGODB_DB", "rag_db")
COLLECTION   = os.getenv("MONGODB_COLLECTION", "documents")
EMBED_DIM    = int(os.getenv("EMBEDDING_DIM", "1024"))

embedding_service = EmbeddingService(provider="anthropic")

vector_store = MongoVectorStore(
    connection_string=MONGO_URI,
    database=DB_NAME,
    collection=COLLECTION,
)

reranker = Reranker()

pipeline = RAGPipeline(
    vector_store=vector_store,
    embedding_service=embedding_service,
    reranker=reranker,
    retrieval_method="hybrid",
    top_k=6,
    rerank_top_k=3,
)

processor = DocumentProcessor(chunk_size=512, chunk_overlap=64)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "pipeline": "ready"})


@app.post("/ingest")
def ingest():
    """
    Ingest raw documents into MongoDB.

    Body:
      {
        "documents": [
          {"content": "...", "metadata": {"source": "...", "category": "..."}}
        ]
      }
    """
    body = request.get_json(force=True)
    raw_docs = body.get("documents", [])
    if not raw_docs:
        return jsonify({"error": "No documents provided"}), 400

    try:
        chunks = processor.process_documents(raw_docs, embedding_service)
        count  = vector_store.upsert_documents(chunks)
        return jsonify({"ingested_chunks": count, "source_docs": len(raw_docs)})
    except Exception as e:
        logger.exception("Ingestion failed")
        return jsonify({"error": str(e)}), 500


@app.post("/query")
def query():
    """
    Query the RAG pipeline.

    Body:
      {
        "question": "What is vector search?",
        "metadata_filter": {"category": "databases"},   // optional
        "retrieval_method": "hybrid"                     // optional override
      }
    """
    body     = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    metadata_filter = body.get("metadata_filter")

    # Allow per-request override of retrieval method
    method = body.get("retrieval_method")
    if method and method in ("vector", "text", "hybrid"):
        pipeline.retrieval_method = method

    try:
        response = pipeline.query(question, metadata_filter)
        return jsonify({
            "answer":  response.answer,
            "query":   response.query,
            "method":  response.retrieval_method,
            "tokens":  response.tokens_used,
            "sources": [
                {
                    "content":  r.document.content[:300],
                    "score":    round(r.score, 4),
                    "method":   r.retrieval_method,
                    "metadata": r.document.metadata,
                }
                for r in response.sources
            ],
        })
    except Exception as e:
        logger.exception("Query failed")
        return jsonify({"error": str(e)}), 500


@app.post("/setup-index")
def setup_index():
    """Create the Atlas Vector Search index (run once)."""
    try:
        vector_store.create_vector_index(num_dimensions=EMBED_DIM, similarity="cosine")
        return jsonify({"status": "index created", "dimensions": EMBED_DIM})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=port)
