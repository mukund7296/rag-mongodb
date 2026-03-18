"""
Central configuration — load from environment variables or .env file.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Anthropic / Claude
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    voyage_api_key: str    = os.getenv("VOYAGE_API_KEY", "")
    generation_model: str  = os.getenv("GENERATION_MODEL", "claude-sonnet-4-20250514")

    # MongoDB
    mongodb_uri: str        = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db: str         = os.getenv("MONGODB_DB", "rag_db")
    mongodb_collection: str = os.getenv("MONGODB_COLLECTION", "documents")
    vector_index_name: str  = os.getenv("VECTOR_INDEX_NAME", "vector_index")

    # Embedding
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "anthropic")
    embedding_model: str    = os.getenv("EMBEDDING_MODEL", "voyage-3")
    embedding_dim: int      = int(os.getenv("EMBEDDING_DIM", "1024"))

    # Retrieval
    retrieval_method: str  = os.getenv("RETRIEVAL_METHOD", "hybrid")  # vector | text | hybrid
    top_k: int             = int(os.getenv("TOP_K", "6"))
    rerank_top_k: int      = int(os.getenv("RERANK_TOP_K", "3"))
    min_score: float       = float(os.getenv("MIN_SCORE", "0.5"))

    # Chunking
    chunk_size: int    = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # Server
    port: int        = int(os.getenv("PORT", "5000"))
    flask_debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"


config = Config()
