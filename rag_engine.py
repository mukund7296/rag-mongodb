"""
Advanced RAG Engine with MongoDB Vector Search
Supports multi-stage retrieval, reranking, and hybrid search.
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass, field

import anthropic
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: Optional[str] = None
    embedding: Optional[list] = None


@dataclass
class RetrievalResult:
    document: Document
    score: float
    retrieval_method: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievalResult]
    query: str
    retrieval_method: str
    tokens_used: int = 0


# ─── Embedding Service ────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Generates embeddings using Anthropic's API or OpenAI as a fallback.
    Swap out the provider here without touching the rest of the pipeline.
    """

    def __init__(self, provider: str = "anthropic", model: str = None):
        self.provider = provider
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "voyage-3"          # Anthropic's Voyage embedding model
            self.dimension = 1024
        elif provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "text-embedding-3-small"
            self.dimension = 1536
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if self.provider == "anthropic":
            return self._embed_anthropic(texts)
        return self._embed_openai(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_anthropic(self, texts: list[str]) -> list[list[float]]:
        # Uses Anthropic's Voyage API under the hood
        response = self.client.beta.messages.batches.create(
            requests=[{
                "custom_id": str(i),
                "params": {
                    "model": self.model,
                    "input": text,
                    "input_type": "document",
                }
            } for i, text in enumerate(texts)]
        )
        # For simplicity in direct usage, call embed directly via voyage
        # (actual Voyage integration shown in voyage_embedder.py)
        return self._voyage_embed(texts)

    def _voyage_embed(self, texts: list[str]) -> list[list[float]]:
        import voyageai
        vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        result = vo.embed(texts, model=self.model, input_type="document")
        return result.embeddings

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


# ─── MongoDB Vector Store ──────────────────────────────────────────────────────

class MongoVectorStore:
    """
    Wraps MongoDB Atlas Vector Search.
    Supports:
      - Vector (ANN) search
      - Full-text search
      - Hybrid search (vector + text score fusion)
      - Metadata filtering
    """

    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        index_name: str = "vector_index",
        embedding_field: str = "embedding",
        text_field: str = "content",
    ):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db[collection]
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.client.admin.command("ping")
            logger.info("MongoDB connection OK")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Cannot reach MongoDB: {e}")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def upsert_documents(self, documents: list[Document]) -> int:
        """Insert or update documents with their embeddings."""
        ops = []
        from pymongo import UpdateOne
        for doc in documents:
            if doc.embedding is None:
                logger.warning("Document %s has no embedding — skipping", doc.doc_id)
                continue
            filter_ = {"_id": doc.doc_id} if doc.doc_id else {}
            update = {"$set": {
                self.text_field: doc.content,
                self.embedding_field: doc.embedding,
                "metadata": doc.metadata,
            }}
            ops.append(UpdateOne(filter_, update, upsert=True))

        if not ops:
            return 0

        result = self.collection.bulk_write(ops)
        inserted = result.upserted_count + result.modified_count
        logger.info("Upserted %d documents", inserted)
        return inserted

    def create_vector_index(self, num_dimensions: int, similarity: str = "cosine"):
        """
        Create an Atlas Vector Search index.
        Run once per collection — idempotent if index already exists.
        """
        index_spec = {
            "name": self.index_name,
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        self.embedding_field: {
                            "type": "knnVector",
                            "dimensions": num_dimensions,
                            "similarity": similarity,
                        }
                    }
                }
            }
        }
        try:
            self.collection.create_search_index(index_spec)
            logger.info("Vector index '%s' created", self.index_name)
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Index already exists")
            else:
                raise

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def vector_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        metadata_filter: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """ANN vector search via $vectorSearch aggregation stage."""
        search_stage = {
            "$vectorSearch": {
                "index": self.index_name,
                "path": self.embedding_field,
                "queryVector": query_embedding,
                "numCandidates": k * 10,
                "limit": k,
            }
        }
        if metadata_filter:
            search_stage["$vectorSearch"]["filter"] = metadata_filter

        pipeline = [
            search_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": min_score}}},
            {"$project": {self.embedding_field: 0}},   # don't return the vector itself
        ]

        results = []
        for doc in self.collection.aggregate(pipeline):
            results.append(RetrievalResult(
                document=Document(
                    content=doc.get(self.text_field, ""),
                    metadata=doc.get("metadata", {}),
                    doc_id=str(doc.get("_id", "")),
                ),
                score=doc.get("score", 0.0),
                retrieval_method="vector",
            ))
        return results

    def full_text_search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Atlas full-text search via $search aggregation stage."""
        search_stage: dict = {
            "$search": {
                "index": "default",
                "text": {
                    "query": query,
                    "path": self.text_field,
                    "fuzzy": {"maxEdits": 1},
                },
            }
        }
        if metadata_filter:
            search_stage["$search"]["filter"] = {"text": {
                "query": str(list(metadata_filter.values())[0]),
                "path": list(metadata_filter.keys())[0],
            }}

        pipeline = [
            search_stage,
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$limit": k},
            {"$project": {self.embedding_field: 0}},
        ]

        results = []
        for doc in self.collection.aggregate(pipeline):
            results.append(RetrievalResult(
                document=Document(
                    content=doc.get(self.text_field, ""),
                    metadata=doc.get("metadata", {}),
                    doc_id=str(doc.get("_id", "")),
                ),
                score=doc.get("score", 0.0),
                retrieval_method="full_text",
            ))
        return results

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) over vector + text results.
        Combines ranked lists without needing normalised score scales.
        """
        fetch_k = k * 3

        vector_results = self.vector_search(query_embedding, fetch_k, metadata_filter)
        text_results   = self.full_text_search(query, fetch_k, metadata_filter)

        # Build RRF score map  (k=60 is the standard RRF constant)
        rrf_k = 60
        scores: dict[str, float] = {}
        doc_map: dict[str, RetrievalResult] = {}

        for rank, res in enumerate(vector_results):
            did = res.document.doc_id
            scores[did] = scores.get(did, 0.0) + vector_weight / (rrf_k + rank + 1)
            doc_map[did] = res

        for rank, res in enumerate(text_results):
            did = res.document.doc_id
            scores[did] = scores.get(did, 0.0) + text_weight / (rrf_k + rank + 1)
            if did not in doc_map:
                doc_map[did] = res

        # Sort by combined RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for did, score in ranked:
            r = doc_map[did]
            results.append(RetrievalResult(
                document=r.document,
                score=score,
                retrieval_method="hybrid",
            ))
        return results


# ─── Document Processor ───────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Splits raw text into overlapping chunks for better context coverage.
    Overlap ensures sentences that span chunk boundaries are not lost.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: dict = None) -> list[Document]:
        words = text.split()
        chunks: list[Document] = []
        step = self.chunk_size - self.chunk_overlap

        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start: start + self.chunk_size]
            if not chunk_words:
                break
            chunks.append(Document(
                content=" ".join(chunk_words),
                metadata={**(metadata or {}), "chunk_index": i},
                doc_id=f"{metadata.get('source', 'doc')}_{i}" if metadata else str(i),
            ))

        logger.info("Split into %d chunks (size=%d, overlap=%d)",
                    len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    def process_documents(
        self,
        raw_docs: list[dict],     # [{"content": "...", "metadata": {...}}]
        embedding_service: EmbeddingService,
    ) -> list[Document]:
        all_chunks: list[Document] = []
        for raw in raw_docs:
            chunks = self.chunk_text(raw["content"], raw.get("metadata", {}))
            all_chunks.extend(chunks)

        # Batch embed all chunks
        texts = [c.content for c in all_chunks]
        embeddings = embedding_service.embed(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        return all_chunks


# ─── Reranker ─────────────────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker that uses Claude to score (query, passage) pairs.
    More accurate than bi-encoder retrieval for the final top-k selection.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        passages = "\n\n".join(
            f"[{i+1}] {r.document.content[:400]}"
            for i, r in enumerate(results)
        )

        prompt = f"""Score how relevant each passage is to the query.
Return ONLY a JSON list of integers in order, one score per passage (0-10).
Example: [8, 3, 9, 1]

Query: {query}

Passages:
{passages}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )

        import json, re
        text = response.content[0].text
        match = re.search(r'\[[\d,\s]+\]', text)
        if not match:
            return results[:top_k]

        scores = json.loads(match.group())
        for i, res in enumerate(results):
            if i < len(scores):
                res.score = scores[i] / 10.0

        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Steps:
      1. Embed the user query
      2. Retrieve candidate passages (vector / text / hybrid)
      3. Optionally rerank with a cross-encoder
      4. Build a grounded prompt and generate with Claude
    """

    SYSTEM_PROMPT = """You are a precise, factual assistant.
Answer questions using ONLY the provided context passages.
If the context does not contain enough information, say so clearly.
Always cite the source passage numbers like [1], [2] when making claims."""

    def __init__(
        self,
        vector_store: MongoVectorStore,
        embedding_service: EmbeddingService,
        reranker: Optional[Reranker] = None,
        generation_model: str = "claude-sonnet-4-20250514",
        retrieval_method: str = "hybrid",   # "vector" | "text" | "hybrid"
        top_k: int = 5,
        rerank_top_k: int = 3,
    ):
        self.vector_store      = vector_store
        self.embedding_service = embedding_service
        self.reranker          = reranker
        self.generation_model  = generation_model
        self.retrieval_method  = retrieval_method
        self.top_k             = top_k
        self.rerank_top_k      = rerank_top_k
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedding_service.embed_one(query)

        if self.retrieval_method == "vector":
            results = self.vector_store.vector_search(
                query_embedding, self.top_k, metadata_filter)
        elif self.retrieval_method == "text":
            results = self.vector_store.full_text_search(
                query, self.top_k, metadata_filter)
        else:   # hybrid (default)
            results = self.vector_store.hybrid_search(
                query, query_embedding, self.top_k,
                metadata_filter=metadata_filter)

        if self.reranker and results:
            results = self.reranker.rerank(query, results, self.rerank_top_k)

        return results

    # ── Generation ────────────────────────────────────────────────────────────

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, res in enumerate(results):
            meta = res.document.metadata
            source = meta.get("source", "Unknown")
            parts.append(
                f"[{i+1}] (source: {source}, score: {res.score:.3f})\n"
                f"{res.document.content}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, context: str) -> tuple[str, int]:
        prompt = f"""Context passages:

{context}

---

Question: {query}

Answer based strictly on the context above:"""

        response = self.client.messages.create(
            model=self.generation_model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text   = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        metadata_filter: Optional[dict] = None,
    ) -> RAGResponse:
        logger.info("RAG query: %s", question)

        results = self.retrieve(question, metadata_filter)
        if not results:
            return RAGResponse(
                answer="No relevant documents found.",
                sources=[],
                query=question,
                retrieval_method=self.retrieval_method,
            )

        context = self._build_context(results)
        answer, tokens = self.generate(question, context)

        return RAGResponse(
            answer=answer,
            sources=results,
            query=question,
            retrieval_method=self.retrieval_method,
            tokens_used=tokens,
        )
