"""
End-to-end demo: ingest sample documents, then query the RAG pipeline.
Run: python demo.py
"""

import os
import json
import requests

BASE_URL = os.getenv("RAG_API_URL", "http://localhost:5000")

SAMPLE_DOCS = [
    {
        "content": (
            "MongoDB Atlas Vector Search enables developers to build intelligent applications "
            "by performing semantic similarity searches directly on data stored in MongoDB. "
            "It supports k-Nearest Neighbor (kNN) search using the HNSW algorithm, offering "
            "extremely fast approximate nearest neighbor lookups over millions of vectors with "
            "configurable precision vs. performance trade-offs."
        ),
        "metadata": {"source": "mongodb_docs", "category": "databases", "version": "7.0"}
    },
    {
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances large language "
            "model outputs by grounding responses in retrieved factual documents. Instead of "
            "relying solely on parametric knowledge baked into model weights, RAG fetches "
            "relevant passages at inference time and includes them in the context window. "
            "This dramatically reduces hallucination rates and enables up-to-date answers "
            "without expensive model retraining."
        ),
        "metadata": {"source": "ai_research", "category": "machine_learning"}
    },
    {
        "content": (
            "Hybrid search combines dense vector retrieval with sparse keyword search to "
            "achieve better recall than either method alone. Reciprocal Rank Fusion (RRF) "
            "merges the ranked result lists by assigning each document a score of "
            "1/(k + rank) where k is a smoothing constant (typically 60). Documents "
            "appearing in both ranked lists receive additive RRF scores, naturally "
            "surfacing documents that are both semantically similar and keyword-relevant."
        ),
        "metadata": {"source": "search_engineering", "category": "information_retrieval"}
    },
    {
        "content": (
            "Cross-encoder reranking is a second-stage retrieval step that scores "
            "(query, passage) pairs jointly instead of independently. Unlike bi-encoders "
            "that embed query and document separately, cross-encoders see both inputs "
            "simultaneously, enabling rich attention interactions that produce more accurate "
            "relevance scores. The trade-off is speed: cross-encoders are 100x slower, "
            "so they are typically applied to the top-20 candidates from the first stage."
        ),
        "metadata": {"source": "ir_textbook", "category": "information_retrieval"}
    },
    {
        "content": (
            "The Anthropic Claude API provides access to Claude's language models via REST. "
            "Claude Sonnet 4 offers the best balance of speed, cost, and capability for "
            "production RAG workloads. The Messages API supports system prompts, multi-turn "
            "conversations, tool use, and streaming responses. Token usage is billed separately "
            "for input and output tokens."
        ),
        "metadata": {"source": "anthropic_docs", "category": "llm_apis"}
    },
]

SAMPLE_QUERIES = [
    "How does MongoDB Atlas Vector Search work?",
    "What is the difference between RAG and fine-tuning?",
    "Explain hybrid search and RRF fusion.",
    "When should I use a cross-encoder reranker?",
]


def ingest():
    print("\n=== Ingesting sample documents ===")
    resp = requests.post(f"{BASE_URL}/ingest", json={"documents": SAMPLE_DOCS})
    resp.raise_for_status()
    result = resp.json()
    print(f"  Source docs:    {result['source_docs']}")
    print(f"  Ingested chunks: {result['ingested_chunks']}")


def run_queries():
    for question in SAMPLE_QUERIES:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print("─" * 60)

        resp = requests.post(f"{BASE_URL}/query", json={
            "question": question,
            "retrieval_method": "hybrid",
        })
        resp.raise_for_status()
        data = resp.json()

        print(f"A: {data['answer']}\n")
        print(f"   Method : {data['method']}  |  Tokens: {data['tokens']}")
        print(f"   Sources ({len(data['sources'])}):")
        for i, src in enumerate(data["sources"]):
            print(f"     [{i+1}] score={src['score']:.3f} | {src['metadata'].get('source','?')} "
                  f"| {src['content'][:80]}...")


def main():
    print("RAG Pipeline Demo")
    print("Make sure `python app.py` is running in another terminal.\n")

    try:
        requests.get(f"{BASE_URL}/health").raise_for_status()
    except Exception:
        print(f"ERROR: Could not reach API at {BASE_URL}")
        print("Start the server first:  python app.py")
        return

    ingest()
    run_queries()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
