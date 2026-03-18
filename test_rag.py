"""
Tests for the Advanced RAG pipeline.
Uses mocking so no live MongoDB or API keys are needed to run them.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.rag_engine import (
    Document, RetrievalResult, DocumentProcessor,
    RAGPipeline, MongoVectorStore, EmbeddingService,
)


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)

    def test_chunk_text_basic(self):
        text = " ".join([f"word{i}" for i in range(25)])
        chunks = self.processor.chunk_text(text, {"source": "test"})
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content.split()), 10)

    def test_chunk_overlap(self):
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = self.processor.chunk_text(text)
        if len(chunks) > 1:
            words0 = set(chunks[0].content.split())
            words1 = set(chunks[1].content.split())
            self.assertTrue(len(words0 & words1) > 0, "Expected overlap between chunks")

    def test_metadata_preserved(self):
        chunks = self.processor.chunk_text("hello world foo bar", {"source": "unit_test"})
        for c in chunks:
            self.assertEqual(c.metadata["source"], "unit_test")
            self.assertIn("chunk_index", c.metadata)

    def test_process_documents_embeds(self):
        fake_emb = EmbeddingService.__new__(EmbeddingService)
        fake_emb.embed = MagicMock(return_value=[[0.1, 0.2] * 512])
        raw = [{"content": "test document content for embedding", "metadata": {"source": "doc1"}}]
        result = self.processor.process_documents(raw, fake_emb)
        self.assertTrue(all(c.embedding is not None for c in result))


class TestMongoVectorStore(unittest.TestCase):
    def setUp(self):
        with patch("src.rag_engine.MongoClient") as mock_client:
            mock_client.return_value.admin.command.return_value = True
            self.store = MongoVectorStore(
                connection_string="mongodb://localhost",
                database="test_db",
                collection="test_col",
            )
            self.store.collection = MagicMock()

    def test_upsert_skips_no_embedding(self):
        docs = [Document(content="test", doc_id="d1")]   # no embedding
        count = self.store.upsert_documents(docs)
        self.assertEqual(count, 0)

    def test_vector_search_returns_results(self):
        fake_doc = {
            "_id": "abc",
            "content": "relevant passage",
            "metadata": {"source": "s1"},
            "score": 0.92,
        }
        self.store.collection.aggregate.return_value = iter([fake_doc])
        results = self.store.vector_search([0.1] * 1024, k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 0.92)
        self.assertEqual(results[0].retrieval_method, "vector")


class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        with patch("src.rag_engine.MongoClient") as mock_client:
            mock_client.return_value.admin.command.return_value = True
            self.store = MongoVectorStore(
                connection_string="mongodb://localhost",
                database="test_db",
                collection="test_col",
            )
            self.store.collection = MagicMock()

    def _make_result(self, doc_id, score, method="vector"):
        return RetrievalResult(
            document=Document(content=f"content for {doc_id}", doc_id=doc_id),
            score=score,
            retrieval_method=method,
        )

    def test_rrf_merges_results(self):
        vec_results  = [self._make_result("d1", 0.9), self._make_result("d2", 0.8)]
        text_results = [self._make_result("d2", 0.7, "text"), self._make_result("d3", 0.6, "text")]

        self.store.vector_search    = MagicMock(return_value=vec_results)
        self.store.full_text_search = MagicMock(return_value=text_results)

        results = self.store.hybrid_search("test query", [0.1] * 1024, k=3)
        doc_ids = [r.document.doc_id for r in results]

        # d2 appears in both lists — should score high
        self.assertIn("d2", doc_ids)
        self.assertLessEqual(len(results), 3)
        # Scores should be descending
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestRAGPipeline(unittest.TestCase):
    def _make_pipeline(self):
        mock_store = MagicMock(spec=MongoVectorStore)
        mock_emb   = MagicMock(spec=EmbeddingService)
        mock_emb.embed_one.return_value = [0.1] * 1024

        return RAGPipeline(
            vector_store=mock_store,
            embedding_service=mock_emb,
            retrieval_method="vector",
            top_k=3,
        ), mock_store, mock_emb

    def test_empty_retrieval_returns_no_docs_message(self):
        pipeline, mock_store, _ = self._make_pipeline()
        mock_store.vector_search.return_value = []

        response = pipeline.query("What is RAG?")
        self.assertIn("No relevant documents", response.answer)
        self.assertEqual(response.sources, [])

    def test_query_calls_generate_on_results(self):
        pipeline, mock_store, _ = self._make_pipeline()
        mock_store.vector_search.return_value = [
            RetrievalResult(
                document=Document(content="RAG stands for Retrieval-Augmented Generation.", doc_id="d1"),
                score=0.95,
                retrieval_method="vector",
            )
        ]
        pipeline.generate = MagicMock(return_value=("RAG is a retrieval technique.", 150))

        response = pipeline.query("What is RAG?")
        self.assertEqual(response.answer, "RAG is a retrieval technique.")
        self.assertEqual(response.tokens_used, 150)
        self.assertEqual(len(response.sources), 1)

    def test_metadata_filter_passed_through(self):
        pipeline, mock_store, _ = self._make_pipeline()
        mock_store.vector_search.return_value = []

        pipeline.query("question", metadata_filter={"category": "finance"})
        mock_store.vector_search.assert_called_once()
        _, kwargs = mock_store.vector_search.call_args
        self.assertEqual(kwargs.get("metadata_filter") or
                         mock_store.vector_search.call_args[0][2],
                         {"category": "finance"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
