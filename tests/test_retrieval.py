"""Embeddings, the FAISS index, filtering and retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from hiring_assistant.errors import (
    EmptyCorpusError,
    IndexCorruptError,
    IndexMismatchError,
    IndexNotFoundError,
)
from hiring_assistant.retrieval.embeddings import (
    HashingEmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
)
from hiring_assistant.retrieval.search import SearchFilters
from hiring_assistant.storage.vector_store import IndexMetadata, VectorStore

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


class TestEmbeddings:
    def test_vectors_are_normalised(self):
        provider = HashingEmbeddingProvider(dimension=128)
        vectors = provider.embed_documents(["alpha beta", "gamma delta"])
        assert vectors.shape == (2, 128)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embedding_is_deterministic(self):
        a = HashingEmbeddingProvider(dimension=128).embed_query("python sql analytics")
        b = HashingEmbeddingProvider(dimension=128).embed_query("python sql analytics")
        assert np.allclose(a, b)

    def test_similar_text_scores_higher_than_unrelated_text(self):
        provider = HashingEmbeddingProvider(dimension=512)
        query = provider.embed_query("python sql analytics dashboard")
        near = provider.embed_query("sql analytics dashboard reporting")
        far = provider.embed_query("welding carpentry plumbing")
        assert cosine_similarity(query, near) > cosine_similarity(query, far)

    def test_empty_input_does_not_produce_nan(self):
        vectors = HashingEmbeddingProvider(dimension=64).embed_documents(["", "real text"])
        assert not np.isnan(vectors).any()

    def test_batch_size_does_not_change_results(self):
        provider = HashingEmbeddingProvider(dimension=128)
        texts = [f"document number {i} about topic {i % 7}" for i in range(25)]
        small = provider.embed_documents(texts, batch_size=3)
        large = provider.embed_documents(texts, batch_size=100)
        assert np.allclose(small, large)

    def test_openai_model_without_key_falls_back(self, tmp_settings):
        settings = tmp_settings.with_overrides(embedding_model="openai-small")
        provider = build_embedding_provider(settings, allow_fallback=True)
        assert provider.name in {"local", "hashing"}

    def test_openai_model_without_key_raises_when_fallback_disabled(self, tmp_settings):
        from hiring_assistant.errors import MissingCredentialsError

        settings = tmp_settings.with_overrides(embedding_model="openai-small")
        with pytest.raises(MissingCredentialsError):
            build_embedding_provider(settings, allow_fallback=False)


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------


class TestVectorStore:
    def _metadata(self, dimension: int = 8, **overrides) -> IndexMetadata:
        base = {
            "embedding_model": "local-minilm",
            "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": dimension,
            "chunk_size": 900,
            "chunk_overlap": 150,
        }
        base.update(overrides)
        return IndexMetadata(**base)

    def _vectors(self, count: int, dimension: int = 8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(count, dimension)).astype("float32")

    def test_build_and_search_returns_best_match(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        vectors = self._vectors(5)
        ids = [f"c{i}" for i in range(5)]
        store.build(vectors, ids, self._metadata())

        results = store.search(vectors[2], top_k=3)
        assert results
        assert results[0][0] == "c2"
        assert results[0][1] > 0.99

    def test_save_and_load_round_trip(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        vectors = self._vectors(6)
        ids = [f"c{i}" for i in range(6)]
        store.build(vectors, ids, self._metadata())
        store.save()

        reloaded = VectorStore(tmp_path / "idx", dimension=8)
        reloaded.load()
        assert reloaded.size == 6
        assert reloaded._chunk_ids == ids

    def test_atomic_save_validates_before_swapping(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        store.build(self._vectors(4), [f"c{i}" for i in range(4)], self._metadata())
        store.save_atomic()

        assert (tmp_path / "idx" / "index_meta.json").exists()
        assert not (tmp_path / "idx.new").exists(), "staging directory must be cleaned up"

    def test_load_without_index_raises(self, tmp_path):
        with pytest.raises(IndexNotFoundError) as exc:
            VectorStore(tmp_path / "missing", dimension=8).load()
        assert exc.value.remedy

    def test_model_mismatch_is_detected(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        store.build(self._vectors(3), ["a", "b", "c"], self._metadata(embedding_model="local-minilm"))
        store.save()

        fingerprint = {
            "embedding_model": "openai-small",
            "embedding_model_id": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "chunk_size": 900,
            "chunk_overlap": 150,
        }
        with pytest.raises(IndexMismatchError) as exc:
            VectorStore(tmp_path / "idx", dimension=1536).load(fingerprint=fingerprint)
        assert "built with" in exc.value.message

    def test_chunk_size_mismatch_is_detected(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        store.build(self._vectors(3), ["a", "b", "c"], self._metadata(chunk_size=900))
        store.save()

        fingerprint = {
            "embedding_model": "local-minilm",
            "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 8,
            "chunk_size": 400,
            "chunk_overlap": 150,
        }
        with pytest.raises(IndexMismatchError):
            VectorStore(tmp_path / "idx", dimension=8).load(fingerprint=fingerprint)

    def test_corrupt_sidecar_raises(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir(parents=True)
        (index_dir / "index_meta.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(IndexCorruptError):
            VectorStore(index_dir, dimension=8).load()

    def test_sidecar_without_vectors_raises(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir(parents=True)
        (index_dir / "index_meta.json").write_text(
            '{"metadata": {"dimension": 8}, "chunk_ids": ["a"]}', encoding="utf-8"
        )
        with pytest.raises(IndexCorruptError):
            VectorStore(index_dir, dimension=8).load()

    def test_remove_chunks_drops_them_from_search(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        vectors = self._vectors(5)
        ids = [f"c{i}" for i in range(5)]
        store.build(vectors, ids, self._metadata())

        removed = store.remove_chunks(["c2"])
        assert removed == 1
        assert store.size == 4
        assert all(cid != "c2" for cid, _ in store.search(vectors[2], top_k=5))

    def test_filter_restricts_results(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        vectors = self._vectors(6)
        ids = [f"c{i}" for i in range(6)]
        store.build(vectors, ids, self._metadata())

        results = store.search(vectors[0], top_k=5, allowed_chunk_ids={"c3", "c4"})
        assert {cid for cid, _ in results} <= {"c3", "c4"}

    def test_dimension_mismatch_on_query_raises(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        store.build(self._vectors(3), ["a", "b", "c"], self._metadata())
        with pytest.raises(IndexMismatchError):
            store.search(np.zeros(99, dtype="float32"))

    def test_empty_index_initialises_and_accepts_adds(self, tmp_path):
        store = VectorStore(tmp_path / "idx", dimension=8)
        store.initialise_empty(self._metadata())
        assert store.is_loaded
        assert store.size == 0
        store.add(self._vectors(2), ["x", "y"])
        assert store.size == 2


# --------------------------------------------------------------------------
# Search service
# --------------------------------------------------------------------------


class TestSearchService:
    def test_returns_results_with_metadata(self, loaded_assistant):
        result = loaded_assistant.search("annual leave entitlement")
        assert result.results
        top = result.results[0]
        assert top.chunk.filename
        assert top.chunk.collection
        assert top.chunk.locator
        assert 0.0 <= top.score <= 1.0

    def test_scores_are_sorted_descending(self, loaded_assistant):
        result = loaded_assistant.search("data visualisation dashboards", min_relevance=0.0)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_collection_filter_restricts_scope(self, loaded_assistant):
        filters = SearchFilters(collections=["hiring_documents"])
        result = loaded_assistant.search("interview process stages", filters=filters, min_relevance=0.0)
        assert result.results
        assert all(r.chunk.collection == "hiring_documents" for r in result.results)

    def test_file_type_filter_restricts_scope(self, loaded_assistant):
        filters = SearchFilters(file_types=["md"])
        result = loaded_assistant.search("hiring", filters=filters, min_relevance=0.0)
        assert all(r.chunk.file_type == "md" for r in result.results)

    def test_document_filter_restricts_scope(self, loaded_assistant):
        candidates = loaded_assistant.candidates()
        target = candidates[0]
        filters = SearchFilters(document_ids=[target.document_id])
        result = loaded_assistant.search(
            "core skills additional skills selected projects",
            filters=filters,
            min_relevance=0.0,
        )
        assert result.results
        assert all(r.chunk.document_id == target.document_id for r in result.results)

    def test_impossible_filter_returns_nothing_not_everything(self, loaded_assistant):
        filters = SearchFilters(document_ids=["doc_does_not_exist"])
        result = loaded_assistant.search("anything", filters=filters)
        assert result.results == []
        assert "filters" in result.message.lower()

    def test_empty_query_returns_message(self, loaded_assistant):
        result = loaded_assistant.search("")
        assert result.results == []
        assert result.message

    def test_empty_corpus_raises_actionable_error(self, assistant):
        with pytest.raises((EmptyCorpusError, IndexNotFoundError)) as exc:
            assistant.search("anything at all")
        assert exc.value.remedy

    def test_low_evidence_flag_for_absent_topic(self, loaded_assistant):
        result = loaded_assistant.search("quantum computing frameworks for protein folding")
        assert result.low_evidence
        assert result.message

    def test_top_k_is_respected(self, loaded_assistant):
        result = loaded_assistant.search("hiring process", top_k=3, min_relevance=0.0)
        assert len(result.results) <= 3

    def test_dedupe_collapses_near_identical_passages(self, loaded_assistant):
        result = loaded_assistant.search("annual leave", min_relevance=0.0)
        texts = [" ".join(r.chunk.text.split()).lower() for r in result.results]
        assert len(texts) == len(set(texts))

    def test_matched_terms_are_reported(self, loaded_assistant):
        result = loaded_assistant.search("kubernetes container orchestration", min_relevance=0.0)
        assert any(r.matched_terms for r in result.results)

    def test_paraphrase_finds_differently_worded_passage(self, loaded_assistant):
        # The corpus says "Data Visualization"; the query says "visualisation".
        result = loaded_assistant.search("data visualisation dashboards")
        assert result.results
        combined = " ".join(r.chunk.text.lower() for r in result.results)
        assert "visualization" in combined or "dashboard" in combined
