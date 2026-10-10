"""Synthetic corpus determinism, configuration handling and storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hiring_assistant.config import (
    EMBEDDING_MODELS,
    GENERATION_MODELS,
    Settings,
    load_settings,
)
from hiring_assistant.synthetic import (
    CORPUS_VERSION,
    DEFAULT_SEED,
    generate_candidates,
    generate_corpus,
    load_corpus_files,
)

# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


class TestSyntheticCorpus:
    def test_meets_minimum_content_requirements(self, corpus):
        assert corpus.candidate_count >= 20, "at least 20 candidate profiles required"
        assert len(corpus.jobs) >= 5, "at least 5 job descriptions required"
        assert len(corpus.documents) >= 3, "at least 3 hiring documents required"

    def test_generation_is_deterministic(self):
        first = generate_corpus(candidate_count=20)
        second = generate_corpus(candidate_count=20)
        assert first.candidates == second.candidates
        assert first.files == second.files

    def test_candidate_ids_are_stable_across_runs(self):
        a = [c["profile_id"] for c in generate_candidates(20)]
        b = [c["profile_id"] for c in generate_candidates(20)]
        assert a == b
        assert a[0] == "cand_001"

    def test_no_protected_attributes_in_corpus(self, corpus):
        """No candidate record may carry a sensitive characteristic."""
        forbidden = {
            "age", "gender", "sex", "marital_status", "married", "religion", "caste",
            "nationality", "race", "ethnicity", "disability", "health", "pregnan",
            "children", "family_status", "date_of_birth", "dob", "photo",
        }
        for candidate in corpus.candidates:
            keys = {k.lower() for k in candidate}
            overlap = keys & forbidden
            assert not overlap, f"candidate carries protected field(s): {overlap}"

    def test_corpus_is_labelled_synthetic(self, corpus):
        manifest = corpus.manifest()
        assert manifest["synthetic"] is True
        for candidate in corpus.candidates:
            assert candidate["synthetic"] is True

    def test_writes_multiple_file_formats(self, corpus_dir: Path):
        extensions = {path.suffix.lower() for path in corpus_dir.rglob("*") if path.is_file()}
        for required in {".csv", ".json", ".md", ".txt", ".pdf", ".docx"}:
            assert required in extensions, f"corpus is missing a {required} example"

    def test_malformed_fixtures_exist(self, corpus_dir: Path):
        malformed = corpus_dir / "_malformed"
        assert malformed.exists()
        names = {p.name for p in malformed.iterdir()}
        for expected in ("empty.txt", "broken.json", "truncated.pdf", "not_a_real.docx"):
            assert expected in names

    def test_malformed_directory_is_excluded_from_loading(self, corpus_dir: Path):
        entries = load_corpus_files(corpus_dir)
        assert entries
        assert not any("_malformed" in name for name, _, _ in entries)

    def test_collections_are_assigned_correctly(self, corpus_dir: Path):
        entries = load_corpus_files(corpus_dir)
        collections = {collection for _, _, collection in entries}
        assert "candidate_profiles" in collections
        assert "job_descriptions" in collections
        assert "hiring_documents" in collections

    def test_manifest_records_version_and_seed(self, corpus_dir: Path):
        manifest = json.loads((corpus_dir / "corpus_manifest.json").read_text())
        assert manifest["corpus_version"] == CORPUS_VERSION
        assert manifest["seed"] == DEFAULT_SEED

    def test_conflicting_sources_exist(self, corpus):
        """The corpus must contain two documents that disagree, for the
        conflicting-sources evaluation category."""
        guide = next(d for d in corpus.documents if d["doc_id"] == "doc_hiring_guide")
        memo = corpus.files["documents/recruiting_operations_memo.txt"].decode()
        assert "12 working days" in guide["body"]
        assert "14 working days" in memo

    def test_pdf_output_is_parseable(self, corpus_dir: Path):
        from hiring_assistant.ingestion.parsers import parse_document

        pdfs = list((corpus_dir / "documents").glob("*.pdf"))
        assert pdfs
        result = parse_document(pdfs[0].read_bytes(), pdfs[0].name)
        assert result.char_count > 200
        assert result.page_count and result.page_count >= 1

    def test_docx_output_is_parseable(self, corpus_dir: Path):
        from hiring_assistant.ingestion.parsers import parse_document

        docx_files = list((corpus_dir / "documents").glob("*.docx"))
        assert docx_files
        result = parse_document(docx_files[0].read_bytes(), docx_files[0].name)
        assert result.char_count > 200

    def test_profiles_are_readable_and_citable(self, corpus):
        profile = corpus.files["profiles/cand_001_neel_rao.txt"].decode()
        assert "CANDIDATE PROFILE" in profile
        assert "CORE SKILLS" in profile
        assert "SYNTHETIC DATA NOTICE" in profile


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class TestSettings:
    def test_defaults_need_no_credentials(self):
        settings = Settings()
        settings.validate()
        resolved = settings.resolved()
        assert resolved.embedding_provider == "local"
        assert resolved.generation_provider == "extractive"
        assert not resolved.external_calls_enabled

    def test_openai_without_key_falls_back_with_a_note(self):
        settings = Settings(embedding_model="openai-small", generation_model="openai-mini")
        resolved = settings.resolved()
        assert resolved.embedding_model == "local-minilm"
        assert resolved.generation_model == "extractive"
        assert resolved.fallback_notes
        assert "OPENAI_API_KEY" in resolved.fallback_notes[0]

    def test_openai_with_key_stays_selected(self):
        settings = Settings(
            openai_api_key="sk-test-not-a-real-key",
            embedding_model="openai-small",
            generation_model="openai-mini",
        )
        resolved = settings.resolved()
        assert resolved.embedding_model == "openai-small"
        assert resolved.generation_model == "openai-mini"
        assert resolved.external_calls_enabled

    def test_public_summary_never_contains_the_key(self):
        secret = "sk-super-secret-value-do-not-log"
        settings = Settings(openai_api_key=secret, embedding_model="openai-small")
        summary = settings.public_summary()
        serialised = json.dumps(summary)
        assert secret not in serialised
        assert summary["openai_key_present"] is True

    def test_index_fingerprint_captures_embedding_and_chunking(self):
        settings = Settings(chunk_size=500, chunk_overlap=80)
        fingerprint = settings.index_fingerprint()
        assert fingerprint["chunk_size"] == 500
        assert fingerprint["chunk_overlap"] == 80
        assert "embedding_model_id" in fingerprint

    @pytest.mark.parametrize(
        "overrides",
        [
            {"chunk_size": 50},
            {"chunk_size": 300, "chunk_overlap": 300},
            {"top_k": 0},
            {"top_k": 100},
            {"min_relevance": 1.5},
            {"embedding_model": "does-not-exist"},
            {"generation_model": "does-not-exist"},
            {"max_upload_mb": 0},
        ],
    )
    def test_invalid_configuration_is_rejected(self, overrides):
        settings = Settings(**overrides)
        with pytest.raises(ValueError):
            settings.validate()

    def test_with_overrides_returns_a_validated_copy(self):
        settings = Settings()
        updated = settings.with_overrides(top_k=12)
        assert updated.top_k == 12
        assert settings.top_k != 12  # original untouched
        with pytest.raises(ValueError):
            settings.with_overrides(top_k=999)

    def test_model_catalogue_is_well_formed(self):
        for key, spec in EMBEDDING_MODELS.items():
            assert spec["provider"] in {"local", "openai"}, key
            assert spec["dimension"] > 0, key
            assert spec["model_id"], key
        for key, spec in GENERATION_MODELS.items():
            assert "requires_key" in spec, key
            assert spec["provider"] in {"extractive", "openai"}, key

    def test_load_settings_from_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOP_K", "9")
        monkeypatch.setenv("CHUNK_SIZE", "640")
        monkeypatch.setenv("DB_PATH", str(tmp_path / "custom.sqlite3"))
        settings = load_settings(force_reload=True)
        assert settings.top_k == 9
        assert settings.chunk_size == 640
        assert str(settings.db_path).endswith("custom.sqlite3")


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


class TestMetadataStore:
    def test_document_round_trip(self, store):
        from hiring_assistant.models import Document

        document = Document(
            document_id="doc_1",
            filename="test.txt",
            file_type="txt",
            collection="general",
            source_hash="abc123",
            char_count=500,
            chunk_count=3,
            status="indexed",
            warnings=["a warning"],
        )
        store.upsert_document(document)
        loaded = store.get_document("doc_1")
        assert loaded is not None
        assert loaded.filename == "test.txt"
        assert loaded.warnings == ["a warning"]
        assert loaded.chunk_count == 3

    def test_find_by_hash(self, store):
        from hiring_assistant.models import Document

        store.upsert_document(
            Document(
                document_id="d1", filename="a.txt", file_type="txt",
                collection="general", source_hash="hash-a",
            )
        )
        assert store.find_by_hash("hash-a") is not None
        assert store.find_by_hash("hash-missing") is None

    def test_chunks_round_trip_with_index_ids(self, store):
        from hiring_assistant.models import Chunk, Document

        store.upsert_document(
            Document(
                document_id="d1", filename="a.txt", file_type="txt",
                collection="general", source_hash="h1",
            )
        )
        chunks = [
            Chunk(
                chunk_id=f"d1::c{i}", document_id="d1", chunk_index=i,
                text=f"passage {i}", filename="a.txt", collection="general",
            )
            for i in range(3)
        ]
        store.insert_chunks(chunks)
        store.set_index_ids({c.chunk_id: i for i, c in enumerate(chunks)})

        loaded = store.get_chunk("d1::c1")
        assert loaded is not None
        assert loaded.index_id == 1
        assert store.count_chunks() == 3

    def test_delete_document_cascades_to_chunks(self, store):
        from hiring_assistant.models import Chunk, Document

        store.upsert_document(
            Document(
                document_id="d1", filename="a.txt", file_type="txt",
                collection="general", source_hash="h1",
            )
        )
        store.insert_chunks(
            [Chunk(chunk_id="c1", document_id="d1", chunk_index=0, text="x", filename="a.txt")]
        )
        assert store.count_chunks() == 1
        store.delete_document("d1")
        assert store.count_chunks() == 0
        assert store.get_document("d1") is None

    def test_messages_are_ordered_chronologically(self, store):
        from hiring_assistant.models import Message

        store.ensure_conversation("s1")
        for i in range(5):
            store.add_message(
                Message(
                    message_id=f"m{i}", session_id="s1",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"message {i}",
                )
            )
        messages = store.get_messages("s1")
        assert [m.content for m in messages] == [f"message {i}" for i in range(5)]

    def test_limited_messages_return_the_most_recent_in_order(self, store):
        from hiring_assistant.models import Message

        store.ensure_conversation("s1")
        for i in range(10):
            store.add_message(
                Message(message_id=f"m{i}", session_id="s1", role="user", content=f"m{i}")
            )
        recent = store.get_messages("s1", limit=3)
        assert [m.content for m in recent] == ["m7", "m8", "m9"]

    def test_state_key_value_round_trip(self, store):
        store.set_state("index_built_at", {"at": "2026-01-01"})
        assert store.get_state("index_built_at")["at"] == "2026-01-01"
        assert store.get_state("missing", "default") == "default"

    def test_health_reports_counts(self, store):
        health = store.health()
        assert health["ok"] is True
        assert health["documents"] == 0
        assert health["chunks"] == 0
