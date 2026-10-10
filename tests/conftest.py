"""Shared test fixtures.

Every fixture is scoped to a temporary directory and uses the deterministic
hashing embedder by default, so the full suite runs offline, without downloading
model weights and without an API key.  Tests that genuinely need the real
sentence-transformers model are marked ``slow``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hiring_assistant.config import Settings, reset_settings_cache
from hiring_assistant.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
)
from hiring_assistant.services import HiringAssistant


@pytest.fixture(autouse=True)
def _clean_settings_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointed entirely at a temporary directory, using the local
    hashing embedder so tests never download model weights."""
    settings = Settings(
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "data" / "index",
        synthetic_dir=tmp_path / "data" / "synthetic",
        db_path=tmp_path / "data" / "test.sqlite3",
        embedding_model="local-minilm",
        generation_model="extractive",
        chunk_size=600,
        chunk_overlap=100,
        top_k=5,
        min_relevance=0.05,
        embedding_batch_size=16,
    )
    settings.validate()
    return settings


@pytest.fixture
def hashing_embedder() -> EmbeddingProvider:
    return HashingEmbeddingProvider(dimension=256)


@pytest.fixture
def store(tmp_settings: Settings):
    from hiring_assistant.storage.database import MetadataStore

    return MetadataStore(tmp_settings.db_path)


@pytest.fixture
def vector_store(tmp_settings: Settings):
    from hiring_assistant.storage.vector_store import VectorStore

    return VectorStore(tmp_settings.index_dir, dimension=256)


@pytest.fixture(scope="session")
def corpus():
    """The in-memory synthetic corpus (session-scoped: generation is pure)."""
    from hiring_assistant.synthetic import generate_corpus

    return generate_corpus(candidate_count=24)


@pytest.fixture
def corpus_dir(tmp_path: Path):
    """The synthetic corpus written to a temporary directory."""
    from hiring_assistant.synthetic import write_corpus

    target = tmp_path / "corpus"
    write_corpus(target, candidate_count=24)
    return target


@pytest.fixture
def assistant(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> HiringAssistant:
    """A fully wired assistant using the deterministic hashing embedder.

    Patching :func:`build_embedding_provider` keeps the tests offline and fast
    while still exercising the real ingestion, indexing, retrieval and
    generation code paths.
    """
    from hiring_assistant import retrieval, services

    def _hashing_provider(settings, *, allow_fallback=True):
        return HashingEmbeddingProvider(dimension=256)

    monkeypatch.setattr(services, "build_embedding_provider", _hashing_provider)
    monkeypatch.setattr(retrieval, "build_embedding_provider", _hashing_provider)

    return HiringAssistant(tmp_settings, load_index=False)


@pytest.fixture
def loaded_assistant(assistant: HiringAssistant) -> HiringAssistant:
    """An assistant with the synthetic corpus ingested and indexed."""
    assistant.load_synthetic_corpus()
    assistant.load_index()
    return assistant


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
