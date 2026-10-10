"""Central configuration for the AI-Powered Hiring Assistant.

Every tunable in the RAG pipeline lives here so that ingestion, retrieval,
generation and evaluation all read the same values.  Configuration is resolved
in this order (later wins):

1. Built-in defaults (safe, zero-credential, fully local).
2. Environment variables / ``.env`` file.
3. Explicit overrides passed at call time (e.g. the Streamlit sidebar).

Secrets are *never* logged.  :meth:`Settings.public_summary` is the only
sanctioned way to render configuration in the UI and it masks credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is declared in requirements
    _DOTENV_AVAILABLE = False


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------
# Model catalogue
# --------------------------------------------------------------------------

#: Local sentence-transformers models that are known to work well for short,
#: English-language retrieval over resumes and job descriptions.
LOCAL_EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "local-minilm": {
        "provider": "local",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_tokens": 256,
        "label": "Local MiniLM-L6 (fast, 384-d)",
    },
    "local-mpnet": {
        "provider": "local",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "max_tokens": 384,
        "label": "Local MPNet-base (higher quality, 768-d)",
    },
}

#: OpenAI embedding models.  Require ``OPENAI_API_KEY``.
OPENAI_EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "openai-small": {
        "provider": "openai",
        "model_id": "text-embedding-3-small",
        "dimension": 1536,
        "max_tokens": 8191,
        "label": "OpenAI text-embedding-3-small (1536-d)",
    },
    "openai-large": {
        "provider": "openai",
        "model_id": "text-embedding-3-large",
        "dimension": 3072,
        "max_tokens": 8191,
        "label": "OpenAI text-embedding-3-large (3072-d)",
    },
}

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    **LOCAL_EMBEDDING_MODELS,
    **OPENAI_EMBEDDING_MODELS,
}

#: Generation models.
GENERATION_MODELS: dict[str, dict[str, Any]] = {
    "extractive": {
        "provider": "extractive",
        "model_id": "extractive-evidence-only",
        "label": "Extractive (no API key — quotes retrieved evidence)",
        "requires_key": False,
    },
    "openai-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "label": "OpenAI gpt-4o-mini",
        "requires_key": True,
    },
    "openai-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "label": "OpenAI gpt-4o",
        "requires_key": True,
    },
}


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------

COLLECTIONS: dict[str, str] = {
    "candidate_profiles": "Candidate profiles (synthetic)",
    "job_descriptions": "Job descriptions (synthetic)",
    "hiring_documents": "Hiring, process & company documents (synthetic)",
    "uploads": "User-uploaded documents",
    "general": "Unclassified",
}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved application configuration."""

    # -- provider / credentials -------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_model: str = "local-minilm"
    generation_model: str = "extractive"
    allow_openai: bool = False

    # -- storage ----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "index")
    synthetic_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "synthetic")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "hiring_assistant.sqlite3")

    # -- chunking ---------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 40

    # -- retrieval --------------------------------------------------------
    top_k: int = 6
    candidate_pool: int = 40
    min_relevance: float = 0.18
    max_context_chunks: int = 10
    dedupe_similarity: float = 0.97

    # -- ingestion --------------------------------------------------------
    max_upload_mb: int = 25
    allowed_extensions: tuple[str, ...] = (
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
    )
    embedding_batch_size: int = 32

    # -- generation -------------------------------------------------------
    max_answer_tokens: int = 700
    request_timeout_seconds: int = 45

    # -- safety -----------------------------------------------------------
    max_query_chars: int = 2000
    evidence_status_supported: float = 0.45
    evidence_status_partial: float = 0.22

    # -- app --------------------------------------------------------------
    app_title: str = "AI-Powered Hiring Assistant"
    demo_mode: bool = True

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def embedding_spec(self) -> dict[str, Any]:
        return EMBEDDING_MODELS.get(self.embedding_model, EMBEDDING_MODELS["local-minilm"])

    @property
    def embedding_model_id(self) -> str:
        return str(self.embedding_spec["model_id"])

    @property
    def embedding_dimension(self) -> int:
        return int(self.embedding_spec["dimension"])

    @property
    def embedding_provider(self) -> str:
        return str(self.embedding_spec["provider"])

    @property
    def generation_spec(self) -> dict[str, Any]:
        return GENERATION_MODELS.get(self.generation_model, GENERATION_MODELS["extractive"])

    @property
    def generation_model_id(self) -> str:
        return str(self.generation_spec["model_id"])

    @property
    def generation_provider(self) -> str:
        return str(self.generation_spec["provider"])

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def generation_is_llm(self) -> bool:
        return self.generation_provider != "extractive"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def index_fingerprint(self) -> dict[str, Any]:
        """Configuration that an index must match to be considered valid."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimension": self.embedding_dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def public_summary(self) -> dict[str, Any]:
        """A UI-safe summary.  Never contains the API key itself."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_model_id": self.embedding_model_id,
            "embedding_provider": self.embedding_provider,
            "embedding_dimension": self.embedding_dimension,
            "generation_model": self.generation_model,
            "generation_model_id": self.generation_model_id,
            "generation_provider": self.generation_provider,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "min_relevance": self.min_relevance,
            "max_upload_mb": self.max_upload_mb,
            "openai_key_present": self.has_openai_key,
            "external_calls_enabled": self.external_calls_enabled,
        }

    @property
    def external_calls_enabled(self) -> bool:
        """True when document or query text may leave this machine."""
        if not self.has_openai_key:
            return False
        return self.embedding_provider == "openai" or self.generation_provider == "openai"

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def with_overrides(self, **kwargs: Any) -> Settings:
        """Return a copy with the given fields replaced (validated)."""
        clean = {k: v for k, v in kwargs.items() if v is not None and hasattr(self, k)}
        if not clean:
            return self
        candidate = replace(self, **clean)
        candidate.validate()
        return candidate

    def validate(self) -> None:
        """Raise :class:`ValueError` for unusable combinations."""
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0.0 and 1.0")
        if self.embedding_model not in EMBEDDING_MODELS:
            raise ValueError(f"unknown embedding model: {self.embedding_model}")
        if self.generation_model not in GENERATION_MODELS:
            raise ValueError(f"unknown generation model: {self.generation_model}")
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive")

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def resolved(self) -> Settings:
        """Downgrade to local-only operation when credentials are missing.

        This is what keeps the zero-credential promise: an OpenAI model can be
        *requested* in config, but if no key is present we transparently fall
        back to the local model and record the fact in ``fallback_notes``.
        """
        notes: list[str] = []
        updates: dict[str, Any] = {}

        if self.embedding_provider == "openai" and not self.has_openai_key:
            updates["embedding_model"] = "local-minilm"
            notes.append(
                "Embedding model requested an OpenAI model but OPENAI_API_KEY is not set; "
                "fell back to local sentence-transformers/all-MiniLM-L6-v2."
            )

        if self.generation_provider == "openai" and not self.has_openai_key:
            updates["generation_model"] = "extractive"
            notes.append(
                "Generation model requested OpenAI but OPENAI_API_KEY is not set; "
                "fell back to clearly-labelled extractive evidence mode."
            )

        if self.allow_openai and not self.has_openai_key:
            notes.append(
                "ALLOW_OPENAI is true but no key was found; the app runs fully locally."
            )

        if not updates:
            resolved = self
        else:
            resolved = replace(self, **updates)
        object.__setattr__(resolved, "_fallback_notes", tuple(notes))
        return resolved

    @property
    def fallback_notes(self) -> tuple[str, ...]:
        return getattr(self, "_fallback_notes", ())


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------

_SETTINGS_CACHE: Settings | None = None


def load_settings(force_reload: bool = False, **overrides: Any) -> Settings:
    """Build :class:`Settings` from the environment, optionally overriding."""
    global _SETTINGS_CACHE

    if _SETTINGS_CACHE is None or force_reload or overrides:
        if _DOTENV_AVAILABLE:
            env_file = PROJECT_ROOT / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=False)

        base = Settings(
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
            embedding_model=_env_str("EMBEDDING_MODEL", "local-minilm"),
            generation_model=_env_str("GENERATION_MODEL", "extractive"),
            allow_openai=_env_bool("ALLOW_OPENAI", False),
            data_dir=Path(_env_str("DATA_DIR", str(PROJECT_ROOT / "data"))),
            index_dir=Path(_env_str("INDEX_DIR", str(PROJECT_ROOT / "data" / "index"))),
            synthetic_dir=Path(_env_str("SYNTHETIC_DIR", str(PROJECT_ROOT / "data" / "synthetic"))),
            db_path=Path(_env_str("DB_PATH", str(PROJECT_ROOT / "data" / "hiring_assistant.sqlite3"))),
            chunk_size=_env_int("CHUNK_SIZE", 900),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
            top_k=_env_int("TOP_K", 6),
            min_relevance=_env_float("MIN_RELEVANCE", 0.18),
            max_context_chunks=_env_int("MAX_CONTEXT_CHUNKS", 10),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 25),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
            max_answer_tokens=_env_int("MAX_ANSWER_TOKENS", 700),
            request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 45),
            demo_mode=_env_bool("DEMO_MODE", True),
        )

        if overrides:
            base = base.with_overrides(**overrides)

        base.validate()
        resolved = base.resolved()
        if overrides or force_reload:
            return resolved
        _SETTINGS_CACHE = resolved

    return _SETTINGS_CACHE


def reset_settings_cache() -> None:
    """Test helper: forget the memoised settings."""
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None


__all__ = [
    "COLLECTIONS",
    "EMBEDDING_MODELS",
    "GENERATION_MODELS",
    "LOCAL_EMBEDDING_MODELS",
    "OPENAI_EMBEDDING_MODELS",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "Settings",
    "load_settings",
    "reset_settings_cache",
]
