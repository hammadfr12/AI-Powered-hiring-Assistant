"""Embedding providers behind one narrow interface.

Three implementations, chosen automatically:

``LocalEmbeddingProvider``
    ``sentence-transformers``.  Runs offline, no credentials, no cost.  This is
    the default and what makes the zero-credential promise real.

``OpenAIEmbeddingProvider``
    ``text-embedding-3-*`` via the REST API.  Sends document and query text to
    OpenAI — disclosed in the UI and in docs/PRIVACY_AND_RESPONSIBLE_USE.md.

``HashingEmbeddingProvider``
    A deterministic hashed bag-of-words projection.  It is *not* a semantic
    model, but it is stable, instant, dependency-free and good enough for the
    test-suite to exercise the whole pipeline without downloading weights.
    It is only ever selected explicitly or as a last-resort fallback, and it
    always announces itself via :attr:`EmbeddingProvider.label`.

All providers expose the same contract: :meth:`embed_documents` (batched),
:meth:`embed_query` (single) and :attr:`dimension`.  Vectors are returned
L2-normalised so downstream cosine similarity is a plain inner product.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import Settings
from ..errors import (
    EmbeddingError,
    MissingCredentialsError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from ..utils import tokenize

# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Common interface for every embedding backend."""

    #: Stable identifier stored on every chunk so an index can be validated.
    name: str = "base"
    #: Human-readable label for the Settings view.
    label: str = "Embedding provider"
    #: True when document text leaves this machine.
    external: bool = False

    def __init__(self, dimension: int, model_id: str) -> None:
        self._dimension = dimension
        self.model_id = model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @abstractmethod
    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Return a ``(len(texts), dimension)`` float32 matrix."""

    # -- public API ----------------------------------------------------

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        progress: Any = None,
    ) -> np.ndarray:
        """Embed many texts, batched, L2-normalised."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        chunks: list[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, max(1, batch_size)):
            window = list(texts[start : start + batch_size])
            vectors = self._embed_batch(window)
            if vectors.shape[0] != len(window):
                raise EmbeddingError(
                    f"Provider returned {vectors.shape[0]} vectors for {len(window)} inputs."
                )
            chunks.append(vectors)
            if progress is not None:
                progress(min(start + len(window), total), total)

        matrix = np.vstack(chunks).astype("float32")
        if matrix.shape[1] != self.dimension:
            raise EmbeddingError(
                f"Provider produced {matrix.shape[1]}-dimensional vectors but declared "
                f"{self.dimension}. The model may have changed upstream."
            )
        return _l2_normalise(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string, L2-normalised."""
        return self.embed_documents([text])[0]

    def warmup(self) -> None:
        """Load weights eagerly.  Optional; failures are non-fatal."""
        try:
            self._embed_batch(["warmup"])
        except Exception:  # pragma: no cover - warmup is best-effort
            pass

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "external": self.external,
            "label": self.label,
        }


# --------------------------------------------------------------------------
# Local sentence-transformers
# --------------------------------------------------------------------------


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers backend.  Offline, no credentials."""

    name = "local"
    external = False

    def __init__(self, model_id: str, dimension: int, *, label: str = "") -> None:
        super().__init__(dimension, model_id)
        self.label = label or f"Local · {model_id.split('/')[-1]}"
        self._model: Any = None
        self._load_error: str | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error:
            raise EmbeddingError(self._load_error)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            self._load_error = (
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers`, or set EMBEDDING_MODEL=local-minilm "
                "after installing, or use the hashing fallback for offline testing."
            )
            raise EmbeddingError(self._load_error) from exc

        try:
            self._model = SentenceTransformer(self.model_id, device="cpu")
        except Exception as exc:  # model download failure, corrupt cache, etc.
            self._load_error = (
                f"Could not load '{self.model_id}': {exc}. The first run downloads the model "
                "from Hugging Face and needs network access; subsequent runs work offline."
            )
            raise EmbeddingError(self._load_error) from exc
        return self._model

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=len(texts) or 1,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def is_available(self) -> bool:
        try:
            self._ensure_model()
            return True
        except EmbeddingError:
            return False


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI ``text-embedding-3-*`` via REST (no SDK dependency)."""

    name = "openai"
    external = True

    def __init__(
        self,
        model_id: str,
        dimension: int,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: int = 45,
    ) -> None:
        super().__init__(dimension, model_id)
        self.label = f"OpenAI · {model_id}"
        self._api_key = api_key
        self._base_url = (base_url or _OPENAI_EMBEDDINGS_URL).rstrip("/")
        self._timeout = timeout

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not self._api_key:
            raise MissingCredentialsError(
                "OPENAI_API_KEY is not set, so OpenAI embeddings cannot be used."
            )

        payload = json.dumps(
            {"model": self.model_id, "input": list(texts), "encoding_format": "float"}
        ).encode("utf-8")
        request = urllib.request.Request(
            self._base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover
                pass
            if exc.code == 429:
                raise ProviderRateLimitError(
                    f"OpenAI rate limit reached (HTTP 429). {detail}"
                ) from exc
            if exc.code in (401, 403):
                raise MissingCredentialsError(
                    "OpenAI rejected the API key (HTTP "
                    f"{exc.code}). Check OPENAI_API_KEY in your .env file."
                ) from exc
            raise ProviderResponseError(f"OpenAI returned HTTP {exc.code}. {detail}") from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower():
                raise ProviderTimeoutError(
                    f"OpenAI embeddings timed out after {self._timeout}s."
                ) from exc
            raise ProviderResponseError(f"Could not reach OpenAI: {reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise ProviderTimeoutError(f"OpenAI embeddings timed out: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("OpenAI returned a non-JSON response.") from exc

        rows = body.get("data")
        if not isinstance(rows, list) or not rows:
            raise ProviderResponseError(
                "OpenAI response contained no embedding data.",
                context={"keys": sorted(body.keys())},
            )

        rows = sorted(rows, key=lambda r: r.get("index", 0))
        vectors = [row.get("embedding") for row in rows]
        if any(v is None for v in vectors):
            raise ProviderResponseError("OpenAI response contained a row without an embedding.")
        return np.asarray(vectors, dtype="float32")


# --------------------------------------------------------------------------
# Deterministic hashing fallback
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#._/-]*")


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-words projection.

    Not a semantic model.  It maps lexical tokens into a fixed-dimensional
    space with sublinear term weighting, which makes it a reasonable
    *lexical* retriever and a perfectly reproducible one.  Used for offline
    testing and as a last-resort fallback when neither sentence-transformers
    nor an API key is available.
    """

    name = "hashing"
    label = "Hashing fallback (lexical, deterministic — not a semantic model)"
    external = False

    def __init__(self, dimension: int = 512, model_id: str = "hashing-bow-v1") -> None:
        super().__init__(dimension, model_id)

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            tokens = tokenize(text or "")
            if not tokens:
                # Deterministic non-zero vector for empty input so that
                # normalisation never produces NaN.
                matrix[row, 0] = 1.0
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                weight = 1.0 + math.log(count)
                for bucket, sign in self._buckets(token):
                    matrix[row, bucket] += sign * weight
        return matrix

    def _buckets(self, token: str) -> list[tuple[int, float]]:
        """Two independent hashes per token to reduce collision bias."""
        out: list[tuple[int, float]] = []
        for salt in (b"a", b"b"):
            digest = hashlib.blake2b(salt + token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            out.append((value % self.dimension, 1.0 if (value >> 63) & 1 else -1.0))
        return out


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_embedding_provider(
    settings: Settings,
    *,
    allow_fallback: bool = True,
) -> EmbeddingProvider:
    """Instantiate the configured provider, falling back gracefully.

    Order of preference when the requested provider cannot be built:

    1. The requested provider.
    2. ``sentence-transformers`` local model (if installed).
    3. :class:`HashingEmbeddingProvider`.

    The fallback is recorded on the provider's ``label`` so the UI can state
    plainly which model actually produced the vectors.
    """
    spec = settings.embedding_spec
    provider = spec["provider"]

    if provider == "openai":
        if settings.has_openai_key:
            return OpenAIEmbeddingProvider(
                str(spec["model_id"]),
                int(spec["dimension"]),
                api_key=settings.openai_api_key or "",
                base_url=settings.openai_base_url,
                timeout=settings.request_timeout_seconds,
            )
        if not allow_fallback:
            raise MissingCredentialsError(
                "Embedding model is set to an OpenAI model but OPENAI_API_KEY is missing."
            )
        provider = "local"

    if provider == "local":
        candidate = LocalEmbeddingProvider(
            str(spec["model_id"]),
            int(spec["dimension"]),
            label=str(spec.get("label", "")),
        )
        if candidate.is_available():
            return candidate
        if not allow_fallback:
            raise EmbeddingError(
                f"Local embedding model '{spec['model_id']}' could not be loaded."
            )
        fallback = HashingEmbeddingProvider(dimension=512)
        fallback.label = (
            "Hashing fallback (lexical, deterministic) — "
            "sentence-transformers unavailable, so semantic quality is reduced"
        )
        return fallback

    return HashingEmbeddingProvider(dimension=512)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_pairwise_similarity(vectors: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity; used by the evaluation report."""
    matrix = _l2_normalise(vectors)
    if matrix.shape[0] < 2:
        return 0.0
    sims = matrix @ matrix.T
    n = sims.shape[0]
    total = float(sims.sum() - np.trace(sims))
    return total / (n * (n - 1))


__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "build_embedding_provider",
    "cosine_similarity",
    "mean_pairwise_similarity",
]
