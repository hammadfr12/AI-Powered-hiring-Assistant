"""FAISS-backed vector index with an atomic, mismatch-aware rebuild.

Design notes
------------
* The index lives on disk next to a JSON sidecar describing exactly how it was
  built (embedding model id, dimension, chunking config, chunk-id order).
  Loading an index whose sidecar does not match the *current* configuration
  raises :class:`IndexMismatchError` rather than silently returning garbage —
  vectors from two different models are not comparable.
* Rebuilds are **atomic**: we build into ``index.new/`` and only swap once the
  new index is written, validated and its sidecar parsed back.  A crash midway
  leaves the previous working index untouched.
* ``faiss`` is an optional import.  When it is unavailable we fall back to a
  NumPy inner-product search with identical semantics, so the app and the full
  test-suite still run.  The active backend is reported in the UI.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Must run before `import faiss`: on macOS arm64, importing FAISS before PyTorch
# aborts the process with a bus error once the encoder runs.  See _compat.py.
from .. import _compat  # noqa: F401  (import side effect is the point)
from ..errors import IndexCorruptError, IndexMismatchError, IndexNotFoundError
from ..models import utcnow
from ..utils import ensure_dir

try:  # pragma: no cover - exercised by the fallback test
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


INDEX_FILENAME = "faiss.index"
SIDECAR_FILENAME = "index_meta.json"
BACKEND_FILENAME = "backend.txt"


@dataclass
class IndexMetadata:
    """Everything needed to decide whether a saved index is still usable."""

    embedding_model: str = ""
    embedding_model_id: str = ""
    dimension: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    backend: str = "faiss" if FAISS_AVAILABLE else "numpy"
    vector_count: int = 0
    built_at: str = field(default_factory=utcnow)
    corpus_fingerprint: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexMetadata:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def matches(self, fingerprint: dict[str, Any]) -> tuple[bool, str]:
        """Compare against :meth:`Settings.index_fingerprint`."""
        for key in ("embedding_model", "embedding_model_id", "embedding_dimension"):
            expected = fingerprint.get(key)
            actual = getattr(
                self,
                {"embedding_dimension": "dimension"}.get(key, key),
                None,
            )
            if expected is not None and actual != expected:
                return False, (
                    f"Index was built with {key}={actual!r} but the current configuration uses "
                    f"{expected!r}."
                )
        for key in ("chunk_size", "chunk_overlap"):
            expected = fingerprint.get(key)
            if expected is not None and getattr(self, key, None) != expected:
                return False, (
                    f"Index was built with {key}={getattr(self, key)!r} but the current "
                    f"configuration uses {expected!r}."
                )
        return True, ""


class VectorStore:
    """Persisted similarity index over chunk embeddings."""

    def __init__(self, index_dir: Path | str, *, dimension: int = 384) -> None:
        self.index_dir = Path(index_dir)
        self.dimension = dimension
        self._index: Any = None
        self._vectors: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self.metadata: IndexMetadata | None = None

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return "faiss" if FAISS_AVAILABLE else "numpy"

    @property
    def is_loaded(self) -> bool:
        return self._index is not None or self._vectors is not None

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        vectors: np.ndarray,
        chunk_ids: Sequence[str],
        metadata: IndexMetadata,
    ) -> None:
        """Construct an in-memory index from ``vectors`` (row i ↔ ``chunk_ids[i]``)."""
        if len(chunk_ids) != len(vectors):
            raise ValueError(
                f"vector/chunk-id length mismatch: {len(vectors)} vectors, {len(chunk_ids)} ids"
            )
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if matrix.shape[0] == 0:
            raise ValueError("cannot build an index with zero vectors")
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"expected {self.dimension}-dimensional vectors, received {matrix.shape[1]}"
            )

        # L2-normalise so inner product == cosine similarity.
        matrix = _normalise(matrix)

        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(matrix.shape[1])  # type: ignore[union-attr]
            index.add(matrix)
            self._index = index
            self._vectors = None
        else:
            self._index = None
            self._vectors = matrix

        self._chunk_ids = list(chunk_ids)
        metadata.dimension = matrix.shape[1]
        metadata.vector_count = matrix.shape[0]
        metadata.backend = self.backend
        metadata.built_at = utcnow()
        self.metadata = metadata

    def initialise_empty(self, metadata: IndexMetadata | None = None) -> None:
        """Create a valid, empty index ready for :meth:`add` calls.

        Used by the ingestion pipeline when the very first batch is written: we
        need an index object to append to, but ``build`` (correctly) refuses
        zero vectors.
        """
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self.dimension)  # type: ignore[union-attr]
            self._vectors = None
        else:
            self._vectors = np.zeros((0, self.dimension), dtype="float32")
            self._index = None
        self._chunk_ids = []
        meta = metadata or IndexMetadata(dimension=self.dimension)
        meta.dimension = self.dimension
        meta.vector_count = 0
        meta.backend = self.backend
        meta.built_at = utcnow()
        self.metadata = meta

    def add(self, vectors: np.ndarray, chunk_ids: Sequence[str]) -> None:
        """Append vectors to a loaded index (used for incremental ingestion)."""
        if not self.is_loaded:
            raise IndexNotFoundError("No index is loaded to append to.")
        matrix = _normalise(np.asarray(vectors, dtype="float32"))
        if matrix.shape[1] != self.dimension:
            raise IndexMismatchError(
                f"Expected {self.dimension}-dimensional vectors, received {matrix.shape[1]}.",
            )
        if FAISS_AVAILABLE and self._index is not None:
            self._index.add(matrix)
        else:
            if self._vectors is None:
                self._vectors = matrix
            else:
                self._vectors = np.vstack([self._vectors, matrix])
        self._chunk_ids.extend(chunk_ids)
        if self.metadata:
            self.metadata.vector_count = len(self._chunk_ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, *, directory: Path | None = None) -> Path:
        """Write index + sidecar to ``directory`` (defaults to ``index_dir``)."""
        if not self.is_loaded:
            raise IndexNotFoundError("Nothing to save — build or load an index first.")

        target = Path(directory) if directory else self.index_dir
        ensure_dir(target)

        if FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, str(target / INDEX_FILENAME))  # type: ignore[union-attr]
        else:
            assert self._vectors is not None
            np.save(target / "vectors.npy", self._vectors)

        meta = self.metadata or IndexMetadata(dimension=self.dimension)
        meta.vector_count = len(self._chunk_ids)
        meta.dimension = self.dimension
        meta.backend = self.backend

        (target / SIDECAR_FILENAME).write_text(
            json.dumps(
                {"metadata": meta.to_dict(), "chunk_ids": self._chunk_ids},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (target / BACKEND_FILENAME).write_text(self.backend, encoding="utf-8")
        return target

    def save_atomic(self, *, directory: Path | None = None) -> Path:
        """Write to a staging directory, validate, then swap into place.

        The previous index is only replaced after the new one has been written
        and read back successfully.  A failure at any point leaves the old
        index exactly as it was.
        """
        target = Path(directory) if directory else self.index_dir
        staging = target.parent / f"{target.name}.new"
        backup = target.parent / f"{target.name}.prev"

        if staging.exists():
            shutil.rmtree(staging)
        ensure_dir(staging)

        # 1. Write into staging.
        self.save(directory=staging)

        # 2. Validate by reading it back with a throwaway store.
        probe = VectorStore(staging, dimension=self.dimension)
        probe.load(strict_fingerprint=False)
        if probe.size != len(self._chunk_ids):
            shutil.rmtree(staging, ignore_errors=True)
            raise IndexCorruptError(
                f"Validation failed: wrote {len(self._chunk_ids)} vectors but read back {probe.size}."
            )

        # 3. Swap: current → .prev, staging → current.
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        if target.exists():
            target.rename(backup)
        staging.rename(target)
        shutil.rmtree(backup, ignore_errors=True)

        return target

    def load(
        self,
        *,
        directory: Path | None = None,
        fingerprint: dict[str, Any] | None = None,
        strict_fingerprint: bool = True,
    ) -> VectorStore:
        """Load the persisted index.

        Raises :class:`IndexNotFoundError` when nothing is saved and
        :class:`IndexMismatchError` when the sidecar disagrees with
        ``fingerprint`` (typically the current embedding configuration).
        """
        source = Path(directory) if directory else self.index_dir
        sidecar_path = source / SIDECAR_FILENAME
        if not sidecar_path.exists():
            raise IndexNotFoundError(
                "No saved index was found.",
                remedy="Load the synthetic corpus or upload documents to build one.",
                context={"index_dir": str(source)},
            )

        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            meta = IndexMetadata.from_dict(payload.get("metadata", {}))
            chunk_ids = list(payload.get("chunk_ids", []))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            raise IndexCorruptError(
                f"The index sidecar could not be read: {exc}",
                context={"path": str(sidecar_path)},
            ) from exc

        if fingerprint is not None:
            ok, why = meta.matches(fingerprint)
            if not ok:
                if strict_fingerprint:
                    raise IndexMismatchError(
                        why,
                        context={
                            "saved": meta.to_dict(),
                            "requested": dict(fingerprint),
                        },
                    )
                # Non-strict: accept but keep the mismatch visible.
                meta.corpus_fingerprint = f"MISMATCH: {why}"

        backend_now = self.backend
        vectors_path = source / "vectors.npy"
        faiss_path = source / INDEX_FILENAME

        if faiss_path.exists() and FAISS_AVAILABLE:
            self._index = faiss.read_index(str(faiss_path))  # type: ignore[union-attr]
            self._vectors = None
            if self._index.ntotal != len(chunk_ids):
                raise IndexCorruptError(
                    f"Index holds {self._index.ntotal} vectors but the sidecar lists "
                    f"{len(chunk_ids)} chunk ids.",
                )
        elif vectors_path.exists():
            self._vectors = np.load(vectors_path)
            self._index = None
            if self._vectors.shape[0] != len(chunk_ids):
                raise IndexCorruptError(
                    f"Vector file holds {self._vectors.shape[0]} rows but the sidecar lists "
                    f"{len(chunk_ids)} chunk ids.",
                )
        else:
            raise IndexCorruptError(
                "The index sidecar exists but the vector data is missing.",
                remedy="Rebuild the index from the Knowledge Base view.",
            )

        self._chunk_ids = chunk_ids
        self.metadata = meta
        self.dimension = meta.dimension or self.dimension
        if backend_now != meta.backend:
            self.metadata.backend = backend_now
        return self

    def exists(self, *, directory: Path | None = None) -> bool:
        source = Path(directory) if directory else self.index_dir
        return (source / SIDECAR_FILENAME).exists()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 6,
        allowed_chunk_ids: set[str] | None = None,
        pool_size: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_similarity)`` pairs, best first.

        ``allowed_chunk_ids`` applies metadata filters *after* the ANN search:
        we over-fetch into ``pool_size`` and then keep only permitted ids.  This
        keeps filtering exact while still using a single index pass.
        """
        if not self.is_loaded:
            raise IndexNotFoundError("No index is loaded.")
        if self.size == 0:
            return []

        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise IndexMismatchError(
                f"Query vector has {query.shape[1]} dimensions but the index expects {self.dimension}. "
                "The embedding model changed — rebuild the index."
            )
        query = _normalise(query)

        k = min(pool_size or max(top_k * 8, 40), self.size)
        k = max(k, min(top_k, self.size))

        if FAISS_AVAILABLE and self._index is not None:
            scores, indices = self._index.search(query, k)
            pairs = [
                (self._chunk_ids[i], float(s))
                for s, i in zip(scores[0], indices[0])
                if 0 <= i < len(self._chunk_ids)
            ]
        else:
            assert self._vectors is not None
            sims = (self._vectors @ query[0]).astype("float32")
            order = np.argsort(-sims)[:k]
            pairs = [(self._chunk_ids[i], float(sims[i])) for i in order]

        if allowed_chunk_ids is not None:
            pairs = [(cid, s) for cid, s in pairs if cid in allowed_chunk_ids]

        return pairs[:top_k]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def remove_chunks(self, chunk_ids_to_remove: Sequence[str]) -> int:
        """Physically drop vectors.  Requires a full rebuild (FAISS has no delete)."""
        drop = set(chunk_ids_to_remove)
        if not drop:
            return 0
        keep_idx = [i for i, cid in enumerate(self._chunk_ids) if cid not in drop]
        removed = len(self._chunk_ids) - len(keep_idx)
        if removed == 0:
            return 0

        if self._vectors is not None:
            self._vectors = self._vectors[keep_idx] if keep_idx else np.zeros((0, self.dimension), "float32")
            if self._vectors.shape[0] == 0:
                self._vectors = None
                self._index = None
                self._chunk_ids = []
                return removed
        elif self._index is not None:
            if not keep_idx:
                self._index = None
                self._chunk_ids = []
                return removed
            dense = self._index.reconstruct_n(0, self._index.ntotal)  # type: ignore[union-attr]
            dense = dense[keep_idx]
            new_index = faiss.IndexFlatIP(self.dimension)  # type: ignore[union-attr]
            new_index.add(dense)
            self._index = new_index

        self._chunk_ids = [self._chunk_ids[i] for i in keep_idx]
        if self.metadata:
            self.metadata.vector_count = len(self._chunk_ids)
        return removed

    def clear(self) -> None:
        self._index = None
        self._vectors = None
        self._chunk_ids = []
        self.metadata = None

    def delete_files(self, *, directory: Path | None = None) -> None:
        source = Path(directory) if directory else self.index_dir
        if source.exists():
            shutil.rmtree(source, ignore_errors=True)
        self.clear()

    def health(self, *, fingerprint: dict[str, Any] | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": self.backend,
            "faiss_available": FAISS_AVAILABLE,
            "loaded": self.is_loaded,
            "vectors": self.size,
            "dimension": self.dimension,
            "index_dir": str(self.index_dir),
            "exists_on_disk": self.exists(),
        }
        if self.metadata:
            info["built_at"] = self.metadata.built_at
            info["embedding_model_id"] = self.metadata.embedding_model_id
            if fingerprint:
                ok, why = self.metadata.matches(fingerprint)
                info["matches_configuration"] = ok
                info["mismatch_reason"] = why
        return info


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, safe against zero vectors."""
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


__all__ = [
    "FAISS_AVAILABLE",
    "INDEX_FILENAME",
    "SIDECAR_FILENAME",
    "IndexMetadata",
    "VectorStore",
]
