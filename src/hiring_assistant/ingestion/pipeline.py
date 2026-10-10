"""End-to-end ingestion: validate → parse → normalise → chunk → embed → index.

The pipeline is the only place that writes to both the metadata store and the
vector index, and it keeps them consistent:

* Files are hashed first.  An unchanged file (same hash, same embedding model,
  same chunking config) is skipped rather than re-embedded — this is the
  "avoid unnecessary re-embedding" requirement and it is what makes reloading
  the corpus instant after the first run.
* A document is only marked ``indexed`` once its chunks are durably in both
  stores.
* A single bad file never aborts the batch; it is recorded as ``failed`` with
  its error and remedy, and the rest of the run continues.

Embedding happens for the whole batch at once so the provider can batch
efficiently, then vectors are written to FAISS in the same order as the chunk
ids, giving a stable index-id ↔ chunk-id mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Settings
from ..errors import (
    EmptyDocumentError,
    FileTooLargeError,
    HiringAssistantError,
    IngestionError,
    ParseError,
    UnsupportedFileTypeError,
)
from ..models import Chunk, Document, new_id, utcnow
from ..retrieval.embeddings import EmbeddingProvider
from ..safety.injection import scan_for_injection
from ..storage.database import MetadataStore
from ..storage.vector_store import IndexMetadata, VectorStore
from ..utils import (
    content_hash,
    extension_of,
    human_size,
    safe_filename,
    sha256_bytes,
)
from .chunking import ChunkingConfig, chunk_sections
from .parsers import ParseResult, parse_document

# --------------------------------------------------------------------------
# Report types
# --------------------------------------------------------------------------


@dataclass
class IngestItemResult:
    """Per-file outcome, rendered as a row in the Knowledge Base view."""

    filename: str
    document_id: str | None
    status: str  # indexed | skipped | failed | duplicate
    chunks: int = 0
    chars: int = 0
    collection: str = ""
    message: str = ""
    remedy: str = ""
    warnings: list[str] = field(default_factory=list)
    injection_flags: list[str] = field(default_factory=list)
    file_type: str = ""
    size_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.status in {"indexed", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "document_id": self.document_id,
            "status": self.status,
            "chunks": self.chunks,
            "chars": self.chars,
            "collection": self.collection,
            "message": self.message,
            "remedy": self.remedy,
            "warnings": self.warnings,
            "injection_flags": self.injection_flags,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
        }


@dataclass
class IngestReport:
    """Aggregate result of one ingestion run."""

    items: list[IngestItemResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    index_rebuilt: bool = False
    total_chunks_indexed: int = 0
    total_chunks_reused: int = 0
    embedding_model: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def indexed(self) -> list[IngestItemResult]:
        return [i for i in self.items if i.status == "indexed"]

    @property
    def skipped(self) -> list[IngestItemResult]:
        return [i for i in self.items if i.status == "skipped"]

    @property
    def failed(self) -> list[IngestItemResult]:
        return [i for i in self.items if i.status == "failed"]

    @property
    def duplicates(self) -> list[IngestItemResult]:
        return [i for i in self.items if i.status == "duplicate"]

    @property
    def summary(self) -> str:
        parts = []
        if self.indexed:
            parts.append(f"{len(self.indexed)} indexed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} unchanged")
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} duplicate")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts) or "nothing to ingest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "duration_seconds": self.duration_seconds,
            "index_rebuilt": self.index_rebuilt,
            "total_chunks_indexed": self.total_chunks_indexed,
            "total_chunks_reused": self.total_chunks_reused,
            "embedding_model": self.embedding_model,
            "warnings": self.warnings,
            "summary": self.summary,
        }


@dataclass
class SourceFile:
    """A file queued for ingestion."""

    filename: str
    data: bytes
    collection: str
    origin: str = "synthetic"  # synthetic | upload

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class IngestionPipeline:
    """Coordinates parsing, chunking, embedding and indexing."""

    def __init__(
        self,
        settings: Settings,
        store: MetadataStore,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, filename: str, data: bytes) -> None:
        """Raise a specific :class:`IngestionError` for an unacceptable file."""
        extension = extension_of(filename)
        allowed = tuple(self.settings.allowed_extensions)
        if extension not in allowed:
            raise UnsupportedFileTypeError(
                f"'{filename}' has an unsupported extension ('{extension or 'none'}').",
                remedy=f"Supported types: {', '.join(allowed)}.",
                context={"filename": filename},
            )
        if not data:
            raise EmptyDocumentError(
                f"'{filename}' is empty.",
                remedy="Upload a file that contains text.",
            )
        if len(data) > self.settings.max_upload_bytes:
            raise FileTooLargeError(
                f"'{filename}' is {human_size(len(data))}, above the "
                f"{self.settings.max_upload_mb} MB limit.",
                remedy=(
                    "Reduce the file size, or raise the MAX_UPLOAD_MB limit in your .env file."
                ),
                context={"size_bytes": len(data), "limit_bytes": self.settings.max_upload_bytes},
            )
        if b"\x00" in data[:4096] and extension in {".txt", ".md", ".markdown", ".csv", ".json"}:
            raise ParseError(
                f"'{filename}' looks like a binary file with a text extension.",
                remedy="Re-export the file as plain text, or use PDF/DOCX.",
            )

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def _hash(self, data: bytes) -> str:
        """Hash content *and* the configuration that affects its vectors."""
        return content_hash(
            [
                sha256_bytes(data),
                self.settings.embedding_model_id,
                str(self.settings.chunk_size),
                str(self.settings.chunk_overlap),
            ]
        )

    def ingest_file(
        self,
        source: SourceFile,
        *,
        force: bool = False,
    ) -> tuple[IngestItemResult, list[Chunk]]:
        """Validate, parse, chunk and prepare one file.  Does not embed."""
        filename = safe_filename(source.filename)
        result = IngestItemResult(
            filename=filename,
            document_id=None,
            status="failed",
            collection=source.collection,
            file_type=extension_of(filename).lstrip("."),
            size_bytes=source.size_bytes,
        )

        try:
            self.validate(filename, source.data)
        except IngestionError as exc:
            result.message = exc.message
            result.remedy = exc.remedy
            return result, []

        source_hash = self._hash(source.data)
        existing = self.store.find_by_hash(source_hash)

        if existing and not force:
            if existing.status == "indexed" and existing.chunk_count > 0:
                result.document_id = existing.document_id
                result.status = "skipped"
                result.chunks = existing.chunk_count
                result.chars = existing.char_count
                result.message = (
                    "Unchanged since the last run (identical content hash, embedding model and "
                    "chunking configuration) — existing vectors were reused."
                )
                return result, []

        if existing and force:
            # Re-ingest in place: drop the old vectors first (read the chunk
            # ids *before* deleting the rows), then the metadata.
            stale_chunk_ids = [c.chunk_id for c in self.store.chunks_for_document(existing.document_id)]
            if self.vector_store.is_loaded and stale_chunk_ids:
                self.vector_store.remove_chunks(stale_chunk_ids)
            self.store.delete_document(existing.document_id)

        # -- parse ------------------------------------------------------
        document_id = new_id("doc")
        try:
            parsed: ParseResult = parse_document(
                source.data, filename, allowed_extensions=tuple(self.settings.allowed_extensions)
            )
        except (UnsupportedFileTypeError, EmptyDocumentError, ParseError) as exc:
            document = Document(
                document_id=document_id,
                filename=filename,
                file_type=extension_of(filename).lstrip("."),
                collection=source.collection,
                source_hash=source_hash,
                status="failed",
                error=exc.message,
                synthetic=source.origin == "synthetic",
                origin=source.origin,
            )
            self.store.upsert_document(document)
            result.document_id = document_id
            result.message = exc.message
            result.remedy = exc.remedy
            return result, []
        except HiringAssistantError as exc:
            result.message = exc.message
            result.remedy = exc.remedy
            return result, []
        except Exception as exc:  # unexpected parser bug — surface, do not crash the batch
            result.message = f"Unexpected error while parsing: {exc}"
            result.remedy = "Re-export the file and try again; if it persists, report the file."
            return result, []

        warnings = list(parsed.warnings)

        # -- untrusted-content scan -------------------------------------
        injection_flags: list[str] = []
        scan = scan_for_injection(parsed.full_text)
        if scan.detected:
            injection_flags = sorted({f.kind for f in scan.findings})
            warnings.append(
                f"Prompt-injection patterns detected ({', '.join(injection_flags)}). "
                "The text was treated as data and neutralised before reaching the model."
            )

        # -- chunk ------------------------------------------------------
        config = ChunkingConfig(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            min_chunk_chars=self.settings.min_chunk_chars,
        )
        chunks, stats = chunk_sections(
            parsed.sections,
            document_id=document_id,
            filename=filename,
            collection=source.collection,
            file_type=parsed.file_type,
            embedding_model=self.settings.embedding_model_id,
            config=config,
            synthetic=source.origin == "synthetic",
        )
        warnings.extend(stats.warnings)

        if not chunks:
            document = Document(
                document_id=document_id,
                filename=filename,
                file_type=parsed.file_type,
                collection=source.collection,
                source_hash=source_hash,
                char_count=parsed.char_count,
                status="failed",
                warnings=warnings,
                error="No chunks were produced from this document.",
                synthetic=source.origin == "synthetic",
                origin=source.origin,
            )
            self.store.upsert_document(document)
            result.document_id = document_id
            result.message = document.error or ""
            result.remedy = "The document is too short to chunk. Supply a longer document."
            return result, []

        document = Document(
            document_id=document_id,
            filename=filename,
            file_type=parsed.file_type,
            collection=source.collection,
            source_hash=source_hash,
            char_count=parsed.char_count,
            chunk_count=len(chunks),
            page_count=parsed.page_count,
            status="pending",
            warnings=warnings,
            synthetic=source.origin == "synthetic",
            origin=source.origin,
        )
        self.store.upsert_document(document)

        result.document_id = document_id
        result.status = "pending"
        result.chunks = len(chunks)
        result.chars = parsed.char_count
        result.warnings = warnings
        result.injection_flags = injection_flags
        result.message = f"Parsed {parsed.char_count:,} characters into {len(chunks)} chunks."
        return result, chunks

    def _index_metadata(self, dimension: int) -> IndexMetadata:
        """Index metadata describing the *current* configuration."""
        return IndexMetadata(
            embedding_model=self.settings.embedding_model,
            embedding_model_id=self.settings.embedding_model_id,
            dimension=dimension,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def ingest_many(
        self,
        sources: Sequence[SourceFile],
        *,
        force: bool = False,
        progress: Callable[[float, str], None] | None = None,
        persist: bool = True,
    ) -> IngestReport:
        """Ingest a batch of files and rebuild the index once at the end."""
        import time

        started = time.perf_counter()
        report = IngestReport(embedding_model=self.settings.embedding_model_id)

        prepared: list[tuple[IngestItemResult, list[Chunk]]] = []
        seen_hashes: set[str] = set()

        total = max(1, len(sources))
        for position, source in enumerate(sources, start=1):
            if progress:
                progress((position - 1) / total, f"Parsing {source.filename}")
            item, chunks = self.ingest_file(source, force=force)

            # Within-batch duplicate detection (same file uploaded twice).
            if item.status == "pending" and chunks:
                digest = content_hash([sha256_bytes(source.data), self.settings.embedding_model_id])
                if digest in seen_hashes:
                    item.status = "duplicate"
                    item.message = "Duplicate of another file in this same batch; skipped."
                    chunks = []
                else:
                    seen_hashes.add(digest)

            prepared.append((item, chunks))

        # -- embed every new chunk in one batched call ------------------
        pending_chunks: list[Chunk] = []
        for item, chunks in prepared:
            if item.status == "pending":
                pending_chunks.extend(chunks)

        if progress:
            progress(0.85, f"Embedding {len(pending_chunks)} chunks")

        if pending_chunks:
            try:
                vectors = self.embedder.embed_documents(
                    [c.text for c in pending_chunks],
                    batch_size=self.settings.embedding_batch_size,
                )
            except HiringAssistantError as exc:
                for item, chunks in prepared:
                    if item.status == "pending":
                        item.status = "failed"
                        item.message = f"Embedding failed: {exc.message}"
                        item.remedy = exc.remedy
                        if item.document_id:
                            doc = self.store.get_document(item.document_id)
                            if doc:
                                doc.status = "failed"
                                doc.error = item.message
                                self.store.upsert_document(doc)
                report.items = [item for item, _ in prepared]
                report.duration_seconds = time.perf_counter() - started
                report.warnings.append(f"Embedding failed: {exc.message}")
                return report

            # Persist chunks + index together.
            if persist:
                self._persist_batch(prepared, pending_chunks, vectors, report)
            else:
                for item, _ in prepared:
                    if item.status == "pending":
                        item.status = "indexed"
                report.total_chunks_indexed = len(pending_chunks)
        else:
            report.total_chunks_reused = sum(i.chunks for i in (item for item, _ in prepared) if item.status == "skipped")

        if progress:
            progress(1.0, "Done")

        report.items = [item for item, _ in prepared]
        report.duration_seconds = time.perf_counter() - started
        return report

    def _persist_batch(
        self,
        prepared: list[tuple[IngestItemResult, list[Chunk]]],
        pending_chunks: list[Chunk],
        vectors: np.ndarray,
        report: IngestReport,
    ) -> None:
        """Write chunks to SQLite and vectors to FAISS, keeping both in step."""
        # Ensure a loaded index exists.
        if not self.vector_store.is_loaded:
            if self.vector_store.exists():
                try:
                    self.vector_store.load(
                        fingerprint=self.settings.index_fingerprint(),
                        strict_fingerprint=False,
                    )
                except HiringAssistantError:
                    self.vector_store.clear()
            if not self.vector_store.is_loaded:
                self.vector_store.dimension = vectors.shape[1]
                self.vector_store.initialise_empty(self._index_metadata(vectors.shape[1]))

        start_row = self.vector_store.size
        self.vector_store.add(vectors, [c.chunk_id for c in pending_chunks])

        mapping = {
            chunk.chunk_id: start_row + offset for offset, chunk in enumerate(pending_chunks)
        }
        for chunk in pending_chunks:
            chunk.index_id = mapping[chunk.chunk_id]

        self.store.insert_chunks(pending_chunks)
        self.store.set_index_ids(mapping)

        # Mark documents indexed.
        for item, _ in prepared:
            if item.status != "pending" or not item.document_id:
                continue
            doc = self.store.get_document(item.document_id)
            if doc:
                doc.status = "indexed"
                doc.indexed_at = utcnow()
                doc.chunk_count = item.chunks
                doc.warnings = item.warnings
                self.store.upsert_document(doc)
            item.status = "indexed"

        self.vector_store.metadata = self.vector_store.metadata or IndexMetadata()
        self.vector_store.metadata.embedding_model = self.settings.embedding_model
        self.vector_store.metadata.embedding_model_id = self.settings.embedding_model_id
        self.vector_store.metadata.chunk_size = self.settings.chunk_size
        self.vector_store.metadata.chunk_overlap = self.settings.chunk_overlap
        self.vector_store.metadata.corpus_fingerprint = content_hash(
            [self.settings.embedding_model_id, str(self.store.count_chunks())]
        )

        self.vector_store.save_atomic()
        self.store.set_state(
            "index_built_at", self.vector_store.metadata.built_at
        )
        self.store.set_state("index_vector_count", self.vector_store.size)
        report.index_rebuilt = True
        report.total_chunks_indexed = len(pending_chunks)

    # ------------------------------------------------------------------
    # Removal & rebuild
    # ------------------------------------------------------------------

    def remove_document(self, document_id: str) -> dict[str, Any]:
        """Remove a document and physically drop its vectors.

        FAISS has no delete, so we rebuild the flat index from the surviving
        vectors.  This is cheap at this corpus size and guarantees removed
        documents can never be retrieved.
        """
        chunk_ids = [c.chunk_id for c in self.store.chunks_for_document(document_id)]
        removed_vectors = 0
        if self.vector_store.is_loaded and chunk_ids:
            removed_vectors = self.vector_store.remove_chunks(chunk_ids)
            self.vector_store.save_atomic()

        chunks_deleted = self.store.delete_document(document_id)

        if self.vector_store.is_loaded and self.vector_store.size == 0:
            self.vector_store.delete_files()
            self.store.clear_state()

        return {
            "document_id": document_id,
            "chunks_deleted": chunks_deleted,
            "vectors_removed": removed_vectors,
            "index_size_after": self.vector_store.size if self.vector_store.is_loaded else 0,
        }

    def reindex_document(self, document_id: str) -> IngestItemResult:
        """Force re-embedding of one document from its stored chunks."""
        document = self.store.get_document(document_id)
        if not document:
            return IngestItemResult(
                filename=document_id,
                document_id=None,
                status="failed",
                message="Document not found in the metadata store.",
                remedy="Refresh the Knowledge Base inventory.",
            )

        chunks = self.store.chunks_for_document(document_id)
        if not chunks:
            return IngestItemResult(
                filename=document.filename,
                document_id=document_id,
                status="failed",
                message="No stored chunks to re-embed.",
                remedy="Re-upload the original file.",
            )

        vectors = self.embedder.embed_documents(
            [c.text for c in chunks], batch_size=self.settings.embedding_batch_size
        )
        if self.vector_store.is_loaded:
            self.vector_store.remove_chunks([c.chunk_id for c in chunks])
        else:
            self.vector_store.dimension = vectors.shape[1]
            self.vector_store.initialise_empty(self._index_metadata(vectors.shape[1]))

        start = self.vector_store.size
        self.vector_store.add(vectors, [c.chunk_id for c in chunks])
        self.store.set_index_ids({c.chunk_id: start + i for i, c in enumerate(chunks)})
        self.store.set_embedding_model(self.settings.embedding_model_id)
        self.vector_store.save_atomic()

        document.status = "indexed"
        document.indexed_at = utcnow()
        document.error = None
        self.store.upsert_document(document)

        return IngestItemResult(
            filename=document.filename,
            document_id=document_id,
            status="indexed",
            chunks=len(chunks),
            chars=document.char_count,
            collection=document.collection,
            message=f"Re-embedded {len(chunks)} chunks with {self.settings.embedding_model_id}.",
        )

    def rebuild_all(
        self,
        *,
        progress: Callable[[float, str], None] | None = None,
    ) -> IngestReport:
        """Re-embed every stored chunk with the current embedding model."""
        import time

        started = time.perf_counter()
        report = IngestReport(embedding_model=self.settings.embedding_model_id)

        documents = self.store.list_documents()
        all_chunks: list[Chunk] = []
        for document in documents:
            all_chunks.extend(self.store.chunks_for_document(document.document_id))

        if not all_chunks:
            report.warnings.append("Nothing to rebuild — the knowledge base is empty.")
            report.duration_seconds = time.perf_counter() - started
            return report

        if progress:
            progress(0.3, f"Embedding {len(all_chunks)} chunks")

        vectors = self.embedder.embed_documents(
            [c.text for c in all_chunks], batch_size=self.settings.embedding_batch_size
        )

        # Build a brand-new index in memory, then swap atomically on disk.
        fresh = VectorStore(self.settings.index_dir, dimension=vectors.shape[1])
        fresh.build(
            vectors,
            [c.chunk_id for c in all_chunks],
            IndexMetadata(
                embedding_model=self.settings.embedding_model,
                embedding_model_id=self.settings.embedding_model_id,
                dimension=vectors.shape[1],
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            ),
        )
        fresh.save_atomic()

        self.vector_store.clear()
        self.vector_store.load(fingerprint=self.settings.index_fingerprint(), strict_fingerprint=False)
        self.store.set_index_ids({c.chunk_id: i for i, c in enumerate(all_chunks)})
        self.store.set_embedding_model(self.settings.embedding_model_id)
        self.store.set_state("index_built_at", fresh.metadata.built_at if fresh.metadata else utcnow())
        self.store.set_state("index_vector_count", len(all_chunks))

        for document in documents:
            document.status = "indexed"
            document.indexed_at = utcnow()
            self.store.upsert_document(document)

        report.index_rebuilt = True
        report.total_chunks_indexed = len(all_chunks)
        report.items = [
            IngestItemResult(
                filename=document.filename,
                document_id=document.document_id,
                status="indexed",
                chunks=document.chunk_count,
                collection=document.collection,
                message="Re-embedded in a full index rebuild.",
            )
            for document in documents
        ]
        report.duration_seconds = time.perf_counter() - started
        if progress:
            progress(1.0, "Rebuild complete")
        return report


__all__ = [
    "IngestItemResult",
    "IngestReport",
    "IngestionPipeline",
    "SourceFile",
]
