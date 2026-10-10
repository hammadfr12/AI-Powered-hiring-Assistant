"""SQLite metadata store.

Plain ``sqlite3`` rather than an ORM: the schema is small, the queries are
simple, and a zero-dependency store keeps the "runs on a fresh clone with no
setup" promise easy to hold.  Connections are per-thread because Streamlit
reruns script blocks on different threads; each thread gets its own handle via
``threading.local``.

Tables (see docs/DATA_DICTIONARY.md for full column documentation):
``documents``, ``chunks``, ``conversations``, ``messages``, ``evaluation_cases``,
``evaluation_runs``, ``index_state``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..errors import HiringAssistantError
from ..models import Chunk, Citation, Conversation, Document, Message, utcnow
from ..utils import ensure_dir

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id    TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    file_type      TEXT NOT NULL,
    collection     TEXT NOT NULL,
    source_hash    TEXT NOT NULL,
    char_count     INTEGER NOT NULL DEFAULT 0,
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    page_count     INTEGER,
    status         TEXT NOT NULL DEFAULT 'pending',
    warnings       TEXT NOT NULL DEFAULT '[]',
    error          TEXT,
    synthetic      INTEGER NOT NULL DEFAULT 1,
    origin         TEXT NOT NULL DEFAULT 'synthetic',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    indexed_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_hash       ON documents(source_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status     ON documents(status);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    filename        TEXT NOT NULL DEFAULT '',
    collection      TEXT NOT NULL DEFAULT '',
    file_type       TEXT NOT NULL DEFAULT '',
    section         TEXT,
    page            INTEGER,
    char_start      INTEGER NOT NULL DEFAULT 0,
    char_end        INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT NOT NULL DEFAULT '',
    index_id        INTEGER,
    synthetic       INTEGER NOT NULL DEFAULT 1,
    token_estimate  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_index_id ON chunks(index_id);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);

CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'New conversation',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    cited_chunk_ids TEXT NOT NULL DEFAULT '[]',
    metadata        TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS evaluation_cases (
    case_id               TEXT PRIMARY KEY,
    query                 TEXT NOT NULL,
    category              TEXT NOT NULL,
    expected_source_ids   TEXT NOT NULL DEFAULT '[]',
    expected_answer_points TEXT NOT NULL DEFAULT '[]',
    should_abstain        INTEGER NOT NULL DEFAULT 0,
    expected_blocked      INTEGER NOT NULL DEFAULT 0,
    notes                 TEXT NOT NULL DEFAULT '',
    candidate_id          TEXT,
    job_id                TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id           TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    config           TEXT NOT NULL DEFAULT '{}',
    metrics          TEXT NOT NULL DEFAULT '{}',
    failures         TEXT NOT NULL DEFAULT '[]',
    duration_seconds REAL NOT NULL DEFAULT 0,
    results          TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON evaluation_runs(created_at);

CREATE TABLE IF NOT EXISTS index_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _from_json(raw: str | None, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


class MetadataStore:
    """Thread-safe wrapper around a single SQLite database file."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialise()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @property
    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _initialise(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            finally:
                conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connection
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        doc.updated_at = utcnow()
        self.connection.execute(
            """
            INSERT INTO documents (
                document_id, filename, file_type, collection, source_hash, char_count,
                chunk_count, page_count, status, warnings, error, synthetic, origin,
                created_at, updated_at, indexed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename=excluded.filename, file_type=excluded.file_type,
                collection=excluded.collection, source_hash=excluded.source_hash,
                char_count=excluded.char_count, chunk_count=excluded.chunk_count,
                page_count=excluded.page_count, status=excluded.status,
                warnings=excluded.warnings, error=excluded.error,
                synthetic=excluded.synthetic, origin=excluded.origin,
                updated_at=excluded.updated_at, indexed_at=excluded.indexed_at
            """,
            (
                doc.document_id,
                doc.filename,
                doc.file_type,
                doc.collection,
                doc.source_hash,
                doc.char_count,
                doc.chunk_count,
                doc.page_count,
                doc.status,
                _to_json(doc.warnings),
                doc.error,
                1 if doc.synthetic else 0,
                doc.origin,
                doc.created_at,
                doc.updated_at,
                doc.indexed_at,
            ),
        )

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            document_id=row["document_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            collection=row["collection"],
            source_hash=row["source_hash"],
            char_count=row["char_count"],
            chunk_count=row["chunk_count"],
            page_count=row["page_count"],
            status=row["status"],
            warnings=_from_json(row["warnings"], []),
            error=row["error"],
            synthetic=bool(row["synthetic"]),
            origin=row["origin"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            indexed_at=row["indexed_at"],
        )

    def get_document(self, document_id: str) -> Document | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def find_by_hash(self, source_hash: str) -> Document | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE source_hash = ? LIMIT 1", (source_hash,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(
        self,
        *,
        collection: str | None = None,
        status: str | None = None,
        origin: str | None = None,
        limit: int | None = None,
    ) -> list[Document]:
        sql = "SELECT * FROM documents WHERE 1=1"
        params: list[Any] = []
        if collection:
            sql += " AND collection = ?"
            params.append(collection)
        if status:
            sql += " AND status = ?"
            params.append(status)
        if origin:
            sql += " AND origin = ?"
            params.append(origin)
        sql += " ORDER BY collection, filename"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_document(r) for r in self.connection.execute(sql, params)]

    def count_documents(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"]) if row else 0

    def collections_in_use(self) -> list[tuple[str, int]]:
        rows = self.connection.execute(
            "SELECT collection, COUNT(*) AS n FROM documents GROUP BY collection ORDER BY collection"
        ).fetchall()
        return [(r["collection"], int(r["n"])) for r in rows]

    def delete_document(self, document_id: str) -> int:
        """Remove a document and its chunks.  Returns chunks deleted."""
        with self._lock:
            conn = self.connection
            conn.execute("BEGIN")
            try:
                n = conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?", (document_id,)
                ).fetchone()["n"]
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return int(n)

    def delete_all_documents(self) -> None:
        with self._lock:
            conn = self.connection
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM documents")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def insert_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            (
                c.chunk_id,
                c.document_id,
                c.chunk_index,
                c.text,
                c.filename,
                c.collection,
                c.file_type,
                c.section,
                c.page,
                c.char_start,
                c.char_end,
                c.embedding_model,
                c.index_id,
                1 if c.synthetic else 0,
                c.token_estimate,
            )
            for c in chunks
        ]
        self.connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id, document_id, chunk_index, text, filename, collection, file_type,
                section, page, char_start, char_end, embedding_model, index_id, synthetic,
                token_estimate
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                text=excluded.text, index_id=excluded.index_id,
                embedding_model=excluded.embedding_model, section=excluded.section,
                page=excluded.page, token_estimate=excluded.token_estimate
            """,
            rows,
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            filename=row["filename"],
            collection=row["collection"],
            file_type=row["file_type"],
            section=row["section"],
            page=row["page"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            embedding_model=row["embedding_model"],
            index_id=row["index_id"],
            synthetic=bool(row["synthetic"]),
            token_estimate=row["token_estimate"],
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self.connection.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        ids = [c for c in chunk_ids if c]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
        ).fetchall()
        by_id = {r["chunk_id"]: self._row_to_chunk(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        rows = self.connection.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def all_chunks(self, *, collections: Sequence[str] | None = None) -> list[Chunk]:
        sql = "SELECT * FROM chunks"
        params: list[Any] = []
        if collections:
            placeholders = ",".join("?" for _ in collections)
            sql += f" WHERE collection IN ({placeholders})"
            params.extend(collections)
        sql += " ORDER BY collection, filename, chunk_index"
        return [self._row_to_chunk(r) for r in self.connection.execute(sql, params)]

    def count_chunks(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def chunk_count_by_collection(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT collection, COUNT(*) AS n FROM chunks GROUP BY collection"
        ).fetchall()
        return {r["collection"]: int(r["n"]) for r in rows}

    def set_index_ids(self, mapping: dict[str, int]) -> None:
        """Persist the chunk_id → FAISS row mapping."""
        if not mapping:
            return
        self.connection.executemany(
            "UPDATE chunks SET index_id = ? WHERE chunk_id = ?",
            [(idx, cid) for cid, idx in mapping.items()],
        )

    def set_embedding_model(self, model_id: str) -> None:
        self.connection.execute("UPDATE chunks SET embedding_model = ?", (model_id,))

    # ------------------------------------------------------------------
    # Conversations & messages
    # ------------------------------------------------------------------

    def ensure_conversation(self, session_id: str, title: str = "New conversation") -> Conversation:
        row = self.connection.execute(
            "SELECT * FROM conversations WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            return Conversation(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        conv = Conversation(session_id=session_id, title=title)
        self.connection.execute(
            "INSERT INTO conversations(session_id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (conv.session_id, conv.title, conv.created_at, conv.updated_at),
        )
        return conv

    def rename_conversation(self, session_id: str, title: str) -> None:
        self.connection.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE session_id = ?",
            (title[:120], utcnow(), session_id),
        )

    def add_message(self, message: Message) -> None:
        self.connection.execute(
            """
            INSERT INTO messages(message_id, session_id, role, content, cited_chunk_ids, metadata, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                message.message_id,
                message.session_id,
                message.role,
                message.content,
                _to_json(message.cited_chunk_ids),
                _to_json(message.metadata),
                message.created_at,
            ),
        )
        self.connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
            (message.created_at, message.session_id),
        )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Messages in a session, oldest first.

        When ``limit`` is given we take the most recent N and then re-order them
        chronologically, so a caller asking for "the last 8 messages" gets the
        last 8 in reading order.  ``rowid`` is selected explicitly in the inner
        query because it is not part of ``*`` and is therefore unavailable to
        the outer ORDER BY.
        """
        params: list[Any] = [session_id]
        if limit:
            sql = (
                "SELECT * FROM (SELECT rowid AS _rid, * FROM messages WHERE session_id = ? "
                "ORDER BY created_at DESC, _rid DESC LIMIT ?) ORDER BY created_at, _rid"
            )
            params.append(limit)
        else:
            sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, rowid"
        rows = self.connection.execute(sql, params).fetchall()
        return [
            Message(
                message_id=r["message_id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                cited_chunk_ids=_from_json(r["cited_chunk_ids"], []),
                metadata=_from_json(r["metadata"], {}),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def clear_conversation(self, session_id: str) -> int:
        """Delete every message in a session.  Returns messages removed."""
        row = self.connection.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        count = int(row["n"]) if row else 0
        self.connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.connection.execute(
            "UPDATE conversations SET title = 'New conversation', updated_at = ? WHERE session_id = ?",
            (utcnow(), session_id),
        )
        return count

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        rows = self.connection.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Conversation(
                session_id=r["session_id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Citations (resolved from stored chunk ids)
    # ------------------------------------------------------------------

    def resolve_citations(self, chunk_ids: Sequence[str], scores: dict[str, float] | None = None) -> list[Citation]:
        scores = scores or {}
        citations: list[Citation] = []
        for chunk in self.get_chunks_by_ids(chunk_ids):
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    collection=chunk.collection,
                    locator=chunk.locator,
                    excerpt=chunk.excerpt(360),
                    score=float(scores.get(chunk.chunk_id, 0.0)),
                    synthetic=chunk.synthetic,
                )
            )
        return citations

    # ------------------------------------------------------------------
    # Evaluation persistence
    # ------------------------------------------------------------------

    def upsert_evaluation_cases(self, cases: Sequence[Any]) -> None:
        rows = [
            (
                c.case_id,
                c.query,
                c.category,
                _to_json(c.expected_source_ids),
                _to_json(c.expected_answer_points),
                1 if c.should_abstain else 0,
                1 if c.expected_blocked else 0,
                c.notes,
                c.candidate_id,
                c.job_id,
            )
            for c in cases
        ]
        self.connection.executemany(
            """
            INSERT INTO evaluation_cases (
                case_id, query, category, expected_source_ids, expected_answer_points,
                should_abstain, expected_blocked, notes, candidate_id, job_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(case_id) DO UPDATE SET
                query=excluded.query, category=excluded.category,
                expected_source_ids=excluded.expected_source_ids,
                expected_answer_points=excluded.expected_answer_points,
                should_abstain=excluded.should_abstain, expected_blocked=excluded.expected_blocked,
                notes=excluded.notes, candidate_id=excluded.candidate_id, job_id=excluded.job_id
            """,
            rows,
        )

    def save_evaluation_run(self, report: Any) -> None:
        self.connection.execute(
            """
            INSERT INTO evaluation_runs(run_id, created_at, config, metrics, failures, duration_seconds, results)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                metrics=excluded.metrics, failures=excluded.failures,
                duration_seconds=excluded.duration_seconds, results=excluded.results
            """,
            (
                report.run_id,
                report.created_at,
                _to_json(report.config),
                _to_json(report.metrics),
                _to_json(report.failures),
                report.duration_seconds,
                _to_json([r.to_dict() for r in report.results]),
            ),
        )

    def latest_evaluation_run(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "config": _from_json(row["config"], {}),
            "metrics": _from_json(row["metrics"], {}),
            "failures": _from_json(row["failures"], []),
            "duration_seconds": row["duration_seconds"],
            "results": _from_json(row["results"], []),
        }

    def list_evaluation_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT run_id, created_at, metrics, duration_seconds FROM evaluation_runs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "created_at": r["created_at"],
                "metrics": _from_json(r["metrics"], {}),
                "duration_seconds": r["duration_seconds"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Key/value state
    # ------------------------------------------------------------------

    def set_state(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT INTO index_state(key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, _to_json(value), utcnow()),
        )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value FROM index_state WHERE key = ?", (key,)
        ).fetchone()
        return _from_json(row["value"], default) if row else default

    def clear_state(self) -> None:
        self.connection.execute("DELETE FROM index_state")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        try:
            docs = self.count_documents()
            chunks = self.count_chunks()
            return {
                "ok": True,
                "db_path": str(self.db_path),
                "documents": docs,
                "chunks": chunks,
                "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }
        except Exception as exc:  # pragma: no cover - defensive
            raise HiringAssistantError(
                f"Metadata store is unreadable: {exc}",
                remedy="Delete data/hiring_assistant.sqlite3 and reload the synthetic corpus.",
            ) from exc


__all__ = ["SCHEMA_VERSION", "MetadataStore"]
