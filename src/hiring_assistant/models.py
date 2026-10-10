"""Typed domain objects shared across every layer.

These dataclasses mirror the SQLite schema in :mod:`hiring_assistant.storage`
one-to-one, so a row can round-trip through the database without a bespoke
mapper.  They are deliberately plain — no behaviour beyond validation and
serialisation — which keeps the retrieval and generation layers easy to test.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

EvidenceStatus = Literal["supported", "partial", "not_found"]
MessageRole = Literal["user", "assistant", "system"]


def utcnow() -> str:
    """ISO-8601 UTC timestamp, second precision, always with a ``Z`` suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Documents & chunks
# --------------------------------------------------------------------------


@dataclass
class Document:
    """A source file that has been (or failed to be) ingested."""

    document_id: str
    filename: str
    file_type: str
    collection: str
    source_hash: str
    char_count: int = 0
    chunk_count: int = 0
    page_count: int | None = None
    status: str = "pending"  # pending | indexed | skipped | failed | removed
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    synthetic: bool = True
    origin: str = "synthetic"  # synthetic | upload
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    indexed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def warning_text(self) -> str:
        return "; ".join(self.warnings) if self.warnings else ""

    @property
    def is_indexed(self) -> bool:
        return self.status == "indexed" and self.chunk_count > 0


@dataclass
class Chunk:
    """A retrievable passage with full provenance back to its source."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    filename: str = ""
    collection: str = ""
    file_type: str = ""
    section: str | None = None
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    embedding_model: str = ""
    index_id: int | None = None
    synthetic: bool = True
    token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def locator(self) -> str:
        """Human-readable location string used on citation chips."""
        bits: list[str] = []
        if self.section:
            bits.append(str(self.section))
        if self.page:
            bits.append(f"page {self.page}")
        bits.append(f"chunk {self.chunk_index + 1}")
        return " · ".join(bits)

    def excerpt(self, limit: int = 320) -> str:
        text = " ".join(self.text.split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"


@dataclass
class RetrievedChunk:
    """A chunk plus its relevance score for one specific query."""

    chunk: Chunk
    score: float
    rank: int = 0
    matched_terms: list[str] = field(default_factory=list)
    duplicate_of: str | None = None

    @property
    def citation_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self) -> dict[str, Any]:
        payload = self.chunk.to_dict()
        payload.update({"score": self.score, "rank": self.rank, "matched_terms": self.matched_terms})
        return payload


# --------------------------------------------------------------------------
# Conversations
# --------------------------------------------------------------------------


@dataclass
class Citation:
    """A validated reference from an answer back to a retrieved chunk."""

    chunk_id: str
    document_id: str
    filename: str
    collection: str
    locator: str
    excerpt: str
    score: float
    synthetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Message:
    message_id: str
    session_id: str
    role: MessageRole
    content: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Conversation:
    session_id: str
    title: str = "New conversation"
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


@dataclass
class Answer:
    """The full result of one question, ready to render."""

    question: str
    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    mode: str = "extractive"  # extractive | llm
    model_id: str = ""
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
    follow_up: str | None = None

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer_text": self.answer_text,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [r.to_dict() for r in self.retrieved],
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "mode": self.mode,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
            "invalid_citations": self.invalid_citations,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }


# --------------------------------------------------------------------------
# Requirement matrix
# --------------------------------------------------------------------------


@dataclass
class RequirementEvidence:
    """One job requirement mapped against one candidate's documents."""

    requirement: str
    status: EvidenceStatus
    rationale: str
    citations: list[Citation] = field(default_factory=list)
    best_score: float = 0.0

    @property
    def status_label(self) -> str:
        return {
            "supported": "Supported",
            "partial": "Partial",
            "not_found": "Not found in supplied documents",
        }[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "status": self.status,
            "status_label": self.status_label,
            "rationale": self.rationale,
            "best_score": self.best_score,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class EvidenceMatrix:
    """Requirement-by-requirement evidence for one candidate / job pair.

    Deliberately carries **no** aggregate score, ranking or verdict — the PRD
    forbids them.  Only per-requirement evidence statuses.
    """

    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    rows: list[RequirementEvidence] = field(default_factory=list)
    generated_at: str = field(default_factory=utcnow)
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {"supported": 0, "partial": 0, "not_found": 0}
        for row in self.rows:
            out[row.status] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "job_id": self.job_id,
            "job_title": self.job_title,
            "generated_at": self.generated_at,
            "counts": self.counts(),
            "rows": [r.to_dict() for r in self.rows],
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass
class EvaluationCase:
    case_id: str
    query: str
    category: str
    expected_source_ids: list[str] = field(default_factory=list)
    expected_answer_points: list[str] = field(default_factory=list)
    should_abstain: bool = False
    expected_blocked: bool = False
    notes: str = ""
    candidate_id: str | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    case_id: str
    category: str
    query: str
    hit: bool
    recall: float
    precision_at_k: float
    citation_valid: bool
    citation_count: int
    abstained: bool
    abstention_correct: bool
    blocked: bool
    blocked_correct: bool
    latency_ms: float
    passed: bool
    failure_reason: str | None = None
    retrieved_source_ids: list[str] = field(default_factory=list)
    answer_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    run_id: str
    created_at: str
    config: dict[str, Any]
    results: list[CaseResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config,
            "metrics": self.metrics,
            "failures": self.failures,
            "duration_seconds": self.duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }


__all__ = [
    "Answer",
    "CaseResult",
    "Chunk",
    "Citation",
    "Conversation",
    "Document",
    "EvaluationCase",
    "EvaluationReport",
    "EvidenceMatrix",
    "EvidenceStatus",
    "Message",
    "MessageRole",
    "RequirementEvidence",
    "RetrievedChunk",
    "new_id",
    "utcnow",
]
