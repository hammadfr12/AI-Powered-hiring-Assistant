"""Application service layer — the single seam between the UI and the pipeline.

Every Streamlit view talks to :class:`HiringAssistant`, never to the storage or
retrieval layers directly.  That keeps the UI free of orchestration logic, makes
the whole application drivable from a test or a script without Streamlit, and
means there is exactly one place where a session is created and cached.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings, load_settings
from .errors import HiringAssistantError
from .evaluation.reporting import report_to_csv, report_to_markdown
from .evaluation.runner import EvaluationRunner, RunnerConfig
from .evidence.profiles import (
    CandidateRecord,
    JobRecord,
    ProfileSummariser,
    ProfileSummary,
    discover_candidates,
    discover_jobs,
)
from .evidence.requirements import RequirementMatrixBuilder, extract_requirements
from .generation.generator import AnswerGenerator, GenerationResult
from .ingestion.pipeline import IngestionPipeline, IngestReport, SourceFile
from .models import EvaluationReport, EvidenceMatrix, Message, new_id
from .retrieval.embeddings import EmbeddingProvider, build_embedding_provider
from .retrieval.search import SearchFilters, SearchResult, SearchService
from .storage.database import MetadataStore
from .storage.vector_store import VectorStore
from .synthetic import ensure_corpus, load_corpus_files


@dataclass
class SessionState:
    """Everything the app needs to render, assembled once per rerun."""

    settings: Settings
    store: MetadataStore
    embedder: EmbeddingProvider
    vector_store: VectorStore
    search: SearchService
    generator: AnswerGenerator
    pipeline: IngestionPipeline
    index_loaded: bool
    index_message: str
    warnings: list[str]


class HiringAssistant:
    """Facade over the whole RAG application."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        load_index: bool = True,
    ) -> None:
        self.settings = settings or load_settings()
        self.settings.validate()

        self.store = MetadataStore(self.settings.db_path)
        self.embedder = build_embedding_provider(self.settings)
        self.vector_store = VectorStore(
            self.settings.index_dir, dimension=self.embedder.dimension
        )
        self.search_service = SearchService(self.settings, self.store, self.embedder, self.vector_store)
        self.generator = AnswerGenerator(self.settings, self.search_service)
        self.pipeline = IngestionPipeline(
            self.settings, self.store, self.embedder, self.vector_store
        )
        self.summariser = ProfileSummariser(
            self.settings, self.search_service, llm=self.generator.llm
        )
        self.matrix_builder = RequirementMatrixBuilder(
            self.settings, self.search_service, llm=self.generator.llm
        )

        self.warnings: list[str] = list(self.settings.fallback_notes)
        self.index_message = ""
        self.index_loaded = False
        if load_index:
            self.load_index()

    # ------------------------------------------------------------------
    # Session / index lifecycle
    # ------------------------------------------------------------------

    def load_index(self, *, strict: bool = False) -> bool:
        """Try to attach the persisted index.  Never raises for the UI."""
        if not self.vector_store.exists():
            self.index_loaded = False
            self.index_message = "No index has been built yet."
            return False
        try:
            self.vector_store.load(
                fingerprint=self.settings.index_fingerprint(),
                strict_fingerprint=strict,
            )
            self.index_loaded = True
            mismatch = ""
            if self.vector_store.metadata and self.vector_store.metadata.corpus_fingerprint.startswith("MISMATCH"):
                mismatch = " " + self.vector_store.metadata.corpus_fingerprint
            self.index_message = (
                f"Index loaded: {self.vector_store.size:,} passages "
                f"({self.vector_store.backend}).{mismatch}"
            )
            if mismatch:
                self.warnings.append(
                    "The saved index was built with a different configuration. Search results may be "
                    "inaccurate — rebuild the index from the Knowledge Base view."
                )
            return True
        except HiringAssistantError as exc:
            self.index_loaded = False
            self.index_message = exc.message
            self.warnings.append(f"{exc.message} {exc.remedy}")
            return False

    def build_index_from_disk(self) -> IngestReport:
        """Build the index from whatever is already stored in SQLite."""
        return self.pipeline.rebuild_all()

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def ensure_synthetic_corpus(self, *, force_regenerate: bool = False) -> bool:
        """Write the synthetic corpus to disk if missing or stale."""
        before = set(Path(self.settings.synthetic_dir).rglob("*")) if Path(self.settings.synthetic_dir).exists() else set()
        ensure_corpus(self.settings.synthetic_dir, force=force_regenerate)
        after = set(Path(self.settings.synthetic_dir).rglob("*"))
        return before != after

    def load_synthetic_corpus(
        self,
        *,
        force: bool = False,
        progress: Callable[[float, str], None] | None = None,
    ) -> IngestReport:
        """Generate (if needed) and ingest the bundled synthetic corpus."""
        self.ensure_synthetic_corpus()
        entries = load_corpus_files(self.settings.synthetic_dir)
        sources = [
            SourceFile(filename=name, data=data, collection=collection, origin="synthetic")
            for name, data, collection in entries
        ]
        if not sources:
            return IngestReport(
                warnings=[
                    "No corpus files were found on disk. Run scripts/build_corpus.py to regenerate them."
                ]
            )
        return self.pipeline.ingest_many(sources, force=force, progress=progress)

    def ingest_uploads(
        self,
        files: Sequence[tuple[str, bytes]],
        *,
        collection: str = "uploads",
        progress: Callable[[float, str], None] | None = None,
    ) -> IngestReport:
        """Ingest user-uploaded files."""
        sources = [
            SourceFile(filename=name, data=data, collection=collection, origin="upload")
            for name, data in files
        ]
        return self.pipeline.ingest_many(sources, progress=progress)

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def inventory(self) -> dict[str, Any]:
        documents = self.store.list_documents()
        by_status: dict[str, int] = {}
        by_collection: dict[str, int] = {}
        for document in documents:
            by_status[document.status] = by_status.get(document.status, 0) + 1
            by_collection[document.collection] = by_collection.get(document.collection, 0) + 1

        warnings = [
            {"filename": d.filename, "document_id": d.document_id, "warnings": d.warnings}
            for d in documents
            if d.warnings
        ]
        failures = [
            {"filename": d.filename, "document_id": d.document_id, "error": d.error}
            for d in documents
            if d.status == "failed"
        ]
        injected = [
            {
                "filename": d.filename,
                "document_id": d.document_id,
                "warnings": [w for w in d.warnings if "injection" in w.lower()],
            }
            for d in documents
            if any("injection" in w.lower() for w in d.warnings)
        ]

        return {
            "documents": documents,
            "document_count": len(documents),
            "indexed_count": by_status.get("indexed", 0),
            "failed_count": by_status.get("failed", 0),
            "by_status": by_status,
            "by_collection": by_collection,
            "chunk_count": self.store.count_chunks(),
            "chunks_by_collection": self.store.chunk_count_by_collection(),
            "warnings": warnings,
            "failures": failures,
            "injection_flags": injected,
            "index": self.index_health(),
        }

    def index_health(self) -> dict[str, Any]:
        health = self.vector_store.health(fingerprint=self.settings.index_fingerprint())
        health["loaded"] = self.vector_store.is_loaded
        health["chunks_in_store"] = self.store.count_chunks()
        health["indexed_documents"] = len(
            [d for d in self.store.list_documents() if d.status == "indexed"]
        )
        health["built_at"] = self.store.get_state("index_built_at")
        health["stale"] = bool(
            health["loaded"]
            and health["chunks_in_store"] != health.get("vectors", -1)
        )
        health["model_matches"] = health.get("matches_configuration", True)
        return health

    def remove_document(self, document_id: str) -> dict[str, Any]:
        return self.pipeline.remove_document(document_id)

    def reindex_document(self, document_id: str) -> Any:
        return self.pipeline.reindex_document(document_id)

    def rebuild_index(
        self, progress: Callable[[float, str], None] | None = None
    ) -> IngestReport:
        report = self.pipeline.rebuild_all(progress=progress)
        self.load_index()
        return report

    def reset_all(self) -> None:
        """Delete every document, chunk and vector."""
        self.store.delete_all_documents()
        self.vector_store.delete_files()
        self.store.clear_state()
        self.index_loaded = False
        self.index_message = "Knowledge base cleared."

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> GenerationResult:
        """Answer a question, persisting the turn when a session is supplied."""
        history: list[tuple[str, str]] = []
        if session_id:
            self.store.ensure_conversation(session_id)
            history = [
                (m.role, m.content)
                for m in self.store.get_messages(session_id, limit=8)
                if m.role in {"user", "assistant"}
            ]

        result = self.generator.answer(
            question,
            filters=filters,
            history=history,
            top_k=top_k,
            min_relevance=min_relevance,
        )

        if session_id:
            self.store.add_message(
                Message(
                    message_id=new_id("msg"),
                    session_id=session_id,
                    role="user",
                    content=question,
                )
            )
            self.store.add_message(
                Message(
                    message_id=new_id("msg"),
                    session_id=session_id,
                    role="assistant",
                    content=result.answer.answer_text,
                    cited_chunk_ids=[c.chunk_id for c in result.answer.citations],
                    metadata={
                        "abstained": result.answer.abstained,
                        "blocked": result.answer.blocked,
                        "mode": result.answer.mode,
                        "latency_ms": result.answer.latency_ms,
                    },
                )
            )
            existing = self.store.get_messages(session_id, limit=1)
            if len(existing) <= 2 and question:
                self.store.rename_conversation(session_id, question.strip()[:60])

        return result

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        min_relevance: float | None = None,
    ) -> SearchResult:
        return self.search_service.search(
            query, top_k=top_k, filters=filters, min_relevance=min_relevance
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def history(self, session_id: str) -> list[Message]:
        return self.store.get_messages(session_id)

    def clear_history(self, session_id: str) -> int:
        return self.store.clear_conversation(session_id)

    # ------------------------------------------------------------------
    # Candidate explorer
    # ------------------------------------------------------------------

    def candidates(self) -> list[CandidateRecord]:
        return discover_candidates(self.store)

    def jobs(self) -> list[JobRecord]:
        return discover_jobs(self.store)

    def summarise_candidate(
        self, candidate: CandidateRecord, *, use_llm: bool = False
    ) -> ProfileSummary:
        return self.summariser.summarise(candidate, use_llm=use_llm)

    def build_matrix(
        self,
        candidate: CandidateRecord,
        job: JobRecord,
        *,
        use_llm: bool = False,
        max_requirements: int = 14,
    ) -> EvidenceMatrix:
        return self.matrix_builder.build(
            candidate_id=candidate.candidate_id,
            candidate_name=candidate.name,
            job_id=job.job_id,
            job_title=job.title,
            job_text=job.text,
            candidate_document_ids=[candidate.document_id],
            use_llm=use_llm,
            max_requirements=max_requirements,
        )

    def job_requirements(self, job: JobRecord) -> list[str]:
        return [r.text for r in extract_requirements(job.text)]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        config: RunnerConfig | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> EvaluationReport:
        runner = EvaluationRunner(self.settings, self.store, self.search_service, self.generator)
        return runner.run(config=config, progress=progress)

    def latest_evaluation(self) -> dict[str, Any] | None:
        return self.store.latest_evaluation_run()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_answer_markdown(self, result: GenerationResult) -> str:
        answer = result.answer
        lines = [
            "# Cited Answer",
            "",
            f"**Question:** {answer.question}",
            "",
            f"**Mode:** {answer.mode} (`{answer.model_id}`)",
            f"**Latency:** {answer.latency_ms:,.0f} ms",
            f"**Abstained:** {'yes' if answer.abstained else 'no'}",
            "",
            "## Answer",
            "",
            answer.answer_text,
            "",
        ]
        if answer.citations:
            lines += ["## Sources", ""]
            for index, citation in enumerate(answer.citations, start=1):
                origin = "synthetic demo data" if citation.synthetic else "uploaded document"
                lines.append(
                    f"{index}. **{citation.filename}** — {citation.locator} "
                    f"(`{citation.collection}`, {origin}, relevance {citation.score:.3f})"
                )
                lines.append("")
                lines.append(f"   > {citation.excerpt}")
                lines.append("")
        if answer.warnings:
            lines += ["## Notes", ""]
            lines += [f"- {w}" for w in answer.warnings]
            lines.append("")
        lines += [
            "---",
            "",
            "_Generated by the AI-Powered Hiring Assistant. This output retrieves and summarises "
            "evidence from the supplied documents. It is not a hiring recommendation, and the "
            "synthetic corpus contains no real candidate data._",
            "",
        ]
        return "\n".join(lines)

    def export_matrix_markdown(self, matrix: EvidenceMatrix) -> str:
        lines = [
            f"# Requirement Evidence — {matrix.candidate_name}",
            "",
            f"**Job:** {matrix.job_title} (`{matrix.job_id}`)",
            f"**Candidate:** {matrix.candidate_name} (`{matrix.candidate_id}`)",
            f"**Generated:** {matrix.generated_at}",
            "",
            "> This matrix shows document evidence only. It contains no suitability score, no "
            "ranking and no hire/reject recommendation. \"Not found in supplied documents\" means "
            "no evidence was located in the available documents — it is not a statement that the "
            "candidate lacks the skill.",
            "",
            "## Summary of statuses",
            "",
            "| Status | Count |",
            "| --- | --- |",
        ]
        for status, count in matrix.counts().items():
            lines.append(f"| {status.replace('_', ' ').title()} | {count} |")
        lines.append("")

        for index, row in enumerate(matrix.rows, start=1):
            lines.append(f"## {index}. {row.requirement}")
            lines.append("")
            lines.append(f"**Status:** {row.status_label}")
            lines.append("")
            lines.append(row.rationale)
            lines.append("")
            if row.citations:
                lines.append("**Evidence**")
                lines.append("")
                for citation in row.citations:
                    origin = "synthetic demo data" if citation.synthetic else "uploaded document"
                    lines.append(
                        f"- `{citation.filename}` — {citation.locator} "
                        f"({origin}, relevance {citation.score:.3f})"
                    )
                    lines.append(f"  > {citation.excerpt}")
                lines.append("")
            else:
                lines.append("_No supporting passage was retrieved for this requirement._")
                lines.append("")

        if matrix.warnings:
            lines += ["## Notes", ""]
            lines += [f"- {w}" for w in matrix.warnings]
            lines.append("")

        lines += [
            "---",
            "",
            "_Evidence review is a human task. This document supports that review; it does not "
            "replace it._",
            "",
        ]
        return "\n".join(lines)

    def export_matrix_csv(self, matrix: EvidenceMatrix) -> str:
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "candidate_id", "candidate_name", "job_id", "job_title",
                "requirement", "status", "rationale", "best_score",
                "evidence_count", "evidence_sources",
            ]
        )
        for row in matrix.rows:
            writer.writerow(
                [
                    matrix.candidate_id,
                    matrix.candidate_name,
                    matrix.job_id,
                    matrix.job_title,
                    row.requirement,
                    row.status,
                    row.rationale,
                    f"{row.best_score:.4f}",
                    len(row.citations),
                    "; ".join(f"{c.filename} [{c.locator}]" for c in row.citations),
                ]
            )
        return buffer.getvalue()

    def export_evaluation_markdown(self, report: EvaluationReport) -> str:
        return report_to_markdown(report)

    def export_evaluation_csv(self, report: EvaluationReport) -> str:
        return report_to_csv(report)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Everything the Settings view shows.  Never includes secrets."""
        from .config import EMBEDDING_MODELS, GENERATION_MODELS

        health = self.store.health()
        return {
            "settings": self.settings.public_summary(),
            "embedding": self.embedder.describe(),
            "generation": self.generator.health(),
            "store": health,
            "index": self.index_health(),
            "warnings": list(dict.fromkeys(self.warnings)),
            "available_embedding_models": {
                key: spec["label"] for key, spec in EMBEDDING_MODELS.items()
            },
            "available_generation_models": {
                key: spec["label"] for key, spec in GENERATION_MODELS.items()
            },
        }


__all__ = ["HiringAssistant", "SessionState"]
